from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import NotRequired, TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import (
    build_destination,
    generate_document_id,
    set_xmp_mm_metadata_inplace,
)


class UserResponse(TypedDict):
    value: str
    autofitcolumn: bool
    portrait: bool
    recursive: NotRequired[bool]


def _start_excel():
    """Start Excel app hidden in the background with suppressed alerts."""
    from win32com import client  # type: ignore

    xl_obj = client.Dispatch("Excel.Application")
    xl_obj.Visible = False
    xl_obj.DisplayAlerts = False
    return xl_obj


def _get_print_area(worksheet, include_formulas=False):
    """
    Determine print area as A1:<last_col><last_row>, using Excel's Find("*") approach.
    Works for .xls/.xlsx via COM.

    include_formulas:
        False -> LookIn=xlValues (what user sees)
        True  -> LookIn=xlFormulas (treat formula cells as used even if result is "")
    """
    # Excel constants (hardcoded)
    xl_values = -4163
    xl_formulas = -4123
    xl_by_rows = 1
    xl_by_columns = 2
    xl_previous = 2
    xl_part = 2  # LookAt:=xlPart

    look_in = xl_formulas if include_formulas else xl_values

    # Optional: ensure values are current (if app is Manual calculation)
    # worksheet.Calculate()

    after = worksheet.Cells(1, 1)

    common_kwargs = dict(
        What="*",
        After=after,
        LookIn=look_in,
        LookAt=xl_part,
        MatchCase=False,
        SearchFormat=False,
    )

    last_row_cell = worksheet.Cells.Find(
        SearchOrder=xl_by_rows,
        SearchDirection=xl_previous,
        **common_kwargs,
    )
    last_col_cell = worksheet.Cells.Find(
        SearchOrder=xl_by_columns,
        SearchDirection=xl_previous,
        **common_kwargs,
    )

    if last_row_cell is None or last_col_cell is None:
        return "A1:A1"

    last_row = int(last_row_cell.Row)
    last_col = int(last_col_cell.Column)
    last_col_letter = _column_letter(last_col)
    return f"A1:{last_col_letter}{last_row}"


def _page_setup(worksheet, print_area):
    """Set default print to pdf page setup options."""
    worksheet.PageSetup.PrintHeadings = False
    worksheet.PageSetup.PaperSize = 9  # xlPaperA4
    worksheet.PageSetup.Zoom = False
    worksheet.PageSetup.Orientation = 2  # portrait:1, landscape:2
    worksheet.PageSetup.FitToPagesTall = False
    worksheet.PageSetup.FitToPagesWide = 1
    worksheet.PageSetup.PrintTitleColumns = False
    worksheet.PageSetup.PrintTitleRows = False
    worksheet.PageSetup.LeftHeader = ""
    worksheet.PageSetup.CenterHeader = ""
    worksheet.PageSetup.RightHeader = ""
    worksheet.PageSetup.PrintArea = print_area


def _column_letter(index: int) -> str:
    """Convert 1-based column index to Excel column letters."""
    if index < 1:
        return "A"
    letters: list[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _cleanup_sheet(worksheet) -> None:
    worksheet.Columns("L:N").Delete()
    worksheet.Columns("I:I").Delete()
    worksheet.Columns("E:E").AutoFit()
    worksheet.Rows.AutoFit()


def _sanitize_filename_component(value: str) -> str:
    cleaned = (
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    cleaned = re.sub(r"[<>:\\\"/|?!*]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Sheet"


def _extract_show_type(worksheet) -> str | None:
    raw = worksheet.Range("D12").Value
    if not raw:
        return None
    text = str(raw).strip()
    if ":" in text:
        _, value = text.split(":", 1)
        return value.strip()
    return text


def _extract_production_group(worksheet) -> str | None:
    for cell in ("P9", "O9"):
        cell_range = worksheet.Range(cell)
        if cell_range.MergeCells:
            cell_range = cell_range.MergeArea.Cells(1, 1)
        raw = cell_range.Value
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            _, value = text.split(":", 1)
            return value.strip()
        return text
    return None


def _normalize_show_type(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def _normalize_group_folder(value: str | None) -> str:
    if not value:
        return "_unknown"
    cleaned = re.sub(r"[<>:\\\"/|?!*]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if not cleaned:
        return "_unknown"
    return cleaned.replace(" ", "-")


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    confirm = api.confirm(
        "Before continuing, close all Excel windows/files. Continue?",
        default=False,
    )
    if not confirm:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Excel Conversion Canceled by User",
                "Info": "No file operations were performed.",
            }
        )
        return

    target = ctx.target_path

    if ctx.environment["host"]["platform"] != "win32":
        api.log(
            "warn",
            "Excel conversion requires Windows (win32com). Running dry mode only.",
        )
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Dry Run Complete",
                "Dry Run": True,
                "Target": str(target),
                "Recursive": False,
            }
        )
        return

    try:
        xl_window = _start_excel()
    except Exception:
        gen_py = Path(Path.home(), "AppData", "Local", "Temp", "gen_py")
        shutil.rmtree(gen_py, ignore_errors=True)
        xl_window = _start_excel()
    converted: list[str] = []
    failures: list[str] = []
    try:
        # coversion is slow, emit every iteration
        workbook = None
        try:
            workbook = xl_window.Workbooks.Open(str(target))
            sheet_total = int(workbook.Worksheets.Count)
            output_dir = target.parent / "via_converted"
            output_dir.mkdir(parents=True, exist_ok=True)
            skip_show_types = {"digital", "podcast"}
            for sheet_index in range(1, sheet_total + 1):
                worksheet = workbook.Worksheets(sheet_index)
                api.progress(current=sheet_index, total=sheet_total, unit="sheets")
                show_type = _normalize_show_type(_extract_show_type(worksheet))
                if show_type in skip_show_types:
                    api.log(
                        "info",
                        f"Skipping sheet '{worksheet.Name}' (Show Type: {show_type}).",
                    )
                    continue
                group_folder = _normalize_group_folder(
                    _extract_production_group(worksheet)
                )
                group_dir = output_dir / group_folder
                group_dir.mkdir(parents=True, exist_ok=True)
                _cleanup_sheet(worksheet)
                print_area = _get_print_area(worksheet)
                _page_setup(worksheet, print_area)
                document_id = generate_document_id()
                sheet_name = _sanitize_filename_component(worksheet.Name)
                out_base = f"via_{sheet_name}"
                out_path = build_destination(group_dir, out_base, ".pdf")
                worksheet.ExportAsFixedFormat(0, str(out_path))
                try:
                    set_xmp_mm_metadata_inplace(out_path, document_id)
                except Exception as exc:
                    api.log(
                        "warn", f"Failed to add XMP metadata to '{out_path}': {exc}"
                    )
                converted.append(str(out_path))
        except Exception as exc:
            api.log("warn", f"Failed to process '{target}': {exc}")
            failures.append(f"{target}: {exc}")
        finally:
            if workbook is not None:
                workbook.Close(SaveChanges=False)
    finally:
        xl_window.Quit()

    api.emit_result(
        {
            "_title": "Excel Conversion Summary",
            "Failures": failures,
            "Converted": f"{len(converted)} files",
        }
    )


if __name__ == "__main__":
    main()
