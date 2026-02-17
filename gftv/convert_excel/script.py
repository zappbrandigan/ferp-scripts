from __future__ import annotations

import shutil
from pathlib import Path
from typing import NotRequired, TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import build_destination, collect_files


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


def _page_setup(worksheet, print_area, autocolumn, portrait):
    """Set default print to pdf page setup options."""
    if autocolumn:
        worksheet.Columns.AutoFit()
    worksheet.PageSetup.PaperSize = 9  # xlPaperA4
    worksheet.PageSetup.Zoom = False
    worksheet.PageSetup.Orientation = 1 if portrait else 2  # portrait:1, landscape:2
    worksheet.PageSetup.FitToPagesTall = False
    worksheet.PageSetup.FitToPagesWide = 1
    worksheet.PageSetup.PrintTitleColumns = False
    worksheet.PageSetup.PrintTitleRows = False
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


def _select_worksheet(workbook, sheet_value: str | None):
    if not sheet_value:
        return workbook.ActiveSheet
    sheet_value = sheet_value.strip()
    if not sheet_value:
        return workbook.ActiveSheet
    if sheet_value.isdigit():
        return workbook.Worksheets(int(sheet_value))
    return workbook.Worksheets(sheet_value)


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

    recursive_field: sdk.BoolField = {
        "id": "recursive",
        "type": "bool",
        "label": "Recursive",
        "default": False,
    }
    payload = api.request_input_json(
        "Options for Excel to PDF Conversion (text: sheet name or index, optional)",
        id="convert_excel_options",
        default="",
        fields=[
            *([recursive_field] if ctx.target_kind == "directory" else []),
            {
                "id": "autofitcolumn",
                "type": "bool",
                "label": "Autofit columns",
                "default": False,
            },
            {
                "id": "portrait",
                "type": "bool",
                "label": "Portrait orientation",
                "default": False,
            },
        ],
        show_text_input=True,
        text_input_style="single_line",
        payload_type=UserResponse,
    )

    recursive = payload.get("recursive", False)
    autofit_colulmn = payload["autofitcolumn"]
    portrait = payload["portrait"]
    sheet_value = payload.get("value", "").strip()

    target = ctx.target_path
    xl_files = collect_files(
        target,
        "*.xls*",
        recursive=recursive,
        check_cancel=api.check_cancel,
    )
    total_files = len(xl_files)

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
                "Recursive": recursive,
                "Files Found": total_files,
            }
        )
        return

    if total_files == 0:
        api.log("warn", "No Excel files found.")
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: No Excel Files Found",
                **(
                    {"Dry Run": False}
                    if ctx.environment["host"]["platform"] == "darwin"
                    else {}
                ),
                "Target": str(target),
                "Files Found": 0,
            }
        )
        return

    api.log("info", f"Convert Excel to PDF | Files found={total_files}")

    try:
        xl_window = _start_excel()
    except Exception:
        gen_py = Path(Path.home(), "AppData", "Local", "Temp", "gen_py")
        shutil.rmtree(gen_py, ignore_errors=True)
        xl_window = _start_excel()
    converted: list[str] = []
    failures: list[str] = []

    try:
        for index, file_path in enumerate(xl_files, start=1):
            # coversion is slow, emit every iteration
            workbook = None
            try:
                workbook = xl_window.Workbooks.Open(str(file_path))
                try:
                    worksheet = _select_worksheet(workbook, sheet_value)
                except Exception:
                    raise RuntimeError(
                        f"Worksheet not found for '{sheet_value or 'active sheet'}'."
                    )
                print_area = _get_print_area(worksheet)
                _page_setup(worksheet, print_area, autofit_colulmn, portrait)
                out_path = build_destination(file_path.parent, file_path.stem, ".pdf")
                worksheet.ExportAsFixedFormat(0, str(out_path))
                converted.append(str(out_path))
                api.progress(current=index, total=total_files, unit="files")
            except Exception as exc:
                api.log("warn", f"Failed to process '{file_path}': {exc}")
                failures.append(f"{file_path}: {exc}")
            finally:
                if workbook is not None:
                    workbook.Close(SaveChanges=False)
    finally:
        xl_window.Quit()

    api.emit_result(
        {
            "_title": "Excel Conversion Summary",
            "Converted": converted,
            "Autofit Column": autofit_colulmn,
            "Failures": failures,
            "Files Found": total_files,
        }
    )


if __name__ == "__main__":
    main()
