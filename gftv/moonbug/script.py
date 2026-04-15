from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import (
    build_destination,
    collect_files,
    move_to_dir,
)


class UserResponse(TypedDict):
    value: str
    autofitcolumn: bool
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
    xl_values = -4163
    xl_formulas = -4123
    xl_by_rows = 1
    xl_by_columns = 2
    xl_previous = 2
    xl_part = 2

    look_in = xl_formulas if include_formulas else xl_values
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


def _column_letter(index: int) -> str:
    """Convert 1-based column index to Excel column letters."""
    if index < 1:
        return "A"
    letters: list[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _page_setup(worksheet, print_area, autocolumn):
    """Set default print to pdf page setup options."""
    if autocolumn:
        worksheet.Columns.AutoFit()
    worksheet.PageSetup.Zoom = False
    worksheet.PageSetup.Orientation = 2  # landscape:2, portrait:1
    worksheet.PageSetup.FitToPagesTall = False
    worksheet.PageSetup.FitToPagesWide = 1
    worksheet.PageSetup.PrintTitleColumns = False
    worksheet.PageSetup.PrintTitleRows = False
    worksheet.PageSetup.PrintArea = print_area


def _get_outfile_from_cells(worksheet):
    """Gets pd title, ep title, and ep number/date from excel file contents."""
    column_numbers = {chr(v): k + 1 for k, v in enumerate(range(65, 91))}

    pd_cell, ep_cell, no_cell = "I4", "I6", "D15"

    pd_cell_row, pd_cell_col = int(pd_cell[1:]), int(column_numbers[pd_cell[0]])
    pd_title = _cell_text(worksheet.Cells(pd_cell_row, pd_cell_col).Value).upper()
    pd_title = _strip_illegal_filename_chars(pd_title)
    pd_title = _collapse_spaces(pd_title)

    ep_cell_row, ep_cell_col = int(ep_cell[1:]), int(column_numbers[ep_cell[0]])
    ep_title = _cell_text(worksheet.Cells(ep_cell_row, ep_cell_col).Value)
    ep_title = ep_title.replace("\u2019", "'")
    ep_title = _strip_illegal_filename_chars(ep_title)
    ep_title = _collapse_spaces(ep_title)
    ep_title = _titlecase(ep_title) if ep_title else None

    no_cell_row, no_cell_col = int(no_cell[1:]), int(column_numbers[no_cell[0]])
    ep_number = _cell_text(worksheet.Cells(no_cell_row, no_cell_col).Value)
    ep_number = _strip_illegal_filename_chars(ep_number)
    ep_number = _collapse_spaces(ep_number)
    ep_number = f"{ep_number} Vrsn" if ep_number else "None Vrsn"

    return _escape_file_name(f"{pd_title}   {ep_title}  {ep_number}")


def _titlecase(text):
    """Convert string to title case; adjusting for apostrophe."""
    return re.sub(
        r"[A-Za-z]+('[A-Za-z]+)?", lambda word: word.group(0).capitalize(), text
    )


def _cell_text(value) -> str:
    """Normalize worksheet values to a trimmed string."""
    if value is None:
        return ""
    return str(value).strip()


def _collapse_spaces(text: str) -> str:
    """Reduce internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_illegal_filename_chars(text: str) -> str:
    """Replace illegal filename characters with spaces for later collapsing."""
    return re.sub(r"[<>:\\\"/|?!*]", " ", text)


def _escape_file_name(file_name):
    """Escape illegal characters from file name."""
    return re.sub(r"[<>:\\\"/|?!*]", "", file_name)


def _export_pdf(worksheet, out_file):
    """Export Excel worksheet to pdf."""
    # ExportAsFixedFormat is a method on the worksheet, not the workbook.
    worksheet.ExportAsFixedFormat(0, str(out_file))


def _is_in_named_dir(path: Path, dir_name: str) -> bool:
    return any(parent.name == dir_name for parent in path.parents)


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    confirm = api.confirm(
        "Before continuing, close all Excel windows/files. "
        "For best results, run a test first (enable Test mode). Continue?",
        default=False,
    )
    if not confirm:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Moonbug Conversion Canceled by User",
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
        "Options for Moonbug Excel to PDF script",
        id="moonbug_excel_to_pdf_options",
        fields=[
            *([recursive_field] if ctx.target_kind == "directory" else []),
            {
                "id": "autofitcolumn",
                "type": "bool",
                "label": "Autofit columns",
                "default": True,
            },
        ],
        show_text_input=False,
        payload_type=UserResponse,
    )

    recursive = payload.get("recursive", False)
    autofit_column = payload["autofitcolumn"]

    root_path = ctx.target_path
    xl_files = collect_files(
        root_path,
        "*.xls*",
        recursive=recursive,
        check_cancel=api.check_cancel,
    )
    total_files = len(xl_files)

    if ctx.environment["host"]["platform"] != "win32":
        api.log(
            "warn",
            "Moonbug Excel conversion requires Windows (win32com). Running dry mode only.",
        )
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Dry Run Complete",
                "Dry Run": True,
                "Target": str(root_path),
                "Autofit Column": autofit_column,
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
                "Target": str(root_path),
                "Files Found": 0,
            }
        )
        return

    api.log("info", f"Moonbug Excel to PDF | Files found={total_files}")

    try:
        xl_window = _start_excel()
    except Exception:
        gen_py = Path(Path.home(), "AppData", "Local", "Temp", "gen_py")
        shutil.rmtree(gen_py, ignore_errors=True)
        xl_window = _start_excel()
    current_workbook: list[Any | None] = [None]

    def _cleanup() -> None:
        workbook = current_workbook[0]
        if workbook is not None:
            try:
                workbook.Close(SaveChanges=False)
            except Exception:
                pass
        try:
            xl_window.Quit()
        except Exception:
            pass

    api.register_cleanup(_cleanup)

    base_name_map_by_dir: dict[Path, dict[str, Path | None]] = {}
    moved_to_check: set[Path] = set()
    moved_to_none: set[Path] = set()
    converted: list[str] = []
    failures: list[str] = []
    try:
        for index, file in enumerate(xl_files, start=1):
            api.check_cancel()
            # conversion is slow, emit every iteration
            workbook = None
            destination = None
            base_name = None
            base_pdf = None
            collision = False
            preexisting_base_pdf = False
            is_none_vrsn = False
            parent_dir = file.parent
            check_dir = parent_dir / "_check"
            none_dir = parent_dir / "_none_vrsn"
            base_name_map = base_name_map_by_dir.setdefault(parent_dir, {})
            try:
                workbook = xl_window.Workbooks.Open(str(file), UpdateLinks=0)
                current_workbook[0] = workbook
                worksheet = workbook.Worksheets(1)
                print_area = _get_print_area(worksheet)
                _page_setup(worksheet, print_area, autofit_column)
                base_name = _get_outfile_from_cells(worksheet)
                is_none_vrsn = bool(base_name and "None Vrsn" in base_name)
                base_pdf = parent_dir / f"{base_name}.pdf"
                preexisting_base_pdf = base_pdf.exists()
                collision = base_name in base_name_map or preexisting_base_pdf
                destination = build_destination(
                    parent_dir,
                    base_name,
                    ".pdf",
                    force_suffix=collision,
                )
                _export_pdf(worksheet, destination)
                converted.append(str(destination))
                api.progress(
                    current=index,
                    total=total_files,
                    unit="files",
                    message=f"Converting files in {parent_dir.name}",
                )
                if base_name not in base_name_map:
                    base_name_map[base_name] = None if preexisting_base_pdf else file
            except Exception as exc:
                api.log("warn", f"Failed to process '{file}': {exc}")
                failures.append(f"{destination}: {exc}")
            finally:
                if workbook is not None:
                    workbook.Close(SaveChanges=False)
                current_workbook[0] = None
            if base_name and destination:
                if is_none_vrsn:
                    if destination.exists():
                        move_to_dir(destination, none_dir, use_shutil=True)
                    if file.exists() and file not in moved_to_none:
                        move_to_dir(file, none_dir, use_shutil=True)
                        moved_to_none.add(file)
                    if base_pdf is not None and base_pdf.exists():
                        move_to_dir(base_pdf, none_dir, use_shutil=True)
                    original_excel = base_name_map.get(base_name)
                    if (
                        original_excel is not None
                        and original_excel.exists()
                        and original_excel not in moved_to_none
                    ):
                        move_to_dir(original_excel, none_dir, use_shutil=True)
                        moved_to_none.add(original_excel)
                elif collision:
                    if destination.exists():
                        move_to_dir(destination, check_dir, use_shutil=True)
                    if file.exists() and file not in moved_to_check:
                        move_to_dir(file, check_dir, use_shutil=True)
                        moved_to_check.add(file)
                    if base_pdf is not None and base_pdf.exists():
                        move_to_dir(base_pdf, check_dir, use_shutil=True)
                    original_excel = base_name_map.get(base_name)
                    if (
                        original_excel is not None
                        and original_excel.exists()
                        and original_excel not in moved_to_check
                    ):
                        move_to_dir(original_excel, check_dir, use_shutil=True)
                        moved_to_check.add(original_excel)
    finally:
        xl_window.Application.Quit()

    for file in xl_files:
        if file in moved_to_check:
            continue
        if file in moved_to_none:
            continue
        if (
            _is_in_named_dir(file, "_check")
            or _is_in_named_dir(file, "_og")
            or _is_in_named_dir(file, "_none_vrsn")
        ):
            continue
        if not file.exists():
            continue
        og_dir = file.parent / "_og"
        move_to_dir(file, og_dir, use_shutil=True)

    api.emit_result(
        {
            "_title": "Moonbug Conversion Summary",
            "Failures": failures,
            "Converted": f"{len(converted)} files",
        }
    )
    api.exit(code=0)


if __name__ == "__main__":
    main()
