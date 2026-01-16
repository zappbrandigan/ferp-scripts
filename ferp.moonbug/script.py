from __future__ import annotations

import re
import json
import shutil
import platform
from pathlib import Path

from ferp.fscp.scripts import sdk


def _start_excel():
    """Start Excel app hidden in the background with suppressed alerts."""
    from win32com import client # type: ignore
    xl_obj = client.Dispatch("Excel.Application")
    xl_obj.Visible = False
    xl_obj.DisplayAlerts = False
    return xl_obj


def _get_print_area(worksheet):
    """Find last column and last row containing data to determine print area."""
    from win32com import client # type: ignore
    column_letters = [chr(x) for x in range(65, 91)]
    col_last_row = {}

    for letter in column_letters:
        try:
            col_last_row[letter] = (
                worksheet.Cells(worksheet.Rows.Count, letter)
                .End(client.constants.xlShiftUp)
                .Row
            )
        except Exception:
            col_last_row[letter] = 1

    columns_with_data = [k for k, v in col_last_row.items() if v > 1]
    if not columns_with_data:
        return "A1"

    last_row = max(col_last_row.values())
    last_column = max(columns_with_data)

    return f"A1:{last_column}{last_row}"


def _page_setup(worksheet, print_area, autocolumn):
    """Set default print to pdf page setup options."""
    if autocolumn:
        worksheet.Columns.AutoFit()
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

    ep_cell_row, ep_cell_col = int(ep_cell[1:]), int(column_numbers[ep_cell[0]])
    ep_title = _cell_text(worksheet.Cells(ep_cell_row, ep_cell_col).Value)
    ep_title = ep_title.replace("\u2019", "'")
    ep_title = _titlecase(ep_title) if ep_title else None

    no_cell_row, no_cell_col = int(no_cell[1:]), int(column_numbers[no_cell[0]])
    ep_number = _cell_text(worksheet.Cells(no_cell_row, no_cell_col).Value)
    ep_number = f"{ep_number} Vrsn" if ep_number else "None Vrsn"

    return _escape_file_name(
        f"{pd_title}   {ep_title}  {ep_number}"
    )


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


def _escape_file_name(file_name):
    """Escape illegal characters from file name."""
    return re.sub(r"[<>:\\\"/|?!*]", "", file_name)


def _build_destination(
    directory: Path,
    base: str,
    suffix: str,
    force_suffix: bool = False,
) -> Path:
    candidate = directory / f"{base}{suffix}"
    if not candidate.exists() and not force_suffix:
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _export_pdf(worksheet, out_file):
    """Export Excel worksheet to pdf."""
    worksheet.ExportAsFixedFormat(0, str(out_file))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _move_to_dir(source: Path, destination_dir: Path) -> Path:
    if source.parent == destination_dir:
        return source
    _ensure_dir(destination_dir)
    destination = _build_destination(destination_dir, source.stem, source.suffix)
    shutil.move(str(source), destination)
    return destination


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
                "message": "Script cancelled",
            }
        )
        return

    response = api.request_input(
        "Options for Moonbug Excel to PDF script",
        id="moonbug_excel_to_pdf_options",
        fields=[
            {
                "id": "test", 
                "type": "bool", 
                "label": "Test mode", 
                "default": False
            },
            {
                "id": "autofitcolumn", 
                "type": "bool", 
                "label": "Autofit columns", 
                "default": True
            }
        ],
        show_text_input=False 
    )
    if response is None:
        api.exit(code=1)
        return
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = {"value": response}
    
    is_test = bool(payload.get("test", False))
    autofit_column = bool(payload.get("autofitcolumn", True))
    
    root_path = ctx.target_path
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Target must be a directory. Received: {root_path}")

    if platform.system().lower() != "windows":
        api.log(
            "warn",
            "Moonbug Excel conversion requires Windows (win32com). Running dry mode only.",
        )
        api.emit_result(
            {
                "dry_run": True,
                "target": str(root_path),
                "test": is_test,
                "autofitcolumn": autofit_column,
                "files_found": len(sorted(root_path.rglob("*.xls*"))),
            }
        )
        api.exit(code=0)
        return

    xl_files = sorted(root_path.rglob("*.xls*"))
    total_files = len(xl_files)
    if is_test:
        xl_files = xl_files[0:1]
        total_files = 1
        api.log("info", "Running in test mode: only processing first file.")
    else:
        api.log("info", f"Moonbug Excel to PDF | Files found={total_files}")

    xl_window = _start_excel()
    base_name_map_by_dir: dict[Path, dict[str, Path | None]] = {}
    moved_to_check: set[Path] = set()
    moved_to_none: set[Path] = set()

    try:
        for index, file in enumerate(xl_files, start=1):
            api.progress(current=index, total=total_files, unit="files")
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
                worksheet = workbook.Worksheets(1)
                print_area = _get_print_area(worksheet)
                _page_setup(worksheet, print_area, autofit_column)
                base_name = _get_outfile_from_cells(worksheet)
                is_none_vrsn = bool(base_name and "None Vrsn" in base_name)
                base_pdf = parent_dir / f"{base_name}.pdf"
                preexisting_base_pdf = base_pdf.exists()
                collision = base_name in base_name_map or preexisting_base_pdf
                destination = _build_destination(
                    parent_dir, base_name, ".pdf", force_suffix=collision
                )
                _export_pdf(worksheet, destination)
                if base_name not in base_name_map:
                    base_name_map[base_name] = None if preexisting_base_pdf else file
            except Exception as e:
                api.log("warn", f"Failed to process '{file}': {e}")
            finally:
                if workbook is not None:
                    workbook.Close(SaveChanges=False)
            if base_name and destination:
                if is_none_vrsn:
                    if destination.exists():
                        _move_to_dir(destination, none_dir)
                    if file.exists() and file not in moved_to_none:
                        _move_to_dir(file, none_dir)
                        moved_to_none.add(file)
                    if base_pdf is not None and base_pdf.exists():
                        _move_to_dir(base_pdf, none_dir)
                    original_excel = base_name_map.get(base_name)
                    if (
                        original_excel is not None
                        and original_excel.exists()
                        and original_excel not in moved_to_none
                    ):
                        _move_to_dir(original_excel, none_dir)
                        moved_to_none.add(original_excel)
                elif collision:
                    if destination.exists():
                        _move_to_dir(destination, check_dir)
                    if file.exists() and file not in moved_to_check:
                        _move_to_dir(file, check_dir)
                        moved_to_check.add(file)
                    if base_pdf is not None and base_pdf.exists():
                        _move_to_dir(base_pdf, check_dir)
                    original_excel = base_name_map.get(base_name)
                    if (
                        original_excel is not None
                        and original_excel.exists()
                        and original_excel not in moved_to_check
                    ):
                        _move_to_dir(original_excel, check_dir)
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
        _move_to_dir(file, og_dir)

    api.emit_result(
        {
            "dry_run": False,
            "target": str(root_path),
            "test": is_test,
            "autofitcolumn": autofit_column,
            "files_converted": total_files,
        }
    )
    api.exit(code=0)


if __name__ == "__main__":
    main()
