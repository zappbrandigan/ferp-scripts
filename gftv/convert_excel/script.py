from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import build_destination, collect_files


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


def _get_print_area(worksheet):
    """Find last column and last row containing data to determine print area."""
    from win32com import client  # type: ignore

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
    worksheet.PageSetup.Zoom = False
    worksheet.PageSetup.Orientation = 2  # landscape:2, portrait:1
    worksheet.PageSetup.FitToPagesTall = False
    worksheet.PageSetup.FitToPagesWide = 1
    worksheet.PageSetup.PrintTitleColumns = False
    worksheet.PageSetup.PrintTitleRows = False
    worksheet.PageSetup.PrintArea = print_area


def _export_pdf(workbook, out_file: Path) -> None:
    workbook.ExportAsFixedFormat(0, str(out_file))


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
        "Options for Excel to PDF Conversion",
        id="convert_excel_options",
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
    autofit_colulmn = payload["autofitcolumn"]

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

    xl_window = _start_excel()
    converted: list[str] = []
    failures: list[str] = []

    try:
        for index, file_path in enumerate(xl_files, start=1):
            # coversion is slow, emit every iteration
            workbook = None
            try:
                workbook = xl_window.Workbooks.Open(str(file_path))
                out_path = build_destination(file_path.parent, file_path.stem, ".pdf")
                _export_pdf(workbook, out_path)
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
