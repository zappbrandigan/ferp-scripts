from __future__ import annotations

import json
import platform
from pathlib import Path

from ferp.fscp.scripts import sdk


def _start_excel():
    """Start Excel app hidden in the background with suppressed alerts."""
    from win32com import client  # type: ignore

    xl_obj = client.Dispatch("Excel.Application")
    xl_obj.Visible = False
    xl_obj.DisplayAlerts = False
    return xl_obj


def _collect_excel_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    if recursive:
        return sorted(path for path in root.rglob("*.xls*") if path.is_file())
    return sorted(path for path in root.glob("*.xls*") if path.is_file())


def _build_destination(directory: Path, base: str, suffix: str) -> Path:
    candidate = directory / f"{base}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _export_pdf(workbook, out_file: Path) -> None:
    workbook.ExportAsFixedFormat(0, str(out_file))


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target = ctx.target_path
    if not target.exists():
        raise ValueError(f"Target does not exist: {target}")
    if target.is_file() and target.suffix.lower() not in {".xls", ".xlsx", ".xlsm"}:
        raise ValueError(f"Target must be an Excel file or directory: {target}")

    response = api.request_input(
        "Options for Convert Excel to PDF",
        id="convert_excel_options",
        fields=[
            {
                "id": "recursive",
                "type": "bool",
                "label": "Scan subdirectories",
                "default": False,
            },
            {"id": "test", "type": "bool", "label": "Test mode", "default": False},
        ],
        show_text_input=False,
    )
    if response is None:
        api.exit(code=1)
        return

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = {"value": response}

    recursive = bool(payload.get("recursive", True))
    is_test = bool(payload.get("test", False))

    xl_files = _collect_excel_files(target, recursive=recursive)
    total_files = len(xl_files)
    if total_files == 0:
        api.log("warn", "No Excel files found.")
        api.emit_result({"dry_run": False, "target": str(target), "files_found": 0})
        return

    if platform.system().lower() != "windows":
        api.log(
            "warn",
            "Excel conversion requires Windows (win32com). Running dry mode only.",
        )
        api.emit_result(
            {
                "dry_run": True,
                "target": str(target),
                "recursive": recursive,
                "test": is_test,
                "files_found": total_files,
            }
        )
        api.exit(code=0)
        return

    if is_test:
        xl_files = xl_files[:1]
        total_files = 1
        api.log("info", "Running in test mode: only processing first file.")
    else:
        api.log("info", f"Convert Excel to PDF | Files found={total_files}")

    xl_window = _start_excel()
    converted: list[str] = []
    failures: list[str] = []

    try:
        for index, file_path in enumerate(xl_files, start=1):
            api.progress(current=index, total=total_files, unit="files")
            workbook = None
            try:
                workbook = xl_window.Workbooks.Open(str(file_path))
                out_path = _build_destination(file_path.parent, file_path.stem, ".pdf")
                _export_pdf(workbook, out_path)
                converted.append(str(out_path))
            except Exception as exc:
                failures.append(f"{file_path}: {exc}")
            finally:
                if workbook is not None:
                    workbook.Close(SaveChanges=False)
    finally:
        xl_window.Quit()

    api.emit_result(
        {
            "dry_run": False,
            "converted": converted,
            "failures": failures,
            "files_found": total_files,
        }
    )


if __name__ == "__main__":
    main()
