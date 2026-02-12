from __future__ import annotations

import shutil
from pathlib import Path
from typing import NotRequired, TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import collect_files, move_to_dir

_DOCX_PATTERN = "*.docx"
_OG_DIRNAME = "_og"
_WORD_PDF_FORMAT = 17


class UserResponse(TypedDict):
    value: str
    recursive: NotRequired[bool]


def _start_word():
    """Start Word app hidden in the background with suppressed alerts."""
    from win32com import client  # type: ignore

    word_obj = client.Dispatch("Word.Application")
    word_obj.Visible = False
    word_obj.DisplayAlerts = 0
    return word_obj


def _export_pdf(document, out_file: Path) -> None:
    """Export Word document to PDF."""
    document.ExportAsFixedFormat(str(out_file), _WORD_PDF_FORMAT)


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_path = Path(ctx.target_path)

    if ctx.target_kind == "directory":
        payload = api.request_input_json(
            "Options for Word to PDF conversion",
            id="convert_word_options",
            fields=[
                {
                    "id": "recursive",
                    "type": "bool",
                    "label": "Recursive",
                    "default": False,
                }
            ],
            show_text_input=False,
            payload_type=UserResponse,
        )
        recursive = payload.get("recursive", False)
    else:
        confirm = api.confirm(
            "This will convert the selected file(s) to PDF and move the .docx "
            "files into an '_og' folder. Continue?",
            id="convert_word_confirm",
        )
        if not confirm:
            api.emit_result({"_title": "Operation Cancelled"})
            return
        recursive = False

    docx_files = collect_files(
        target_path,
        _DOCX_PATTERN,
        recursive=recursive,
        check_cancel=api.check_cancel,
    )
    total_files = len(docx_files)

    if ctx.environment["host"]["platform"] != "win32":
        api.log(
            "warn",
            "Word conversion requires Windows (pywin32). Running dry mode only.",
        )
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Dry Run Complete",
                "Dry Run": True,
                "Target": str(target_path),
                "Files Found": total_files,
            }
        )
        return

    if total_files == 0:
        api.log("warn", "No Word files found.")
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: No Word Files Found",
                "Target": str(target_path),
                "Files Found": 0,
            }
        )
        return

    api.log("info", f"Word to PDF | Files found={total_files}")

    try:
        word_window = _start_word()
    except Exception:
        gen_py = Path(Path.home(), "AppData", "Local", "Temp", "gen_py")
        shutil.rmtree(gen_py, ignore_errors=True)
        word_window = _start_word()

    converted = 0
    failures: list[str] = []

    try:
        for index, file in enumerate(docx_files, start=1):
            api.check_cancel()
            document = None
            destination = file.with_suffix(".pdf")
            converted_this = False
            try:
                if destination.exists():
                    destination.unlink()
                document = word_window.Documents.Open(str(file), ReadOnly=True)
                _export_pdf(document, destination)
                converted_this = True
                converted += 1
            except Exception as exc:
                api.log("warn", f"Failed to process '{file}': {exc}")
                failures.append(f"{file}: {exc}")
            finally:
                if document is not None:
                    document.Close(SaveChanges=False)
            if converted_this:
                try:
                    og_dir = file.parent / _OG_DIRNAME
                    move_to_dir(file, og_dir, use_shutil=True)
                except Exception as exc:
                    api.log("warn", f"Failed to archive '{file}': {exc}")
                    failures.append(f"{file}: archive failed: {exc}")
            api.progress(
                current=index,
                total=total_files,
                unit="files",
                message=f"Converting files in {file.parent.name}",
            )
    finally:
        word_window.Application.Quit()

    api.emit_result(
        {
            "_title": "Word Conversion Summary",
            "Files Found": total_files,
            "Files Converted": converted,
            "Failures": failures,
        }
    )
    api.exit(code=0)


if __name__ == "__main__":
    main()
