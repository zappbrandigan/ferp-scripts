from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TypedDict

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import (
    build_archive_destination,
    build_destination,
    collect_files,
    load_settings,
    move_to_dir,
    save_settings,
)


class UserResponse(TypedDict):
    recursive: bool
    overwrite: bool


def _resolve_chrome_path(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> str | None:
    settings = load_settings(ctx)
    integrations = settings.get("integrations", {})
    chrome_config = (
        integrations.get("chrome", {}) if isinstance(integrations, dict) else {}
    )
    chrome_path = ""
    if isinstance(chrome_config, dict):
        chrome_path = str(chrome_config.get("path") or "").strip()

    if not chrome_path:
        prompt = "Path to Chrome (for PDF conversion)"
        chrome_path = api.request_input(
            prompt,
        ).strip()

    if not chrome_path:
        return None

    cleaned = chrome_path.strip().strip("\"'")
    resolved_path = cleaned.replace("\\ ", " ")
    candidate = Path(os.path.expanduser(resolved_path))
    if candidate.exists():
        if isinstance(integrations, dict):
            chrome_config = integrations.setdefault("chrome", {})
            if isinstance(chrome_config, dict):
                chrome_config["path"] = str(candidate)
                save_error = save_settings(ctx, settings)
                if save_error:
                    api.emit_result(
                        {
                            "_status": "warn",
                            "_title": "Settings Update Failed",
                            "Info": save_error,
                        }
                    )
        return str(candidate)

    api.emit_result(
        {
            "_status": "error",
            "_title": "Chrome Path Not Found",
            "Info": f"Chrome not found at: {candidate}",
        }
    )
    return None


def _convert_html_to_pdf(
    source: Path, destination: Path, chrome_path: str
) -> str | None:
    try:
        html = source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        html = source.read_text(encoding="latin-1")
    except Exception as exc:
        return f"Unable to read HTML: {exc}"

    temp_destination = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        if temp_destination.exists():
            temp_destination.unlink()
        temp_html = temp_destination.with_suffix(".html")
        temp_html.write_text(html, encoding="utf-8")
        command = [
            chrome_path,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--print-to-pdf={temp_destination}",
            temp_html.as_uri(),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode != 0:
            return f"Conversion failed: Chrome exit code {result.returncode}"
        temp_destination.replace(destination)
    except Exception as exc:
        if temp_destination.exists():
            temp_destination.unlink()
        return f"Conversion failed: {exc}"
    finally:
        temp_html = temp_destination.with_suffix(".html")
        if temp_html.exists():
            temp_html.unlink()

    return None


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path
    overwrite = False
    recursive = False
    if ctx.target_kind == "directory":
        options = api.request_input_json(
            "HTML to PDF Options",
            id="convert_html_to_pdf_options",
            fields=[
                {
                    "id": "recursive",
                    "type": "bool",
                    "label": "Recursive",
                    "default": False,
                }
            ],
            show_text_input=False,
        )
        recursive = bool(options.get("recursive", False))

    settings = UserResponse(
        recursive=recursive,
        overwrite=overwrite,
    )

    if ctx.target_kind == "file":
        html_files = [root] if root.suffix.lower() == ".html" else []
    else:
        html_files = collect_files(root, "*.html", settings["recursive"])
        html_files = [path for path in html_files if path.suffix.lower() == ".html"]
    if not html_files:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "No HTML Files Found",
                "Info": "Select a file or folder with .html files and try again.",
            }
        )
        return

    chrome_path = _resolve_chrome_path(ctx, api)
    if not chrome_path:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Chrome Path Required",
                "Info": "Provide a valid Chrome path to continue.",
            }
        )
        return

    converted = 0
    failures: list[str] = []
    moved = 0
    move_failures: list[str] = []

    total = len(html_files)
    for index, html_file in enumerate(html_files, start=1):
        api.progress(
            current=index,
            total=total,
            unit="files",
            message=f"Converting {html_file.name}",
        )
        if ctx.target_kind == "file":
            destination = build_destination(
                html_file.parent,
                html_file.stem,
                ".pdf",
                overwrite=False,
            )
        else:
            destination = build_destination(
                html_file.parent,
                html_file.stem,
                ".pdf",
                overwrite=settings["overwrite"],
            )
        error = _convert_html_to_pdf(html_file, destination, chrome_path)
        if error:
            failures.append(f"{html_file.name}: {error}")
        else:
            converted += 1
            try:
                archive_dir = html_file.parent / "_og"
                archive_destination = build_archive_destination(
                    archive_dir, html_file.name
                )
                move_to_dir(
                    html_file,
                    archive_dir,
                    base=archive_destination.stem,
                    use_shutil=True,
                )
                moved += 1
            except Exception as exc:
                move_failures.append(f"{html_file.name}: {exc}")

    result: dict[str, object] = {
        "_title": "HTML to PDF Complete",
        "Converted": converted,
        "Total Files": total,
        "Archived": moved,
    }
    if failures or move_failures:
        result["_status"] = "warn"
        errors = failures + move_failures
        result["Errors"] = errors[:10]
        if len(errors) > 10:
            result["Errors"] = errors[:10] + [f"...and {len(errors) - 10} more"]

    api.emit_result(result)


if __name__ == "__main__":
    main()
