from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict

from ferp.fscp.scripts import sdk


class UserResponse(TypedDict):
    recursive: bool
    overwrite: bool


def _collect_html_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    if recursive:
        return sorted(path for path in root.rglob("*.html") if path.is_file())
    return sorted(path for path in root.glob("*.html") if path.is_file())


def _build_destination(directory: Path, base: str, overwrite: bool) -> Path:
    candidate = directory / f"{base}.pdf"
    if overwrite or not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}.pdf"
        if not candidate.exists():
            return candidate
        counter += 1


def _build_archive_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    base = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _load_settings(settings_path: Path) -> dict[str, object]:
    if not settings_path.exists():
        return {}
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(settings_path: Path, payload: dict[str, object]) -> str | None:
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(payload, indent=4))
    except Exception as exc:
        return f"Unable to save settings: {exc}"
    return None


def _resolve_chrome_path(settings_path: Path, api: sdk.ScriptAPI) -> str | None:
    settings = _load_settings(settings_path)
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
                save_error = _save_settings(settings_path, settings)
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
    if not root.exists():
        api.emit_result(
            {
                "_status": "error",
                "_title": "Missing Path",
                "Info": "Select a file or directory and try again.",
            }
        )
        return

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
        if not options:
            api.emit_result(
                {
                    "_status": "warn",
                    "_title": "HTML Conversion Canceled",
                    "Info": "No files were modified.",
                }
            )
            return
        recursive = bool(options.get("recursive", False))

    settings = UserResponse(
        recursive=recursive,
        overwrite=overwrite,
    )

    if ctx.target_kind == "file":
        html_files = [root] if root.suffix.lower() == ".html" else []
    else:
        html_files = _collect_html_files(root, settings["recursive"])
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

    env_paths = ctx.environment.get("paths", {})
    settings_path_value = env_paths.get("settings_file")
    settings_path = Path(settings_path_value) if settings_path_value else None
    if settings_path is None:
        api.emit_result(
            {
                "_status": "error",
                "_title": "Missing Settings Path",
                "Info": "Settings file path is not available.",
            }
        )
        return

    chrome_path = _resolve_chrome_path(settings_path, api)
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
        api.progress(current=index, total=total, message=f"Converting {html_file.name}")
        if ctx.target_kind == "file":
            destination = _build_destination(html_file.parent, html_file.stem, False)
        else:
            destination = _build_destination(
                html_file.parent, html_file.stem, settings["overwrite"]
            )
        error = _convert_html_to_pdf(html_file, destination, chrome_path)
        if error:
            failures.append(f"{html_file.name}: {error}")
        else:
            converted += 1
            try:
                archive_dir = html_file.parent / "_og"
                archive_dir.mkdir(exist_ok=True)
                archive_destination = _build_archive_destination(
                    archive_dir, html_file.name
                )
                html_file.replace(archive_destination)
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
