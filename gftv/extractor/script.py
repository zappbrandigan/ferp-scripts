from __future__ import annotations

import os
import shutil
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Callable

import extract_msg

from ferp.fscp.scripts import sdk

SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
WINDOWS_MAX_PATH = 240
WINDOWS_INVALID_CHARS = set('<>:"/\\|?*')


def _sanitize_component(component: str, max_len: int | None) -> str:
    cleaned = "".join(
        "_" if (ch in WINDOWS_INVALID_CHARS or ord(ch) < 32) else ch
        for ch in component
    )
    cleaned = cleaned.rstrip(" .")
    if not cleaned:
        cleaned = "_"
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[-max_len:]
    return cleaned


def _fit_components(
    components: list[str], base_len: int, max_total: int | None
) -> list[str]:
    if max_total is None:
        return components

    sep_len = 1

    def _total_len(parts: list[str]) -> int:
        return base_len + sum(len(part) for part in parts) + sep_len * len(parts)

    total = _total_len(components)
    if total <= max_total:
        return components

    excess = total - max_total
    trimmed: list[str] = []
    for part in components:
        if excess > 0:
            reducible = max(len(part) - 1, 0)
            cut = min(excess, reducible)
            if cut:
                part = part[cut:]
                excess -= cut
        if not part:
            part = "_"
        trimmed.append(part)

    # If still too long, drop leading folders to keep the tail (unique IDs).
    while _total_len(trimmed) > max_total and len(trimmed) > 1:
        trimmed = trimmed[1:]

    # Final fallback: trim the remaining component even further.
    total = _total_len(trimmed)
    if total > max_total and trimmed:
        overflow = total - max_total
        part = trimmed[-1]
        if overflow >= len(part) - 1:
            part = part[-1:]
        else:
            part = part[overflow:]
        trimmed[-1] = part or "_"

    return trimmed


def _safe_extract_zip(
    archive: zipfile.ZipFile, dest_dir: Path, log: Callable[[str, str], None]
) -> None:
    base_len = len(str(dest_dir))
    max_total = WINDOWS_MAX_PATH if os.name == "nt" else None
    max_component = 120 if os.name == "nt" else None

    for info in archive.infolist():
        rel_path = PurePosixPath(info.filename)
        parts = [p for p in rel_path.parts if p not in ("", ".", "..")]
        if not parts:
            continue

        safe_parts = [_sanitize_component(p, max_component) for p in parts]
        safe_parts = _fit_components(safe_parts, base_len, max_total)
        dest_path = dest_dir.joinpath(*safe_parts)

        # Guard against zip-slip.
        try:
            dest_resolved = dest_path.resolve()
        except FileNotFoundError:
            dest_resolved = dest_path.parent.resolve() / dest_path.name
        dest_root = dest_dir.resolve()
        if dest_resolved != dest_root and dest_root not in dest_resolved.parents:
            log("warn", f"Skipping suspicious zip entry: {info.filename}")
            continue

        if info.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(dest_path, "wb") as target:
            shutil.copyfileobj(source, target)


def _extract_from_msg(
    msg_file: Path,
    output_dir: Path,
    check_cancel: Callable[[], None] | None = None,
) -> int:
    msg = extract_msg.Message(str(msg_file))
    attachments = list(msg.attachments)

    base_len = len(str(output_dir))
    max_total = WINDOWS_MAX_PATH if os.name == "nt" else None
    max_component = 120 if os.name == "nt" else None

    safe_stem = _sanitize_component(msg_file.stem, max_component)
    safe_stem = _fit_components([safe_stem], base_len, max_total)[0]
    email_output_dir = output_dir / safe_stem
    email_output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for attachment in attachments:
        if check_cancel is not None:
            check_cancel()
        name = attachment.longFilename or attachment.shortFilename
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        safe_name = _sanitize_component(name, max_component)
        safe_name = _fit_components(
            [safe_name], len(str(email_output_dir)), max_total
        )[0]
        try:
            attachment.save(customPath=email_output_dir, customFilename=safe_name)
        except TypeError:
            attachment.save(customPath=email_output_dir)
        saved += 1

    return saved


def _extract_from_eml(
    eml_file: Path,
    output_dir: Path,
    check_cancel: Callable[[], None] | None = None,
) -> int:
    with eml_file.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    base_len = len(str(output_dir))
    max_total = WINDOWS_MAX_PATH if os.name == "nt" else None
    max_component = 120 if os.name == "nt" else None

    safe_stem = _sanitize_component(eml_file.stem, max_component)
    safe_stem = _fit_components([safe_stem], base_len, max_total)[0]
    email_output_dir = output_dir / safe_stem
    email_output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for part in message.iter_attachments():
        if check_cancel is not None:
            check_cancel()
        name = part.get_filename()
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        safe_name = _sanitize_component(name, max_component)
        safe_name = _fit_components(
            [safe_name], len(str(email_output_dir)), max_total
        )[0]
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload_bytes = bytes(payload)
        elif hasattr(payload, "as_bytes"):
            payload_bytes = payload.as_bytes()
        else:
            payload_bytes = str(payload).encode("utf-8")
        (email_output_dir / safe_name).write_bytes(payload_bytes)
        saved += 1

    return saved


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_path = ctx.target_path
    if ctx.target_kind == "file" and target_path.suffix.lower() != ".zip":
        api.emit_result(
            {
                "_status": "error",
                "_title": "Unsupported File",
                "Info": "Select a .zip file or a directory and try again.",
            }
        )
        return

    output_dir = target_path.parent / f"{target_path.stem}_attachments"
    temp_dir = output_dir / "_unzipped"
    extracted = False

    def _cleanup() -> None:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if not extracted and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

    source_dir = target_path
    if ctx.target_kind == "file":
        api.register_cleanup(_cleanup)

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        api.log("info", f"Extracting {target_path.name} to {output_dir}")
        with zipfile.ZipFile(target_path, "r") as archive:
            _safe_extract_zip(archive, temp_dir, api.log)
        source_dir = temp_dir

    msg_files = list(source_dir.rglob("*.msg"))
    eml_files = list(source_dir.rglob("*.eml"))
    if not msg_files and not eml_files:
        api.log("warn", "No .msg or .eml files found in archive.")
        if ctx.target_kind == "file":
            shutil.rmtree(temp_dir, ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: No .msg or .eml Files Found",
                "Output": None,
                "Processed Messages": 0,
            }
        )
        return

    total_saved = 0
    total_messages = len(msg_files) + len(eml_files)
    current_index = 0

    for msg_file in msg_files:
        current_index += 1
        api.check_cancel()
        api.progress(
            current=current_index, total=total_messages, unit="messages", every=10
        )
        saved = _extract_from_msg(
            msg_file, output_dir, check_cancel=api.check_cancel
        )
        total_saved += saved
        api.log("info", f"{msg_file.name}: saved {saved} attachment(s)")

    for eml_file in eml_files:
        current_index += 1
        api.check_cancel()
        api.progress(
            current=current_index, total=total_messages, unit="messages", every=10
        )
        saved = _extract_from_eml(
            eml_file, output_dir, check_cancel=api.check_cancel
        )
        total_saved += saved
        api.log("info", f"{eml_file.name}: saved {saved} attachment(s)")

    extracted = True
    if ctx.target_kind == "file":
        shutil.rmtree(temp_dir, ignore_errors=True)

    api.emit_result(
        {
            "_title": "Attachment Extraction Finished",
            "Output Directory": str(output_dir),
            "Messages Processed": total_messages,
            "Attachments Extracted": total_saved,
        }
    )


if __name__ == "__main__":
    main()
