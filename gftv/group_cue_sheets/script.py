from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import collect_files, move_to_dir

PRODUCTION_DELIM = "   "
MAX_FOLDER_LENGTH = 60
ELLIPSIS = "..."
SUFFIX = ".pdf"

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*]')
_WHITESPACE_RE = re.compile(r"\s+")


class UserResponse(TypedDict):
    value: str
    mode: str


def _extract_xmp(reader: PdfReader) -> str | None:
    try:
        root = reader.trailer.get("/Root")
        if not root:
            return None
        if hasattr(root, "get_object"):
            root = root.get_object()
        metadata_ref = root.get("/Metadata")
        if not metadata_ref:
            return None
        if hasattr(metadata_ref, "get_object"):
            metadata_ref = metadata_ref.get_object()
        data = metadata_ref.get_data()
    except Exception:  # noqa: BLE001
        return None

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _parse_publishers(xmp: str) -> list[str]:
    match = re.search(r"(<x:xmpmeta\b.*?</x:xmpmeta>)", xmp, re.DOTALL)
    xml_payload = match.group(1) if match else xmp
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        return []

    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "ferp": "https://tulbox.app/ferp/xmp/1.0",
    }

    publishers: list[str] = []
    agreement_nodes = root.findall(".//ferp:agreements/rdf:Bag/rdf:li", ns)
    for agreement_node in agreement_nodes:
        for li in agreement_node.findall(".//ferp:publishers/rdf:Bag/rdf:li", ns):
            name = (li.text or "").strip()
            if not name or name in publishers:
                continue
            publishers.append(name)
    return publishers


def _truncate_name(value: str, max_len: int = MAX_FOLDER_LENGTH) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= len(ELLIPSIS):
        return value[:max_len]
    trimmed = value[: max_len - len(ELLIPSIS)].rstrip("-_.")
    return f"{trimmed}{ELLIPSIS}"


def _sanitize_segment(value: str, *, space_replacement: str) -> str:
    cleaned = value.strip().lower()
    cleaned = cleaned.replace("_", " ")
    cleaned = _WHITESPACE_RE.sub(space_replacement, cleaned)
    cleaned = _INVALID_NAME_CHARS.sub("", cleaned)
    if space_replacement:
        cleaned = re.sub(
            rf"{re.escape(space_replacement)}+", space_replacement, cleaned
        )
    cleaned = cleaned.strip(f"{space_replacement}._-")
    return cleaned


def _production_group(name: str) -> tuple[str | None, str | None]:
    stem = name.rsplit(".", 1)[0]
    delimiter_index = stem.find(PRODUCTION_DELIM)
    if delimiter_index <= 0:
        return None, "missing production delimiter"
    raw_production = stem[:delimiter_index].strip()
    if not raw_production:
        return None, "missing production title"
    folder = _sanitize_segment(raw_production, space_replacement="-")
    if not folder:
        return None, "invalid production title"
    return _truncate_name(folder), None


def _publisher_group(path: Path) -> tuple[str | None, str | None]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to read PDF ({exc})"

    if reader.is_encrypted:
        return None, "encrypted PDF"

    xmp = _extract_xmp(reader)
    if not xmp:
        return None, "missing XMP metadata"

    publishers = _parse_publishers(xmp)
    if not publishers:
        return None, "missing publisher metadata"

    sanitized: list[str] = []
    for publisher in publishers:
        cleaned = _sanitize_segment(publisher, space_replacement="-")
        if cleaned:
            sanitized.append(cleaned)

    if not sanitized:
        return None, "invalid publisher names"

    folder = "_".join(sanitized)
    folder = _truncate_name(folder)
    if not folder:
        return None, "invalid publisher names"

    return folder, None


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path

    payload = api.request_input_json(
        "Organize cue sheet PDFs",
        id="ferp_group_cue_sheets_mode",
        fields=[
            {
                "id": "mode",
                "type": "select",
                "label": "Group By",
                "options": ["production", "publishers"],
                "default": "production",
            }
        ],
        show_text_input=False,
        payload_type=UserResponse,
    )

    selection = (payload.get("mode") or "production").strip().lower()

    pdf_files = collect_files(
        root,
        f"*{SUFFIX}",
        recursive=False,
        check_cancel=api.check_cancel,
        case_sensitive=False,
    )

    if not pdf_files:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "No PDFs Found",
                "Directory": str(root),
            }
        )
        return

    moved = 0
    skipped = 0
    errors = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        if selection == "production":
            folder, reason = _production_group(pdf_path.name)
        else:
            folder, reason = _publisher_group(pdf_path)

        if not folder:
            skipped += 1
            api.log("warn", f"Skipping '{pdf_path.name}': {reason}.")
        else:
            destination_dir = root / folder
            try:
                move_to_dir(pdf_path, destination_dir)
                moved += 1
                api.log("info", f"Moved '{pdf_path.name}' -> '{folder}/'.")
            except OSError as exc:
                errors += 1
                api.log(
                    "error",
                    f"Failed to move '{pdf_path.name}' to '{folder}/': {exc}",
                )

        api.progress(
            current=index,
            total=len(pdf_files),
            unit="files",
            message="Grouping PDFs...",
        )

    api.emit_result(
        {
            "_title": "Grouping Completed",
            "_status": "success" if errors == 0 else "warn",
            "Mode": selection,
            "Directory": str(root),
            "Moved": moved,
            "Skipped": skipped,
            "Errors": errors,
        }
    )


if __name__ == "__main__":
    main()
