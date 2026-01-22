from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from PyPDF2 import PdfReader

from ferp.fscp.scripts import sdk


class UserResponse(TypedDict):
    value: str
    recursive: bool
    write_csv: bool


def _collect_pdfs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.pdf"))
    return sorted(path for path in root.glob("*.pdf") if path.is_file())


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


def _parse_xmp(xmp: str) -> dict[str, str]:
    match = re.search(r"(<x:xmpmeta\b.*?</x:xmpmeta>)", xmp, re.DOTALL)
    xml_payload = match.group(1) if match else xmp
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        return {}

    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "ferp": "https://tulbox.app/ferp/xmp/1.0",
    }

    publishers = [
        (li.text or "").strip()
        for li in root.findall(".//ferp:publishers/rdf:Bag/rdf:li", ns)
        if (li.text or "").strip()
    ]

    parsed: dict[str, str] = {}
    if publishers:
        parsed["ferp:namespace"] = ns["ferp"]
        parsed["ferp:publishers"] = "; ".join(publishers)
        parsed["ferp:publishers_count"] = str(len(publishers))

    return parsed


def _normalize_metadata(reader: PdfReader) -> dict[str, str]:
    metadata = reader.metadata or {}
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        label = str(key)
        if label.startswith("/"):
            label = label[1:]
        normalized[label] = "" if value is None else str(value)
    return normalized


def _write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["File", "Relative Path", "Tag", "Value"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_results(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for index, entry in enumerate(results):
        relative_path = str(entry.get("Relative Path", ""))
        metadata = entry.get("Metadata")
        lines.append(relative_path)
        if isinstance(metadata, dict) and metadata:
            for tag in sorted(metadata):
                lines.append(f"  {tag}: {metadata[tag]}")
        else:
            lines.append("  (no metadata)")
        lines.append("")
    return "\n".join(lines).rstrip()


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = ctx.target_path

    payload = api.request_input_json(
        "Metadata options",
        id="pdf_metadata_options",
        fields=[
            {
                "id": "recursive",
                "type": "bool",
                "label": "Scan subdirectories",
                "default": False,
            },
            {
                "id": "write_csv",
                "type": "bool",
                "label": "Write CSV summary",
                "default": False,
            },
        ],
        show_text_input=False,
        payload_type=UserResponse,
    )

    recursive = payload["recursive"]
    write_csv = payload["write_csv"]

    pdf_files = _collect_pdfs(target_dir, recursive)
    total_files = len(pdf_files)
    if not total_files:
        api.log("warn", f"No PDF files found: {target_dir}")
        api.emit_result(
            {
                "_status": "warn",
                "_title": "No PDF Files Found",
                "File Path": str(target_dir),
            }
        )
        return

    api.log(
        "info",
        f"PDFs found={total_files} | recursive={recursive} | csv={write_csv}",
    )

    rows: list[dict[str, str]] = []
    results: list[dict[str, object]] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        api.progress(current=index, total=total_files, unit="files", every=10)
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            api.log(
                "warn",
                (f"Failed to read '{pdf_path}' in strict mode; retrying: {exc}"),
            )
            try:
                reader = PdfReader(str(pdf_path), strict=False)
            except Exception as exc:  # noqa: BLE001
                api.log("warn", f"Failed to read '{pdf_path}': {exc}")
                continue

        if reader.is_encrypted:
            api.log("warn", f"Skipped encrypted PDF: {pdf_path}")
            continue

        api.check_cancel()
        metadata = _normalize_metadata(reader)
        xmp = _extract_xmp(reader)
        if xmp:
            parsed_xmp = _parse_xmp(xmp)
            if parsed_xmp:
                metadata.update(parsed_xmp)
            else:
                metadata["XMP"] = xmp
        relative_path = str(pdf_path.relative_to(target_dir))
        if metadata:
            api.log("info", f"{relative_path}: {json.dumps(metadata, sort_keys=True)}")
        else:
            api.log("info", f"{relative_path}: No metadata found")

        results.append(
            {
                "File": pdf_path.name,
                "Relative Path": relative_path,
                "Metadata": metadata,
            }
        )

        if write_csv:
            if not metadata:
                rows.append(
                    {
                        "File": pdf_path.name,
                        "Relative Path": relative_path,
                        "Tag": "",
                        "Value": "",
                    }
                )
            for tag, value in sorted(metadata.items()):
                rows.append(
                    {
                        "File": pdf_path.name,
                        "Relative Path": relative_path,
                        "Tag": tag,
                        "Value": value,
                    }
                )

    csv_path: Path | None = None
    if write_csv:
        csv_path = target_dir.parent / f"{target_dir.name}_pdf_metadata.csv"
        _write_csv(csv_path, rows)

    api.emit_result(
        {
            "_title": "Metadata Extraction Finished",
            "Files Found": total_files,
            "CSV Location": str(csv_path) if csv_path else "Not exported.",
            "Results": f"\n{_format_results(results)}",
        }
    )


if __name__ == "__main__":
    main()
