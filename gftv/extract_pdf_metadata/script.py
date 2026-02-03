from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

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


def _parse_xmp(xmp: str) -> dict[str, object]:
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

    parsed: dict[str, object] = {}
    administrator = root.findtext(".//ferp:administrator", default="", namespaces=ns)
    if administrator.strip():
        parsed["ferp:namespace"] = ns["ferp"]
        parsed["ferp:administrator"] = administrator.strip()

    data_added_date = root.findtext(
        ".//ferp:dataAddedDate", default="", namespaces=ns
    )
    if data_added_date.strip():
        parsed["ferp:namespace"] = ns["ferp"]
        parsed["ferp:dataAddedDate"] = data_added_date.strip()

    stamp_spec_version = root.findtext(
        ".//ferp:stampSpecVersion", default="", namespaces=ns
    )
    if stamp_spec_version.strip():
        parsed["ferp:namespace"] = ns["ferp"]
        parsed["ferp:stampSpecVersion"] = stamp_spec_version.strip()

    agreements: list[dict[str, object]] = []
    agreement_nodes = root.findall(".//ferp:agreements/rdf:Bag/rdf:li", ns)
    for agreement_node in agreement_nodes:
        publishers = [
            (li.text or "").strip()
            for li in agreement_node.findall(".//ferp:publishers/rdf:Bag/rdf:li", ns)
            if (li.text or "").strip()
        ]
        effective_dates: list[dict[str, object]] = []
        date_nodes = agreement_node.findall(".//ferp:effectiveDates/rdf:Seq/rdf:li", ns)
        for date_node in date_nodes:
            date_value = (
                date_node.findtext("ferp:date", default="", namespaces=ns) or ""
            ).strip()
            territories = [
                (li.text or "").strip()
                for li in date_node.findall(".//ferp:territories/rdf:Bag/rdf:li", ns)
                if (li.text or "").strip()
            ]
            if date_value or territories:
                effective_dates.append({"date": date_value, "territories": territories})
        if publishers or effective_dates:
            agreements.append(
                {
                    "publishers": publishers,
                    "effective_dates": effective_dates,
                }
            )

    if agreements:
        parsed["ferp:namespace"] = ns["ferp"]
        parsed["ferp:agreements"] = agreements

    return parsed


def _split_agreements_for_csv(metadata: dict[str, object]) -> list[dict[str, str]]:
    administrator = str(metadata.get("ferp:administrator", "") or "")
    data_added_date = str(metadata.get("ferp:dataAddedDate", "") or "")
    stamp_spec_version = str(metadata.get("ferp:stampSpecVersion", "") or "")
    agreements = metadata.get("ferp:agreements", [])
    if not isinstance(agreements, list):
        agreements = []
    rows: list[dict[str, str]] = []
    if not agreements:
        rows.append(
            {
                "Administrator": administrator,
                "Data Added Date": data_added_date,
                "Stamp Spec Version": stamp_spec_version,
                "Agreement": "",
                "Publishers": "",
                "Effective Date": "",
                "Territories": "",
            }
        )
        return rows
    for index, agreement in enumerate(agreements, start=1):
        if not isinstance(agreement, dict):
            continue
        publishers = agreement.get("publishers", [])
        publisher_text = (
            " | ".join(publishers) if isinstance(publishers, list) else str(publishers)
        )
        effective_dates = agreement.get("effective_dates", [])
        if not isinstance(effective_dates, list):
            effective_dates = []
        if not effective_dates:
            rows.append(
                {
                    "Administrator": administrator,
                    "Data Added Date": data_added_date,
                    "Stamp Spec Version": stamp_spec_version,
                    "Agreement": str(index),
                    "Publishers": publisher_text,
                    "Effective Date": "",
                    "Territories": "",
                }
            )
            continue
        for entry in effective_dates:
            if not isinstance(entry, dict):
                continue
            date_value = str(entry.get("date", "") or "")
            territories = entry.get("territories", [])
            territory_text = (
                " | ".join(territories)
                if isinstance(territories, list)
                else str(territories)
            )
            rows.append(
                {
                    "Administrator": administrator,
                    "Data Added Date": data_added_date,
                    "Stamp Spec Version": stamp_spec_version,
                    "Agreement": str(index),
                    "Publishers": publisher_text,
                    "Effective Date": date_value,
                    "Territories": territory_text,
                }
            )
    return rows


def _write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "File",
                "Relative Path",
                "Administrator",
                "Data Added Date",
                "Stamp Spec Version",
                "Agreement",
                "Publishers",
                "Effective Date",
                "Territories",
            ],
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
                value = metadata[tag]
                if tag == "ferp:agreements":
                    if isinstance(value, list):
                        lines.append("  ferp:agreements:")
                        for agreement in value:
                            if not isinstance(agreement, dict):
                                lines.append(f"    - {agreement}")
                                continue
                            publishers = agreement.get("publishers", [])
                            effective_dates = agreement.get("effective_dates", [])
                            lines.append(
                                f"    - publishers: {', '.join(publishers) if publishers else ''}"
                            )
                            if isinstance(effective_dates, list) and effective_dates:
                                lines.append("      effective_dates:")
                                for entry in effective_dates:
                                    if not isinstance(entry, dict):
                                        lines.append(f"        - {entry}")
                                        continue
                                    date_value = entry.get("date", "")
                                    territories = entry.get("territories", [])
                                    territory_text = (
                                        ", ".join(territories)
                                        if isinstance(territories, list)
                                        else str(territories)
                                    )
                                    lines.append(
                                        f"        - date: {date_value} | territories: {territory_text}"
                                    )
                            continue
                        continue
                lines.append(f"  {tag}: {value}")
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
                "label": "Recursive",
                "default": False,
            },
            {
                "id": "write_csv",
                "type": "bool",
                "label": "Write CSV",
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
        metadata: dict[str, object] = {}
        xmp = _extract_xmp(reader)
        if xmp:
            parsed_xmp = _parse_xmp(xmp)
            if parsed_xmp:
                metadata = parsed_xmp
        relative_path = str(pdf_path.relative_to(target_dir))
        if metadata:
            api.log("info", f"{relative_path}: {json.dumps(metadata, sort_keys=True)}")
        else:
            api.log("info", f"{relative_path}: No FERP metadata found")

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
                        "Administrator": "",
                        "Data Added Date": "",
                        "Stamp Spec Version": "",
                        "Agreement": "",
                        "Publishers": "",
                        "Effective Date": "",
                        "Territories": "",
                    }
                )
            else:
                for row in _split_agreements_for_csv(metadata):
                    rows.append(
                        {
                            "File": pdf_path.name,
                            "Relative Path": relative_path,
                            **row,
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
