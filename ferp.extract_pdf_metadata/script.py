from __future__ import annotations

import csv
import json
from pathlib import Path

from PyPDF2 import PdfReader

from ferp.fscp.scripts import sdk


def _collect_pdfs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.pdf"))
    return sorted(path for path in root.glob("*.pdf") if path.is_file())


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
        writer = csv.DictWriter(handle, fieldnames=["file", "relative_path", "tag", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _format_results(results: list[dict[str, object]]) -> str:
    MAX_DISPLAY = 100
    lines: list[str] = []
    for index, entry in enumerate(results):
        if index >= MAX_DISPLAY:
            lines.append(f"(truncated to {MAX_DISPLAY} results; export CSV for full data)")
            break
        relative_path = str(entry.get("relative_path", ""))
        metadata = entry.get("metadata")
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
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Target '{target_dir}' is not a directory.")

    response = api.request_input(
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
    )

    payload: dict[str, object]
    try:
        payload = json.loads(response) if response else {}
    except json.JSONDecodeError:
        payload = {}

    recursive = bool(payload.get("recursive", False))
    write_csv = bool(payload.get("write_csv", False))

    pdf_files = _collect_pdfs(target_dir, recursive)
    total_files = len(pdf_files)
    api.log(
        "info",
        f"PDFs found={total_files} | recursive={recursive} | csv={write_csv}",
    )

    rows: list[dict[str, str]] = []
    results: list[dict[str, object]] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.progress(current=index, total=total_files or 1, unit="files")
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            api.log("warning", f"Failed to read '{pdf_path}': {exc}")
            continue

        if reader.is_encrypted:
            api.log("warning", f"Skipped encrypted PDF: {pdf_path}")
            continue

        metadata = _normalize_metadata(reader)
        relative_path = str(pdf_path.relative_to(target_dir))
        if metadata:
            api.log("info", f"{relative_path}: {json.dumps(metadata, sort_keys=True)}")
        else:
            api.log("info", f"{relative_path}: No metadata found")

        results.append(
            {
                "file": pdf_path.name,
                "relative_path": relative_path,
                "metadata": metadata,
            }
        )

        if write_csv:
            if not metadata:
                rows.append(
                    {
                        "file": pdf_path.name,
                        "relative_path": relative_path,
                        "tag": "",
                        "value": "",
                    }
                )
            for tag, value in sorted(metadata.items()):
                rows.append(
                    {
                        "file": pdf_path.name,
                        "relative_path": relative_path,
                        "tag": tag,
                        "value": value,
                    }
                )

    csv_path: Path | None = None
    if write_csv:
        csv_path = target_dir.parent / f"{target_dir.name}_pdf_metadata.csv"
        _write_csv(csv_path, rows)

    api.emit_result(
        {
            "files_found": total_files,
            "csv_path": str(csv_path) if csv_path else None,
            "results": _format_results(results),
        }
    )


if __name__ == "__main__":
    main()
