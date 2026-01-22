from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, TypedDict

from pypdf import PdfReader

from ferp.fscp.scripts import sdk


class QueryOptions(TypedDict):
    value: str
    regex: bool
    case_sensitive: bool


def _compile_pattern(
    query: str,
    use_regex: bool,
    case_sensitive: bool,
    api: sdk.ScriptAPI,
) -> re.Pattern[str] | None:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        if use_regex:
            return re.compile(query, flags)
        return re.compile(re.escape(query), flags)
    except re.error as exc:
        api.log("error", f"Invalid regular expression '{query}': {exc}")
        return None


def _process_pdf(
    pdf_path: Path,
    root: Path,
    pattern: re.Pattern[str],
    context_chars: int,
    api: sdk.ScriptAPI,
    check_cancel: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        api.log("warn", f"Skipping encrypted PDF: {pdf_path.name}")
        return []

    pdf_matches: list[dict[str, object]] = []
    relative_path = pdf_path.relative_to(root)

    for page_number, page in enumerate(reader.pages, start=1):
        if check_cancel is not None:
            check_cancel()
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            api.log(
                "warn",
                f"Text extraction failed for '{pdf_path.name}' (page {page_number}): {exc}",
            )
            continue

        if not text:
            continue

        for match in pattern.finditer(text):
            if check_cancel is not None:
                check_cancel()
            span = match.span()
            context = _extract_context(text, span[0], span[1], context_chars)
            pdf_matches.append(
                {
                    "file_name": pdf_path.name,
                    "relative_path": str(relative_path),
                    "page_number": page_number,
                    "match_text": match.group(0),
                    "context_excerpt": context,
                }
            )

    if not pdf_matches:
        api.log("info", f"No matches found in {pdf_path.name}")

    return pdf_matches


def _extract_context(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    snippet = text[left:right].replace("\n", " ").strip()
    return snippet


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "file_name",
        "relative_path",
        "page_number",
        "match_text",
        "context_excerpt",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    payload = api.request_input_json(
        "Enter search query",
        id="query_pdf_options",
        fields=[
            {"id": "regex", "type": "bool", "label": "Use regex", "default": False},
            {
                "id": "case_sensitive",
                "type": "bool",
                "label": "Case sensitive",
                "default": False,
            },
        ],
        payload_type=QueryOptions,
    )

    query = payload["value"]
    regex = payload["regex"]
    case_sensitive = payload["case_sensitive"]

    if not query:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning",
                "Info": "Search query is required. Query not performed.",
            }
        )
        return

    context_chars = 80
    context_response = api.request_input(
        "Context characters around each match",
        default=str(context_chars),
        id="query_pdf_context",
    )
    if context_response:
        try:
            context_chars = int(context_response)
        except ValueError:
            api.emit_result(
                {
                    "_status": "warn",
                    "_title": "Warning",
                    "Info": "Context characters must be a number.",
                }
            )
            return
        if context_chars <= 0:
            api.emit_result(
                {
                    "_status": "warn",
                    "_title": "Warning",
                    "Info": "Context characters must be greater than zero.",
                }
            )
            return

    root_path = ctx.target_path
    pdf_files = sorted(root_path.rglob("*.pdf"))
    total_files = len(pdf_files)
    api.log(
        "info",
        (
            f"Query '{query}' | regex={regex} "
            f"| case_sensitive={case_sensitive} | PDFs found={total_files}"
        ),
    )

    matches: list[dict[str, object]] = []
    files_with_matches: set[Path] = set()

    pattern = _compile_pattern(query, regex, case_sensitive, api)
    if pattern is None:
        api.exit(code=1)
        return

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        api.progress(current=index, total=total_files or 1, unit="files")
        try:
            pdf_matches = _process_pdf(
                pdf_path,
                root_path,
                pattern,
                context_chars,
                api,
                check_cancel=api.check_cancel,
            )
        except Exception as exc:  # noqa: BLE001
            api.log("warn", f"Failed to process '{pdf_path}': {exc}")
            continue

        if pdf_matches:
            matches.extend(pdf_matches)
            files_with_matches.add(pdf_path)

    csv_path: Path | None = None
    if matches:
        csv_path = root_path.parent / f"{root_path.name}_query_results.csv"
        _write_csv(csv_path, matches)

    api.emit_result(
        {
            "_title": "PDF Query Results",
            "Query": query,
            "Files Searched": total_files,
            "Files With Matches": len(files_with_matches),
            "CSV Path": str(csv_path) if csv_path else None,
        }
    )


if __name__ == "__main__":
    main()
