from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PyPDF2 import PdfReader

from ferp.fscp.scripts import sdk


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search PDF files for text.")
    parser.add_argument("query", nargs="?", help="Text or pattern to search for.")
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat the query as a regular expression.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Perform a case-sensitive search (default: insensitive).",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=80,
        help="Number of characters of context to capture around each match (default: 80).",
    )
    return parser


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    parser = _build_parser()
    try:
        options = parser.parse_args(ctx.args or [])
    except SystemExit:
        raise ValueError("Invalid arguments supplied to query_pdf script.")

    query = options.query
    regex = options.regex
    case_sensitive = options.case_sensitive

    if not query:
        response = api.request_input(
            "Enter search query",
            id="query_pdf_options",
            fields=[
                {
                    "id": "regex", 
                    "type": "bool", 
                    "label": "Use regex", 
                    "default": regex
                },
                {
                    "id": "case_sensitive",
                    "type": "bool",
                    "label": "Case sensitive",
                    "default": case_sensitive,
                },
            ],
        )
        if response is None:
            api.exit(code=1)
            return
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            payload = {"value": response}
        query = str(payload.get("value", "")).strip()
        regex = bool(payload.get("regex", regex))
        case_sensitive = bool(payload.get("case_sensitive", case_sensitive))

    if not query:
        raise ValueError("Search query is required.")

    root_path = ctx.target_path
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Target '{root_path}' is not a directory.")

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
        api.progress(current=index, total=total_files or 1, unit="files")
        try:
            pdf_matches = _process_pdf(
                pdf_path,
                root_path,
                pattern,
                options.context_chars,
                api,
            )
        except Exception as exc:  # noqa: BLE001
            api.log("warning", f"Failed to process '{pdf_path}': {exc}")
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
            "files_searched": total_files,
            "files_with_matches": len(files_with_matches),
            "csv_path": str(csv_path) if csv_path else None,
        }
    )
    api.exit(code=0)


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
) -> list[dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        api.log("warning", f"Skipping encrypted PDF: {pdf_path.name}")
        return []

    pdf_matches: list[dict[str, object]] = []
    relative_path = pdf_path.relative_to(root)

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            api.log("warning", f"Text extraction failed for '{pdf_path.name}' (page {page_number}): {exc}")
            continue

        if not text:
            continue

        for match in pattern.finditer(text):
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
    fieldnames = ["file_name", "relative_path", "page_number", "match_text", "context_excerpt"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
