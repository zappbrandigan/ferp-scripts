from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, TypedDict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from pypdf import PdfReader

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import collect_files


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


def _write_xlsx(xlsx_path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "file_name",
        "relative_path",
        "page_number",
        "match_text",
        "context_excerpt",
    ]
    wb = Workbook()
    ws = wb.active
    if ws is None:
        ws = wb.create_sheet()
    ws.title = "Query Results"

    header_font = Font(bold=True)
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    data_alignment = Alignment(vertical="top", wrap_text=True)
    centered_alignment = Alignment(
        horizontal="center", vertical="center", wrap_text=True
    )

    ws.append([header.replace("_", " ").title() for header in headers])
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.alignment = header_alignment

    for row in rows:
        ws.append([row.get(key, "") for key in headers])

    last_row = max(1, len(rows) + 1)
    last_col = len(headers)
    table_ref = f"A1:{get_column_letter(last_col)}{last_row}"
    table = Table(displayName="QueryResults", ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleLight1",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)

    for col_idx in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 24

    ws.column_dimensions["E"].width = 80
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.column in (3, 4):
                cell.alignment = centered_alignment
            else:
                cell.alignment = data_alignment

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


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

    target_path = ctx.target_path
    pdf_files = collect_files(
        target_path,
        "*.pdf",
        recursive=False,
        check_cancel=api.check_cancel,
    )
    root_path = target_path if ctx.target_kind == "directory" else target_path.parent
    total_files = len(pdf_files)
    if not total_files:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning",
                "Info": "No PDF files found to search.",
            }
        )
        return
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

    xlsx_path: Path | None = None
    if matches:
        stem = target_path.stem if ctx.target_kind == "file" else root_path.name
        output_dir = (
            target_path.parent if ctx.target_kind == "file" else root_path.parent
        )
        xlsx_path = output_dir / f"{stem}_query_results.xlsx"
        _write_xlsx(xlsx_path, matches)

    api.emit_result(
        {
            "_title": "PDF Query Results",
            "Query": query,
            "Files Searched": total_files,
            "Files With Matches": len(files_with_matches),
            "XLSX Path": str(xlsx_path) if xlsx_path else None,
        }
    )


if __name__ == "__main__":
    main()
