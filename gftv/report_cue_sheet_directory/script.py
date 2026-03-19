from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import build_destination


@dataclass(frozen=True)
class CueSheetRow:
    date_range: str
    territory_code: str
    revision_status: str
    media_type: str
    catalog_code: str
    cue_sheet_pdf: str


def _parse_cue_sheet_path(pdf_path: Path, root: Path) -> CueSheetRow | None:
    relative_parts = pdf_path.relative_to(root).parts
    if len(relative_parts) < 4:
        return None

    filename = relative_parts[-1]
    catalog_code = relative_parts[-2]
    media_type = relative_parts[-3].lower()
    territory_code = relative_parts[-4]

    if media_type not in {"film", "tv"}:
        return None

    if len(relative_parts) == 4:
        date_range = root.name
        revision_status = "New"
    elif len(relative_parts) >= 5 and relative_parts[-5] == "_REV":
        date_range = root.name if len(relative_parts) == 5 else relative_parts[-6]
        revision_status = "Revision"
    else:
        date_range = relative_parts[-5]
        revision_status = "New"

    if not date_range or date_range == "_REV":
        return None
    if not territory_code:
        return None
    if not catalog_code:
        return None
    if not filename.lower().endswith(".pdf"):
        return None

    return CueSheetRow(
        date_range=date_range,
        territory_code=territory_code,
        revision_status=revision_status,
        media_type=media_type,
        catalog_code=catalog_code,
        cue_sheet_pdf=filename,
    )


def _collect_pdfs(root: Path, api: sdk.ScriptAPI) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.pdf"):
        api.check_cancel()
        if path.is_file() and not path.name.startswith((".", "~$")):
            files.append(path)
    return sorted(files)


def _write_report(xlsx_path: Path, rows: list[CueSheetRow]) -> None:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Date Range",
        "Territory Code",
        "Revision Status",
        "Film or TV",
        "Catalog Code",
        "Cue Sheet",
    ]

    wb = Workbook()
    ws = wb.active
    if ws is None:
        ws = wb.create_sheet(title="Cue Sheet Summary")
    else:
        ws.title = "Cue Sheet Summary"

    header_font = Font(bold=True)
    header_alignment = Alignment(horizontal="center", vertical="center")
    data_alignment = Alignment(vertical="center")

    ws.append(headers)
    for col_idx, _header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.alignment = header_alignment

    for row in rows:
        ws.append(
            [
                row.date_range,
                row.territory_code,
                row.revision_status,
                row.media_type,
                row.catalog_code,
                row.cue_sheet_pdf,
            ]
        )

    if rows:
        last_row = len(rows) + 1
        last_col = len(headers)
        table_ref = f"A1:{get_column_letter(last_col)}{last_row}"
        table = Table(displayName="CueSheetSummary", ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleLight1",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)

    column_widths = {
        "A": 22,
        "B": 18,
        "C": 18,
        "D": 12,
        "E": 20,
        "F": 48,
    }
    for column, width in column_widths.items():
        ws.column_dimensions[column].width = width

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = data_alignment

    wb.save(xlsx_path)


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target = ctx.target_path
    if not target.is_dir():
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Cue Sheet Report Canceled",
                "Info": "Select a directory to build the cue sheet summary.",
            }
        )
        return

    pdf_files = _collect_pdfs(target, api)
    if not pdf_files:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Cue Sheet Report Complete",
                "Rows Written": 0,
                "Skipped PDFs": 0,
                "Info": "No PDF files were found under the selected directory.",
            }
        )
        return

    rows: list[CueSheetRow] = []
    skipped: list[str] = []
    total_files = len(pdf_files)

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        parsed = _parse_cue_sheet_path(pdf_path, target)
        if parsed is None:
            skipped.append(str(pdf_path.relative_to(target)))
        else:
            rows.append(parsed)
        api.progress(current=index, total=total_files, unit="files")

    rows.sort(
        key=lambda row: (
            row.date_range.lower(),
            row.territory_code.lower(),
            row.revision_status.lower(),
            row.media_type.lower(),
            row.catalog_code.lower(),
            row.cue_sheet_pdf.lower(),
        )
    )

    out_path = build_destination(
        target.parent,
        f"{target.name}_cue_sheet_summary",
        ".xlsx",
    )
    _write_report(out_path, rows)

    if skipped:
        preview = ", ".join(skipped[:5])
        api.log(
            "warn",
            f"Skipped {len(skipped)} PDF(s) that did not match the expected folder layout: {preview}",
        )

    api.emit_result(
        {
            "_title": "Cue Sheet Report Complete",
            "XLSX Path": str(out_path),
            "Rows Written": len(rows),
            "Skipped PDFs": len(skipped),
        }
    )


if __name__ == "__main__":
    main()
