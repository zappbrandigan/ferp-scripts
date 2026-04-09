from __future__ import annotations

import re
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
    deal: str
    catalog: str
    pd_code: str
    film_or_series: str
    revision_status: str
    production_title: str
    episode_title: str
    season_number: str
    episode_number: str
    territory_code: str
    territory: str


_PRODUCTION_DELIM = "   "
_EPISODE_DELIM = "  "
_DATEISH_EPISODE_NUMBER_RE = re.compile(
    r"^\d{6,8}(?:-\d+[A-Za-z]?)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"
)
_EPISODE_CODE_RE = re.compile(r"^\d{4,5}$")


def _parse_series_title_fields(stem: str) -> tuple[str, str, str, str]:
    if _PRODUCTION_DELIM not in stem:
        return stem, "", "", ""

    production_title, remainder = stem.split(_PRODUCTION_DELIM, 1)
    production_title = production_title.strip()
    remainder = remainder.strip()
    if not remainder or "Ep No." not in remainder:
        return production_title, "", "", ""

    episode_title = ""
    episode_info = remainder
    if _EPISODE_DELIM in remainder:
        possible_episode_title, possible_episode_info = remainder.rsplit(_EPISODE_DELIM, 1)
        if "Ep No." in possible_episode_info:
            episode_title = possible_episode_title.strip()
            episode_info = possible_episode_info.strip()

    episode_info = episode_info.strip()
    if "Ep No." not in episode_info:
        return production_title, episode_title, "", ""

    _, episode_number = episode_info.split("Ep No.", 1)
    episode_number = episode_number.strip()
    if not episode_number or _DATEISH_EPISODE_NUMBER_RE.fullmatch(episode_number):
        return production_title, episode_title, "", ""

    season_number = ""
    if _EPISODE_CODE_RE.fullmatch(episode_number):
        if len(episode_number) == 4:
            season_number = episode_number[0]
        else:
            season_number = episode_number[:2]
        episode_number = str(int(episode_number[-3:]))

    return production_title, episode_title, season_number, episode_number


def _parse_title_fields(stem: str, media_type: str) -> tuple[str, str, str, str]:
    if media_type == "film":
        return stem, "", "", ""
    return _parse_series_title_fields(stem)


def _parse_cue_sheet_path(pdf_path: Path, root: Path) -> CueSheetRow | None:
    relative_parts = pdf_path.relative_to(root).parts
    if len(relative_parts) < 4:
        return None

    filename = relative_parts[-1]
    catalog_folder = relative_parts[-2]
    media_type = relative_parts[-3].lower()
    territory_code = relative_parts[-4]

    if media_type not in {"film", "tv"}:
        return None

    if len(relative_parts) >= 5 and relative_parts[-5] == "_REV":
        date_range = root.name if len(relative_parts) == 5 else relative_parts[-6]
        revision_status = "Revision"
    elif len(relative_parts) == 4:
        date_range = root.name
        revision_status = "New"
    else:
        date_range = relative_parts[-5]
        revision_status = "New"

    if not date_range or date_range == "_REV":
        return None
    if not catalog_folder:
        return None
    if not territory_code:
        return None
    if not filename.lower().endswith(".pdf"):
        return None

    if " - " in catalog_folder:
        catalog, deal = catalog_folder.split(" - ", 1)
    else:
        catalog = catalog_folder
        deal = ""

    film_or_series = "Film" if media_type == "film" else "Series"
    title_stem = Path(filename).stem
    production_title, episode_title, season_number, episode_number = (
        _parse_title_fields(title_stem, media_type)
    )

    return CueSheetRow(
        deal=deal,
        catalog=catalog,
        pd_code="",
        film_or_series=film_or_series,
        revision_status=revision_status,
        production_title=production_title,
        episode_title=episode_title,
        season_number=season_number,
        episode_number=episode_number,
        territory_code=territory_code,
        territory="",
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
        "Deal",
        "Catalog",
        "PD Code",
        "Film or Series",
        "Revision Status",
        "Production Title",
        "Episode Title",
        "Season Number",
        "Episode Number",
        "Territory Code",
        "Territory",
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
                row.deal,
                row.catalog,
                row.pd_code,
                row.film_or_series,
                row.revision_status,
                row.production_title,
                row.episode_title,
                row.season_number,
                row.episode_number,
                row.territory_code,
                row.territory,
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
        "A": 40,
        "B": 18,
        "C": 18,
        "D": 16,
        "E": 18,
        "F": 40,
        "G": 32,
        "H": 16,
        "I": 20,
        "J": 18,
        "K": 18,
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
            row.deal.lower(),
            row.catalog.lower(),
            row.pd_code.lower(),
            row.film_or_series.lower(),
            row.revision_status.lower(),
            row.production_title.lower(),
            row.episode_title.lower(),
            row.season_number.lower(),
            row.episode_number.lower(),
            row.territory_code.lower(),
            row.territory.lower(),
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
