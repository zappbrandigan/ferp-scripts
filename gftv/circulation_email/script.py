from __future__ import annotations

import json
from dataclasses import dataclass
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import build_destination, load_settings

_REVISION_FOLDER = "_REV"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "temp"
_TV_TEMPLATE_PATH = _TEMPLATE_DIR / "tv-circulations.txt"
_FILM_TEMPLATE_PATH = _TEMPLATE_DIR / "film-circulations.txt"
_OUTPUT_SUFFIX = ".eml"


@dataclass(frozen=True)
class CueSheetRecord:
    catalog_code: str
    catalog_name: str
    catalog_label: str
    media_type: str
    revision_status: str
    territory_code: str


@dataclass(frozen=True)
class CirculationGroup:
    media_type: str
    revision_status: str
    territory_code: str
    territory_name: str


@dataclass(frozen=True)
class DealCatalogGroup:
    deal_owner: str
    records: list[CueSheetRecord]


def _get_saved_cues_inbox(ctx: sdk.ScriptContext) -> str:
    settings = load_settings(ctx)
    integrations = settings.get("integrations", {})
    if not isinstance(integrations, dict):
        return ""
    gftv = integrations.get("gftv", {})
    if not isinstance(gftv, dict):
        return ""
    return str(gftv.get("cuesInboxEmail") or "").strip()


def _resolve_cues_inbox(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> str | None:
    saved = _get_saved_cues_inbox(ctx)
    if saved:
        return saved

    api.emit_result(
        {
            "_status": "warn",
            "_title": "Cues Inbox Required",
            "Info": "Set the GFTV cues inbox in the FERP command palette before generating circulation drafts.",
        }
    )
    return None


def _load_territory_names(ctx: sdk.ScriptContext) -> dict[str, str]:
    rows = _load_cache_group(ctx, "Territories")
    names: dict[str, str] = {}
    for item in rows:
        code = str(item.get("name") or "").strip()
        columns = item.get("columns")
        territory = _column_text(columns, "Territory")
        if code and territory:
            names[code.casefold()] = territory
    return names


def _load_deal_owners(ctx: sdk.ScriptContext) -> dict[str, str]:
    rows = _load_cache_group(ctx, "Deals")
    owners: dict[str, str] = {}
    for item in rows:
        code = str(item.get("name") or "").strip()
        columns = item.get("columns")
        owner = _column_text(columns, "Territory")
        if code and owner:
            owners[code.casefold()] = owner
    return owners


def _load_cache_group(ctx: sdk.ScriptContext, group_name: str) -> list[dict[str, Any]]:
    cache_path = Path(ctx.environment["paths"]["cache_dir"]) / "territories_cache.json"
    if not cache_path.exists():
        return []

    try:
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    groups = raw_cache.get("groups") if isinstance(raw_cache, dict) else None
    rows = groups.get(group_name) if isinstance(groups, dict) else None
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, dict)]


def _column_text(columns: Any, name: str) -> str:
    if not isinstance(columns, dict):
        return ""
    return str(columns.get(name) or "").strip()


def _resolve_territory_name(
    territory_code: str, territory_names: dict[str, str]
) -> str:
    return territory_names.get(territory_code.casefold()) or territory_code


def _resolve_deal_owner(deal_code: str, deal_owners: dict[str, str]) -> str:
    return deal_owners.get(deal_code.casefold()) or "{deal_owner}"


def _collect_pdfs(root: Path, api: sdk.ScriptAPI) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.pdf"):
        api.check_cancel()
        if path.is_file() and not path.name.startswith((".", "~$")):
            files.append(path)
    return sorted(files)


def _parse_catalog_folder(folder_name: str) -> tuple[str, str]:
    if " - " not in folder_name:
        return folder_name.strip(), folder_name.strip()
    catalog_code, catalog_name = folder_name.split(" - ", 1)
    return catalog_code.strip(), catalog_name.strip()


def _parse_cue_sheet_path(pdf_path: Path, root: Path) -> CueSheetRecord | None:
    relative_parts = pdf_path.relative_to(root).parts
    if len(relative_parts) < 4:
        return None

    catalog_folder = relative_parts[-2].strip()
    media_type = relative_parts[-3].strip().lower()
    territory_code = relative_parts[-4].strip()

    if media_type not in {"film", "tv"}:
        return None
    if not territory_code or not catalog_folder:
        return None

    if len(relative_parts) >= 5 and relative_parts[-5] == _REVISION_FOLDER:
        revision_status = "Revised"
    else:
        revision_status = "New"

    catalog_code, catalog_name = _parse_catalog_folder(catalog_folder)
    if not catalog_code or not catalog_name:
        return None

    return CueSheetRecord(
        catalog_code=catalog_code,
        catalog_name=catalog_name,
        catalog_label=catalog_folder,
        media_type=media_type,
        revision_status=revision_status,
        territory_code=territory_code,
    )


def _load_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return text.strip()

    _, _front_matter, remainder = text.split("---", 2)
    return remainder.strip()


def _split_template(template: str) -> tuple[str, str]:
    lines = template.splitlines()
    if not lines:
        return "", ""
    first_line = lines[0].strip()
    if first_line.startswith("Subject:"):
        subject = first_line.removeprefix("Subject:").strip()
        body = "\n".join(lines[1:]).strip()
        return subject, body
    return "", template.strip()


def _catalog_codes(records: list[CueSheetRecord]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        if record.catalog_code in seen:
            continue
        seen.add(record.catalog_code)
        ordered.append(record.catalog_code)
    return ", ".join(ordered)


def _catalog_bullets(records: list[CueSheetRecord]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for record in records:
        if record.catalog_label in seen:
            continue
        seen.add(record.catalog_label)
        lines.append(f"• {record.catalog_label}")
    return "\n".join(lines)


def _catalog_count(records: list[CueSheetRecord]) -> int:
    return len({record.catalog_label for record in records})


def _catalog_list_items(records: list[CueSheetRecord]) -> str:
    seen: set[str] = set()
    items: list[str] = []
    for record in records:
        if record.catalog_label in seen:
            continue
        seen.add(record.catalog_label)
        items.append(
            "<li>"
            f'<span style="font-family: Calibri, Arial, sans-serif; font-size: 11pt;">'
            f"{escape(record.catalog_label)}"
            "</span>"
            "</li>"
        )
    return "".join(items)


def _deal_catalog_groups(
    records: list[CueSheetRecord], deal_owners: dict[str, str]
) -> list[DealCatalogGroup]:
    groups: dict[str, list[CueSheetRecord]] = {}
    owner_labels: dict[str, str] = {}
    for record in records:
        owner = _resolve_deal_owner(record.catalog_code, deal_owners)
        owner_key = owner.casefold()
        owner_labels.setdefault(owner_key, owner)
        groups.setdefault(owner_key, []).append(record)

    deal_groups: list[DealCatalogGroup] = []
    for owner_key, owner_records in groups.items():
        group_records = sorted(
            owner_records,
            key=lambda record: (
                record.catalog_code.lower(),
                record.catalog_label.lower(),
            ),
        )
        deal_groups.append(
            DealCatalogGroup(
                deal_owner=owner_labels[owner_key],
                records=group_records,
            )
        )
    return deal_groups


def _render_text_deal_blocks(
    group: CirculationGroup, deal_groups: list[DealCatalogGroup]
) -> str:
    blocks: list[str] = []
    for deal_group in deal_groups:
        deal_word = "deal" if _catalog_count(deal_group.records) == 1 else "deals"
        blocks.append(
            "\n".join(
                [
                    (
                        f"The {deal_group.deal_owner} office's {deal_word} "
                        f"with the below for the territory of {group.territory_name}."
                    ),
                    _catalog_bullets(deal_group.records),
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_html_deal_blocks(
    group: CirculationGroup, deal_groups: list[DealCatalogGroup]
) -> str:
    territory = escape(group.territory_name)
    blocks: list[str] = []
    for deal_group in deal_groups:
        deal_word = "deal" if _catalog_count(deal_group.records) == 1 else "deals"
        blocks.append(
            "".join(
                [
                    '<p style="margin: 0 0 0 0;">The ',
                    f"{escape(deal_group.deal_owner)} office's {deal_word} ",
                    f"with the below for the territory of {territory}.</p>",
                    '<ul style="margin-top: 0; margin-bottom: 12pt;">',
                    _catalog_list_items(deal_group.records),
                    "</ul>",
                ]
            )
        )
    return "".join(blocks)


def _render_subject(group: CirculationGroup, records: list[CueSheetRecord]) -> str:
    template_path = (
        _TV_TEMPLATE_PATH if group.media_type == "tv" else _FILM_TEMPLATE_PATH
    )
    template = _load_template(template_path)
    subject_template, _body_template = _split_template(template)
    return (
        subject_template.replace("{New/Revised}", group.revision_status)
        .replace("{catalog_code(s)}", _catalog_codes(records))
        .strip()
    )


def _render_text_body(
    group: CirculationGroup,
    records: list[CueSheetRecord],
    deal_owners: dict[str, str],
) -> str:
    media_label = "TV" if group.media_type == "tv" else "film"
    revision_label = "REVISED" if group.revision_status == "Revised" else ""
    revision_notes = "{revision_notes}" if group.revision_status == "Revised" else ""
    deal_blocks = _render_text_deal_blocks(
        group, _deal_catalog_groups(records, deal_owners)
    )

    lines = [
        "Hello:",
        "",
        f"The following link is for {revision_label} {media_label} cue sheets controlled via:",
        "",
        deal_blocks,
        "",
        (
            "Please submit the cue sheet to your society and check work "
            "registrations to ensure royalties for cinema performance and "
            "subsequent TV broadcasting are received. If your society is missing "
            "additional cue sheets, then please search on UMPG Cues in the first "
            "instance or if the documents you require cannot be located then send "
            "a request email to {cues_inbox}."
        ),
    ]
    if revision_notes:
        lines.extend(["", revision_notes])
    lines.extend(
        [
            "",
            "Please note that this link expires after 30 days.",
            "",
            "{link}",
        ]
    )
    rendered = "\n".join(lines)
    rendered = rendered.replace("for  TV cue sheets", "for TV cue sheets")
    rendered = rendered.replace("for  film cue sheets", "for film cue sheets")
    return rendered.strip() + "\n"


def _render_html_body(
    group: CirculationGroup,
    records: list[CueSheetRecord],
    cues_inbox: str,
    deal_owners: dict[str, str],
) -> str:
    media_label = "TV" if group.media_type == "tv" else "film"
    revised_prefix = "REVISED " if group.revision_status == "Revised" else ""
    revision_notes_html = (
        (
            '<p style="font-family: Calibri, Arial, sans-serif; font-size: 11pt; margin: 0 0 12pt 0;">'
            "{revision_notes}"
            "</p>"
        )
        if group.revision_status == "Revised"
        else ""
    )
    deal_blocks = _render_html_deal_blocks(
        group, _deal_catalog_groups(records, deal_owners)
    )
    request_paragraph = "".join(
        [
            '<p style="margin: 0 0 12pt 0;">Please submit the cue sheet to your ',
            "society and check work registrations to ensure royalties for cinema ",
            "performance and subsequent TV broadcasting are received. If your ",
            "society is missing additional cue sheets, then please search on UMPG ",
            "Cues in the first instance or if the documents you require cannot ",
            "be located then send a request email to ",
            f"{escape(cues_inbox)}.</p>",
        ]
    )

    return (
        "<html>"
        "<body>"
        '<div style="font-family: Calibri, Arial, sans-serif; font-size: 11pt;">'
        '<p style="margin: 0 0 12pt 0;">Hello:</p>'
        f'<p style="margin: 0 0 12pt 0;">The following link is for {revised_prefix}{media_label} cue sheets controlled via:</p>'
        f"{deal_blocks}"
        f"{request_paragraph}"
        f"{revision_notes_html}"
        '<p style="margin: 0 0 12pt 0;"><strong>Please note that this link expires after 30 days.</strong></p>'
        '<p style="margin: 0;">{link}</p>'
        "</div>"
        "</body>"
        "</html>"
    )


def _build_message(
    group: CirculationGroup,
    records: list[CueSheetRecord],
    cues_inbox: str,
    deal_owners: dict[str, str],
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = cues_inbox
    message["To"] = cues_inbox
    message["Subject"] = _render_subject(group, records)
    message.set_content(
        _render_text_body(group, records, deal_owners).replace(
            "{cues_inbox}", cues_inbox
        )
    )
    message.add_alternative(
        _render_html_body(group, records, cues_inbox, deal_owners),
        subtype="html",
    )
    return message


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "value"


def _output_stem(target: Path, group: CirculationGroup) -> str:
    return (
        f"{target.name}_"
        f"{_slug(group.media_type)}_"
        f"{_slug(group.territory_code)}_"
        f"{_slug(group.revision_status)}_"
        "circulation"
    )


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target = ctx.target_path
    if not target.is_dir():
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Circulation Drafts Canceled",
                "Info": "Select a directory to generate circulation drafts.",
            }
        )
        return

    cues_inbox = _resolve_cues_inbox(ctx, api)
    if not cues_inbox:
        return

    pdf_files = _collect_pdfs(target, api)
    if not pdf_files:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Circulation Drafts Complete",
                "Drafts Written": 0,
                "Skipped PDFs": 0,
                "Info": "No PDF files were found under the selected directory.",
            }
        )
        return

    grouped: dict[CirculationGroup, list[CueSheetRecord]] = {}
    skipped: list[str] = []
    total_files = len(pdf_files)
    territory_names = _load_territory_names(ctx)
    deal_owners = _load_deal_owners(ctx)

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        parsed = _parse_cue_sheet_path(pdf_path, target)
        if parsed is None:
            skipped.append(str(pdf_path.relative_to(target)))
        else:
            key = CirculationGroup(
                media_type=parsed.media_type,
                revision_status=parsed.revision_status,
                territory_code=parsed.territory_code,
                territory_name=_resolve_territory_name(
                    parsed.territory_code, territory_names
                ),
            )
            grouped.setdefault(key, []).append(parsed)
        api.progress(current=index, total=total_files, unit="files")

    if not grouped:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Circulation Drafts Complete",
                "Drafts Written": 0,
                "Skipped PDFs": len(skipped),
                "Info": "No PDFs matched the expected cue sheet folder layout.",
            }
        )
        return

    out_paths: list[str] = []
    for group in sorted(
        grouped,
        key=lambda item: (
            item.media_type,
            item.territory_code.lower(),
            item.revision_status,
        ),
    ):
        records = sorted(
            grouped[group],
            key=lambda record: (
                record.catalog_code.lower(),
                record.catalog_name.lower(),
            ),
        )
        out_path = build_destination(
            target.parent,
            _output_stem(target, group),
            _OUTPUT_SUFFIX,
        )
        message = _build_message(group, records, cues_inbox, deal_owners)
        out_path.write_bytes(message.as_bytes())
        out_paths.append(str(out_path))

    if skipped:
        preview = ", ".join(skipped[:5])
        api.log(
            "warn",
            f"Skipped {len(skipped)} PDF(s) that did not match the expected folder layout: {preview}",
        )

    api.emit_result(
        {
            "_title": "Circulation Drafts Complete",
            "Drafts Written": len(out_paths),
            "Skipped PDFs": len(skipped),
            "Output Files": "\n".join(out_paths),
        }
    )


if __name__ == "__main__":
    main()
