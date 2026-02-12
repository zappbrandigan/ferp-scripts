from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pdfplumber

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import build_destination, move_to_dir
from ferp.fscp.scripts.common.files import collect_files

_SUFFIX = ".pdf"
_NEEDS_OCR_DIRNAME = "_needs_ocr"
_CHECK_DIRNAME = "_check"

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*]')
_EXTRA_WHITE_SPACE = re.compile(r"\s{2,}")

_DATE_FORMATS = [
    "%B %d %Y",  # December 2 2021
    "%b %d %Y",  # Dec 2 2021
    "%m/%d/%Y",  # 12/2/2021
    "%m/%d/%y",  # 12/2/21
    "%Y-%m-%d",  # 2021-12-02
    "%d %b %Y",  # 2 Dec 2021
    "%d %B %Y",  # 2 December 2021
    "%d-%b-%Y",  # 02-Dec-2021
]


class UserResponse(TypedDict):
    value: str
    recursive: bool


def _sanitize_parts(name: str) -> str:
    cleaned = _INVALID_NAME_CHARS.sub("", name).strip()
    cleaned = _EXTRA_WHITE_SPACE.sub(" ", cleaned)
    return cleaned or "untitled"


def _extract_first_page_text(path: Path) -> str | None:
    with pdfplumber.open(path) as reader:
        try:
            return reader.pages[0].extract_text()
        except IndexError:
            return None


def _parse_air_date(raw: str) -> datetime:
    cleaned = raw.strip()
    cleaned = cleaned.replace(",", " ")
    cleaned = " ".join(cleaned.split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized air date format: '{raw}'")


def _is_episode_unique(production: str, episode: str) -> bool:
    return production.lower() != episode.lower()


def _derive_name(path: Path, text: str) -> str:
    air_match = re.search(
        r"initial\s*air\s*date:\s*(?P<ep_num>.+?)(?=\s+program\s*title:|\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    title_match = re.search(
        r"program\s*title:\s*(?P<prod_title>.+?)(?=\s+network:|\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not air_match or not title_match:
        raise ValueError("Program title or initial air date not found in cue sheet.")
    production_title = title_match.group("prod_title").strip()
    air_date = _parse_air_date(air_match.group("ep_num").strip()).strftime("%d%m%Y")

    ep_match = re.search(
        r"episode\s*title:\s*(?P<ep_title>.+?)(?=\s+program/show\s*duration|\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not ep_match:
        raise ValueError("Episode title not found in cue sheet.")
    ep_num_match = re.search(
        r"episode\s*number:\s*(?P<ep_number>[A-Za-z0-9][A-Za-z0-9_-]*)(?=\s|$)",
        text,
        re.IGNORECASE,
    )
    if not ep_num_match:
        raise ValueError("Episode number not found in cue sheet.")
    episode_title = ep_match.group("ep_title").strip()
    episode_number = ep_num_match.group("ep_number").strip()
    is_episode_unique = _is_episode_unique(production_title, episode_title)
    date_token = f"{air_date}-{episode_number}"

    if is_episode_unique:
        stem = f"{_sanitize_parts(production_title)}   {_sanitize_parts(episode_title)}  Ep No. {date_token}"
    else:
        stem = f"{_sanitize_parts(production_title)}   Ep No. {date_token}"

    return stem.strip()


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target = Path(ctx.target_path)
    base_dir = target if ctx.target_kind == "directory" else target.parent

    payload: UserResponse | dict = {}
    if ctx.target_kind == "directory":
        payload = api.request_input_json(
            "This action will rename one or more files. Continue?",
            id="ferp_rename_sports_confirm",
            fields=[
                {
                    "id": "recursive",
                    "type": "bool",
                    "label": "Recursive",
                    "default": False,
                },
            ],
            show_text_input=False,
            payload_type=UserResponse,
        )
    else:
        confirm = api.confirm(
            "This action will rename this file. Continue?",
            id="ferp_rename_sports_confirm",
        )
        if not confirm:
            api.emit_result({"_title": "Operation Cancelled"})
            return

    recursive = payload.get("recursive", False)
    pdf_files = collect_files(target, "*.pdf", recursive, check_cancel=api.check_cancel)

    if not pdf_files:
        api.log("info", f"No {_SUFFIX} files found to rename.")
        api.emit_result(
            {
                "_status": "warn",
                "_title": f"Warning: No {_SUFFIX} Files Found",
                "File Path": str(target),
            }
        )
        return

    renamed = 0
    skipped = 0
    needs_ocr = 0
    check = 0
    for index, pdf in enumerate(pdf_files, start=1):
        api.check_cancel()
        text = _extract_first_page_text(pdf)
        if not text:
            try:
                move_to_dir(pdf, base_dir / _NEEDS_OCR_DIRNAME)
                needs_ocr += 1
                api.log(
                    "info", f"Moved '{pdf.name}' to '{_NEEDS_OCR_DIRNAME}' (no text)."
                )
            except OSError as exc:
                api.log(
                    "error",
                    f"Failed to move '{pdf.name}' to '{_NEEDS_OCR_DIRNAME}': {exc}",
                )
        else:
            try:
                new_base = _derive_name(pdf, text)
            except (ValueError, AttributeError) as exc:
                skipped += 1
                check_dir = base_dir / _CHECK_DIRNAME
                try:
                    moved = move_to_dir(pdf, check_dir)
                    check += 1
                    api.log(
                        "warn",
                        f"Skipping '{pdf.name}': {exc}. Moved to '{_CHECK_DIRNAME}' as '{moved.name}'.",
                    )
                except OSError as move_exc:
                    api.log(
                        "error",
                        f"Skipping '{pdf.name}': {exc}. Failed to move to '{_CHECK_DIRNAME}': {move_exc}",
                    )
            else:
                new_name = f"{new_base}{_SUFFIX}"
                if pdf.name == new_name:
                    check_dir = base_dir / _CHECK_DIRNAME
                    try:
                        moved = move_to_dir(pdf, check_dir, base=new_base)
                        check += 1
                        api.log(
                            "warn",
                            f"Skipping '{pdf.name}': already matches target pattern. "
                            f"Moved to '{_CHECK_DIRNAME}' as '{moved.name}'.",
                        )
                    except OSError as exc:
                        api.log(
                            "error",
                            f"Skipping '{pdf.name}': already matches target pattern. "
                            f"Failed to move to '{_CHECK_DIRNAME}': {exc}",
                        )
                else:
                    base_destination = base_dir / new_name
                    if base_destination.exists():
                        check_dir = base_dir / _CHECK_DIRNAME
                        try:
                            existing_destination = move_to_dir(
                                base_destination, check_dir, base=new_base
                            )
                            incoming_destination = move_to_dir(
                                pdf, check_dir, base=new_base
                            )
                            check += 1
                            api.log(
                                "warn",
                                "Conflict for '{original}' -> '{target}'. "
                                "Moved existing to '{existing}' and incoming to '{incoming}'.".format(
                                    original=pdf.name,
                                    target=new_name,
                                    existing=existing_destination.name,
                                    incoming=incoming_destination.name,
                                ),
                            )
                        except OSError as exc:
                            api.log(
                                "error", f"Failed to move conflict '{pdf.name}': {exc}"
                            )
                    else:
                        destination = build_destination(base_dir, new_base, _SUFFIX)

                        try:
                            pdf.rename(destination)
                            renamed += 1
                            api.log(
                                "info", f"Renamed '{pdf.name}' -> '{destination.name}'"
                            )
                        except OSError as exc:
                            api.log("error", f"Failed to rename '{pdf.name}': {exc}")

        api.progress(
            current=index,
            total=len(pdf_files),
            unit="files",
            message=f"Renaming files in '{pdf.parent.name}'",
        )

    api.emit_result(
        {
            "_title": "Renaming Completed",
            "Renamed": renamed,
            "Skipped": skipped,
            "Needs OCR": needs_ocr,
            "Moved to Check": check,
            "Total Files": len(pdf_files),
        }
    )


if __name__ == "__main__":
    main()
