from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from ferp.fscp.scripts import sdk

_SUFFIX = ".pdf"
_NEEDS_OCR_DIRNAME = "_needs_ocr"
_CHECK_DIRNAME = "_check"

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def _sanitize_filename(name: str) -> str:
    cleaned = _INVALID_NAME_CHARS.sub("_", name).strip()
    return cleaned or "untitled"


def _extract_first_page_text(path: Path) -> str | None:
    with pdfplumber.open(path) as reader:
        try:
            return reader.pages[0].extract_text()
        except IndexError:
            return None


def _normalize_year(year: str) -> str:
    if len(year) == 2:
        return f"20{year}"
    return year


def _extract_episode_value(text: str) -> str:
    episode_match = re.search(r"Episode Number:\s*([^\s]+)", text)
    if episode_match:
        episode_number = episode_match.group(1)
        if re.fullmatch(r"\d+", episode_number) and 3 <= len(episode_number) <= 5:
            return episode_number

    air_date_match = re.search(
        r"(?:Air Date|Air/Release Date):\s*(?P<month>\d{1,2})/"
        r"(?P<day>\d{1,2})/(?P<year>\d{2,4})",
        text,
    )
    if not air_date_match:
        raise ValueError("No suitable episode number or air date found on page 1.")
    month, day, year = air_date_match.groups()
    day = f"{int(day):02d}"
    month = f"{int(month):02d}"
    year = _normalize_year(year)
    return f"{day}{month}{year}"


def _derive_name(path: Path, text: str) -> str:
    title_match = re.search(r"(?P<pd_title>.+?)\s-", path.name)
    if not title_match:
        raise ValueError("Filename does not match expected '<title> -' pattern.")
    production_title = title_match.group(1)

    episode_value = _extract_episode_value(text)

    stem = f"{production_title}   Ep No. {episode_value}"

    return _sanitize_filename(stem.strip())


def _build_destination(directory: Path, base: str, suffix: str) -> Path:
    candidate = directory / f"{base}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _move_to_dir(path: Path, directory: Path, base: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name_base = base if base is not None else path.stem
    destination = _build_destination(directory, name_base, path.suffix)
    return path.rename(destination)


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = Path(ctx.target_path)

    if not target_dir.exists() or not target_dir.is_dir():
        raise RuntimeError(f"Target directory '{target_dir}' is not accessible.")

    pdf_files = sorted(
        [
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() == _SUFFIX
        ],
        key=lambda item: item.name.lower(),
    )

    if not pdf_files:
        api.log("info", f"No {_SUFFIX} files found to rename.")
        api.emit_result({"renamed": 0})
        return

    renamed = 0
    skipped = 0
    needs_ocr = 0
    check = 0
    for index, pdf in enumerate(pdf_files, start=1):
        text = _extract_first_page_text(pdf)
        if not text:
            try:
                _move_to_dir(pdf, target_dir / _NEEDS_OCR_DIRNAME)
                needs_ocr += 1
                api.log(
                    "info", f"Moved '{pdf.name}' to '{_NEEDS_OCR_DIRNAME}' (no text)."
                )
            except OSError as exc:
                api.log(
                    "error",
                    f"Failed to move '{pdf.name}' to '{_NEEDS_OCR_DIRNAME}': {exc}",
                )
            api.progress(current=index, total=len(pdf_files), unit="files")
            continue

        try:
            new_base = _derive_name(pdf, text)
        except (ValueError, AttributeError) as exc:
            skipped += 1
            check_dir = target_dir / _CHECK_DIRNAME
            try:
                moved = _move_to_dir(pdf, check_dir)
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
            api.progress(current=index, total=len(pdf_files), unit="files")
            continue

        new_name = f"{new_base}{_SUFFIX}"
        if pdf.name == new_name:
            check_dir = target_dir / _CHECK_DIRNAME
            try:
                moved = _move_to_dir(pdf, check_dir, new_base)
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
            api.progress(current=index, total=len(pdf_files), unit="files")
            continue

        base_destination = target_dir / new_name
        if base_destination.exists():
            check_dir = target_dir / _CHECK_DIRNAME
            try:
                existing_destination = _move_to_dir(
                    base_destination, check_dir, new_base
                )
                incoming_destination = _move_to_dir(pdf, check_dir, new_base)
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
                api.log("error", f"Failed to move conflict '{pdf.name}': {exc}")
            api.progress(current=index, total=len(pdf_files), unit="files")
            continue

        destination = _build_destination(target_dir, new_base, _SUFFIX)

        try:
            pdf.rename(destination)
            renamed += 1
            api.log("info", f"Renamed '{pdf.name}' -> '{destination.name}'")
        except OSError as exc:
            api.log("error", f"Failed to rename '{pdf.name}': {exc}")

        api.progress(current=index, total=len(pdf_files), unit="files")

    api.emit_result(
        {
            "renamed": renamed,
            "skipped": skipped,
            "needs_ocr": needs_ocr,
            "check": check,
            "total": len(pdf_files),
        }
    )


if __name__ == "__main__":
    main()
