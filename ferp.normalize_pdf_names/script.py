from __future__ import annotations

import asyncio
import os
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Coroutine, Iterable, Literal, TypeVar

try:
    from googletrans import Translator as _GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    _GoogleTranslator = None
    _GOOGLETRANS_IMPORT_ERROR = True
else:
    _GOOGLETRANS_IMPORT_ERROR = False

from ferp.data.articles import languageArticles
from ferp.fscp.scripts import sdk

MAX_FILENAME_LENGTH = 60
PRODUCTION_DELIM = "   "
EPISODE_DELIM = "  "
SPACE_RUN_RE = re.compile(r"( {2,})")
T = TypeVar("T")


@dataclass
class ParsedName:
    production: str
    episode_title: str | None
    episode_info: str | None
    shape: Literal["film", "show_no_episode", "show_with_episode"]


@dataclass
class FileOutcome:
    kind: Literal["valid", "renamed", "moved"]
    source: Path
    destination: Path | None
    reason: str | None = None


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path
    if not root.exists() or not root.is_dir():
        raise ValueError("Select a directory before running this script.")

    outcomes: list[FileOutcome] = []
    check_dir = root / "_check"

    try:
        entries = list(_iter_directory(root))
    except PermissionError:
        raise ValueError("Permission denied while listing directory.")

    total_entries = len(entries) or 1

    for index, entry in enumerate(entries, start=1):
        path = Path(entry.path)
        
        if total_entries > 1:
            api.progress(current=index, total=total_entries, unit="files")

        if entry.is_symlink() and not path.exists():
            _record_move(outcomes, path, path, check_dir, "broken_symlink")
            continue

        stem = path.stem
        parsed, reason = _parse_name(stem)
        if not parsed:
            _record_move(outcomes, path, path, check_dir, reason or "unrepairable_structure")
            continue

        normalized, normalize_reason = _normalize_name(parsed)
        if not normalized:
            _record_move(outcomes, path, path, check_dir, normalize_reason or "unrepairable_structure")
            continue

        if normalized == stem:
            outcomes.append(FileOutcome("valid", path, None))
            continue

        target = path.with_name(f"{normalized}{path.suffix}")
        collision = False
        try:
            destination = _safe_rename(path, target)
        except PermissionError:
            _record_move(outcomes, path, path, check_dir, "permission_error")
            continue
        except OSError as exc:
            _record_move(outcomes, path, path, check_dir, f"other_error: {exc}")
            continue

        if destination != target:
            collision = True

        if collision:
            _record_move(outcomes, path, destination, check_dir, "collision")
        else:
            outcomes.append(FileOutcome("renamed", path, destination))


    summary = _build_report(root, outcomes)
    if _GOOGLETRANS_IMPORT_ERROR:
        api.log("warn", "googletrans not available; article repositioning skipped.")
    api.log("info", summary)
    api.emit_result({"report": summary})
    api.exit(code=0)


def _iter_directory(root: Path) -> Iterable[os.DirEntry[str]]:
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                is_file = entry.is_file(follow_symlinks=False)
                is_link = entry.is_symlink()
            except OSError:
                continue
            if not (is_file or is_link):
                continue
            if entry.name.lower().endswith(".pdf"):
                yield entry


def _parse_name(raw: str) -> tuple[ParsedName | None, str | None]:
    name = raw.strip()
    if not name:
        return None, "unrepairable_structure"

    matches = list(SPACE_RUN_RE.finditer(name))
    segments: list[str] = []
    last = 0
    for match in matches:
        segments.append(name[last:match.start()])
        last = match.end()
    segments.append(name[last:])
    segments = [segment.strip() for segment in segments]

    if any(not segment for segment in segments):
        return None, "unrepairable_structure"

    delim_count = len(matches)
    if delim_count == 0:
        return ParsedName(segments[0], None, None, "film"), None
    if delim_count == 1 and len(segments) == 2:
        return ParsedName(segments[0], None, segments[1], "show_no_episode"), None
    if delim_count == 2 and len(segments) == 3:
        return ParsedName(segments[0], segments[1], segments[2], "show_with_episode"), None
    return None, "ambiguous_delimiters"


def _normalize_name(parsed: ParsedName) -> tuple[str | None, str | None]:
    production = _normalize_spaces(parsed.production)
    if not production:
        return None, "unrepairable_structure"

    production = _reposition_article(production)
    production = production.upper()

    episode_title = None
    episode_info = None

    if parsed.shape == "show_no_episode":
        episode_info = (parsed.episode_info or "").strip()
        if not episode_info:
            return None, "unrepairable_structure"
    elif parsed.shape == "show_with_episode":
        raw_title = _normalize_spaces(parsed.episode_title or "")
        if not raw_title:
            return None, "unrepairable_structure"
        episode_title = _reposition_article(raw_title)
        episode_title = string.capwords(episode_title, sep=" ")

        episode_info = (parsed.episode_info or "").strip()
        if not episode_info:
            return None, "unrepairable_structure"
    else:
        episode_info = None

    parts = ParsedName(production, episode_title, episode_info, parsed.shape)

    length_reason = _enforce_length(parts)
    if length_reason:
        return None, length_reason

    formatted = _format_name(parts)
    return formatted, None


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split())


def _format_name(parts: ParsedName) -> str:
    if parts.shape == "film":
        return parts.production
    if parts.shape == "show_no_episode":
        return f"{parts.production}{PRODUCTION_DELIM}{parts.episode_info}"
    return f"{parts.production}{PRODUCTION_DELIM}{parts.episode_title}{EPISODE_DELIM}{parts.episode_info}"


def _enforce_length(parts: ParsedName) -> str | None:
    info = parts.episode_info or ""

    def total_length() -> int:
        if parts.shape == "film":
            return len(parts.production)
        if parts.shape == "show_no_episode":
            return len(parts.production) + len(PRODUCTION_DELIM) + len(info)
        return (
            len(parts.production)
            + len(PRODUCTION_DELIM)
            + len(parts.episode_title or "")
            + len(EPISODE_DELIM)
            + len(info)
        )

    if total_length() <= MAX_FILENAME_LENGTH:
        return None

    if parts.shape == "film":
        shortened = _shorten(parts.production, MAX_FILENAME_LENGTH, min_visible=1)
        if not shortened:
            return "length_limit"
        parts.production = shortened
        return None

    if parts.shape == "show_no_episode":
        allowed = MAX_FILENAME_LENGTH - (len(PRODUCTION_DELIM) + len(info))
        if allowed < 3:
            return "length_limit"
        shortened = _shorten(parts.production, allowed, min_visible=1)
        if not shortened:
            return "length_limit"
        parts.production = shortened
        return None if total_length() <= MAX_FILENAME_LENGTH else "length_limit"

    # show_with_episode
    available_for_episode = MAX_FILENAME_LENGTH - (
        len(parts.production) + len(PRODUCTION_DELIM) + len(EPISODE_DELIM) + len(info)
    )
    if available_for_episode < 6:
        return "length_limit"
    updated_episode = _shorten(parts.episode_title or "", available_for_episode, min_visible=3)
    if not updated_episode:
        return "length_limit"
    parts.episode_title = updated_episode

    if total_length() <= MAX_FILENAME_LENGTH:
        return None

    available_for_production = MAX_FILENAME_LENGTH - (
        len(PRODUCTION_DELIM) + len(parts.episode_title or "") + len(EPISODE_DELIM) + len(info)
    )
    if available_for_production < 3:
        return "length_limit"
    shortened_prod = _shorten(parts.production, available_for_production, min_visible=1)
    if not shortened_prod:
        return "length_limit"
    parts.production = shortened_prod
    return None if total_length() <= MAX_FILENAME_LENGTH else "length_limit"


def _shorten(text: str, limit: int, *, min_visible: int) -> str | None:
    if len(text) <= limit:
        return text
    visible = limit - 3
    if visible < min_visible:
        return None
    base = text[:visible]
    trimmed = _trim_trailing_special(base, min_visible)
    return f"{trimmed}..."


def _trim_trailing_special(text: str, min_visible: int) -> str:
    chars = list(text)
    while len(chars) > min_visible and chars and not chars[-1].isalnum():
        chars.pop()
    return "".join(chars)


@lru_cache(maxsize=256)
def _detect_language(text: str) -> str | None:
    if _GoogleTranslator is None:
        return None
    snippet = text.strip()
    if not snippet:
        return None
    try:
        result = _run_async(lambda: _detect_language_async(snippet))
    except Exception:
        return None
    lang = getattr(result, "lang", None)
    if not isinstance(lang, str):
        return None
    if lang not in languageArticles:
        return None
    return lang


def _reposition_article(text: str) -> str:
    lang = _detect_language(text)
    if not lang:
        return text

    articles = languageArticles.get(lang, [])
    tokens = text.split()
    if not tokens:
        return text

    first = tokens[0]
    article = _match_article(first, articles)
    if not article:
        return text

    rest = " ".join(tokens[1:]).strip()
    if not rest:
        return text

    return f"{rest}, {tokens[0]}"


def _match_article(token: str, articles: list[str]) -> str | None:
    for article in articles:
        if token.lower() == article.lower():
            return article
    return None


async def _detect_language_async(text: str):
    assert _GoogleTranslator is not None
    async with _GoogleTranslator() as translator:
        return await translator.detect(text)


def _run_async(factory: Callable[[], Coroutine[object, object, T]]) -> T:
    try:
        return asyncio.run(factory())
    except RuntimeError as exc:
        if "asyncio.run()" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(factory())
        finally:
            loop.close()


def _safe_rename(source: Path, target: Path) -> Path:
    if source == target:
        return target

    if target.exists():
        try:
            if target.samefile(source):
                temp = source.with_name(f".__tmp__{source.name}")
                source.rename(temp)
                temp.rename(target)
                return target
        except FileNotFoundError:
            pass

    if target.exists():
        base_stem = target.stem
        suffix = target.suffix
        counter = 2
        while True:
            candidate = target.with_name(f"{base_stem}_{counter}{suffix}")
            if not candidate.exists():
                target = candidate
                break
            counter += 1

    source.rename(target)
    return target


def _move_to_check(source: Path, check_dir: Path, target_name: str | None) -> Path:
    check_dir.mkdir(exist_ok=True)
    desired_name = target_name or source.name
    destination = check_dir / desired_name
    if destination.exists():
        stem = Path(desired_name).stem
        suffix = Path(desired_name).suffix
        counter = 2
        while True:
            candidate = check_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                destination = candidate
                break
            counter += 1
    source.rename(destination)
    return destination


def _record_move(
    outcomes: list[FileOutcome],
    original_path: Path,
    current_path: Path,
    check_dir: Path,
    reason: str,
    target_name: str | None = None,
) -> None:
    try:
        dest = _move_to_check(current_path, check_dir, target_name)
    except PermissionError:
        outcomes.append(FileOutcome("moved", original_path, None, "permission_error"))
    except OSError as exc:
        outcomes.append(FileOutcome("moved", original_path, None, f"other_error: {exc}"))
    else:
        outcomes.append(FileOutcome("moved", original_path, dest, reason))


def _build_report(root: Path, outcomes: list[FileOutcome]) -> str:
    valid = [out for out in outcomes if out.kind == "valid"]
    renamed = [out for out in outcomes if out.kind == "renamed"]
    moved = [out for out in outcomes if out.kind == "moved"]

    lines = [
        f"\nValid: {len(valid)}",
        f"Renamed: {len(renamed)}",
        f"Moved: {len(moved)}",
    ]

    def add_blocks(title: str, blocks: list[list[tuple[str, str]]]) -> None:
        if not blocks:
            return
        lines.append("---")
        lines.append(title)
        for block in blocks:
            for label, value in block:
                lines.append(f"{label}: {value}")
            lines.append("---")

    renamed_blocks = [
        [
            ("from", _rel(root, out.source)),
            ("to", _rel(root, out.destination)),
        ]
        for out in renamed
    ]
    moved_blocks = [
        [
            ("from", _rel(root, out.source)),
            ("to", _rel(root, out.destination)),
            ("reason", out.reason or ""),
        ]
        for out in moved
    ]

    add_blocks("Renamed Files", renamed_blocks)
    add_blocks("Moved to _check", moved_blocks)

    if lines and lines[-1] == "---":
        lines.pop()

    return "\n".join(lines)


def _rel(root: Path, path: Path | None) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
