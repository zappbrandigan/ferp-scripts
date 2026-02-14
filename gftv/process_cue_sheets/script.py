from __future__ import annotations

import getpass
import json
import logging
import math
import os
import re
import shutil
import tempfile
import unicodedata
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import pdfplumber
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from pypdf.errors import PdfReadWarning
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    PdfObject,
    StreamObject,
    TextStringObject,
)

# from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from typing_extensions import Literal

from ferp.fscp.scripts import sdk
from ferp.fscp.scripts.common import collect_files

warnings.filterwarnings("ignore", category=PdfReadWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Used to suppress false positives when the surrounding text indicates logos or EE usage.
LOGO_RE = re.compile(r"(?<![A-Z0-9])logos?(?![A-Z0-9])", re.IGNORECASE)
_CONTEXT_EE_RE = re.compile(r"(?<!\S)EE(?!\S)")
_LICENSE_NOTE_RE = re.compile(r"(sync\s*license|master\s*license)", re.IGNORECASE)
ADMINISTRATOR_NAME = ""
STAMP_SPEC_VERSION = ""


def parse_board_description_yaml(description: str) -> dict[str, str]:
    if not description:
        return {}

    parsed: dict[str, str] = {}
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
            value = value[1:-1]
        if key:
            parsed[key] = value
    return parsed


class UserResponse(TypedDict):
    value: str
    recursive: bool
    in_place: bool
    adjust_header: bool
    manual_select: bool
    custom_stamp: bool


class PublisherSelectionResponse(TypedDict):
    value: str
    selected_pubs: list[str]


class SplitTerritorySelectionResponse(TypedDict):
    value: str
    territory_codes: list[str]
    use_default_split_selection: bool


class EffectiveDateEntry(TypedDict):
    date: str
    territories: list[str]


class AgreementEntry(TypedDict):
    publishers: list[str]
    effective_dates: list[EffectiveDateEntry]


class CombinedTerritoryEntry(TypedDict):
    dates: list[str]
    groups: set[int]


# =================================================
# Soundmouse
# =================================================

SOUNDMOUSE_COLUMNS: List[Tuple[str, float, float]] = [
    ("cue #", 0, 39),
    ("reel #", 40, 65),
    ("title", 97, 230),
    ("role", 231, 267),
    ("name", 268, 400),
    ("society", 401, 480),
    ("usage", 481, 515),
    ("duration", 516, 600),
]

SOUNDMOUSE_HEADER_RE = re.compile(
    r"#\s*Reel\s*No\s*Cue\s*Title\s*Role\s*Name\s*Society\s*Usage\s*Duration",
    re.IGNORECASE,
)


def is_soundmouse_pdf(text: str) -> bool:
    if not text:
        return False

    score = 0

    # Strong signal: exact column header sequence
    if SOUNDMOUSE_HEADER_RE.search(text):
        score += 3

    # Supporting signals
    if re.search(r"\bSOCAN\s*\[\s*100%\s*\]", text, re.IGNORECASE):
        score += 1
    if re.search(r"\bBMI\s*\[\s*100%\s*\]", text, re.IGNORECASE):
        score += 1
    if re.search(r"\bEIDR\b", text):
        score += 1

    # Role glyphs as standalone tokens
    if re.search(r"(?m)^\s*C\s+", text) and re.search(r"(?m)^\s*E\s+", text):
        score += 1

    return score >= 3


def sm_extract_words(page):
    return page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=1,
        y_tolerance=3,
    )


def sm_cluster_rows(words, y_tol: float = 3):
    rows = []
    for w in sorted(words, key=lambda w: w["top"]):
        for row in rows:
            if abs(row[0]["top"] - w["top"]) <= y_tol:
                row.append(w)
                break
        else:
            rows.append([w])

    for row in rows:
        row.sort(key=lambda w: w["x0"])
    return rows


def sm_assign_column(x: float) -> Optional[str]:
    for name, x0, x1 in SOUNDMOUSE_COLUMNS:
        if x0 <= x <= x1:
            return name
    return None


def sm_find_table_start_y(rows) -> Optional[float]:
    for row in rows:
        text = " ".join(w["text"] for w in row)
        if SOUNDMOUSE_HEADER_RE.search(text):
            return row[0]["top"]
    return None


def parse_soundmouse(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> list[Dict[str, Any]]:
    """
    Parse Soundmouse cue sheets into a list of cue dictionaries.

    Grouping rule:
      - A row with a non-empty "cue #" starts a new cue dict.
      - Rows without "cue #" are continuations of the current cue (e.g., more roles/names).

    Output schema (similar spirit to RapidCue):
      {
        "cue": str,
        "reel": str | None,
        "title": str | None,
        "usage": str | None,
        "duration": str | None,
        "composers": [{"name": ..., "society": ...}],
        "publishers": [{"name": ..., "society": ...}],
        "notes": ["Note", "Note", ...],
      }
    """
    cues: list[Dict[str, Any]] = []
    current_cue: Optional[Dict[str, Any]] = None
    last_seen_role: Optional[str] = None

    def flush_current():
        nonlocal current_cue
        if current_cue:
            # Optional: drop completely empty shells
            has_identity = any(
                current_cue.get(k) for k in ("cue", "title", "usage", "duration")
            )
            has_people = bool(
                current_cue.get("composers") or current_cue.get("publishers")
            )
            if has_identity or has_people:
                cues.append(current_cue)
        current_cue = None

    def norm_join(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    pages_scanned = 0
    pages_with_tables = 0
    words_total = 0
    rows_total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            pages_scanned += 1
            words = sm_extract_words(page)
            if not words:
                continue
            words_total += len(words)

            rows = sm_cluster_rows(words)

            table_start_y = sm_find_table_start_y(rows)
            if table_start_y is None:
                continue

            pages_with_tables += 1
            table_words = [w for w in words if w["top"] > table_start_y]
            table_rows = sm_cluster_rows(table_words)

            for row in table_rows:
                if check_cancel is not None:
                    check_cancel()
                rows_total += 1
                record = {name: [] for name, *_ in SOUNDMOUSE_COLUMNS}
                for w in row:
                    col = sm_assign_column(w["x0"])
                    if col:
                        record[col].append(w["text"])

                # compact: only keep non-empty joined fields
                compact = {k: norm_join(" ".join(v)) for k, v in record.items() if v}
                if not compact:
                    continue

                cue_no = compact.get("cue #")
                reel_no = compact.get("reel #")
                title = compact.get("title")
                role = compact.get("role")
                name = compact.get("name")
                society = compact.get("society")
                usage = compact.get("usage")
                duration = compact.get("duration")
                # New cue boundary: cue # present
                if cue_no:
                    flush_current()
                    current_cue = {
                        "cue": cue_no,
                        "reel": reel_no,
                        "title": title,
                        "usage": usage,
                        "duration": duration,
                        "composers": [],
                        "publishers": [],
                        "notes": [],
                    }
                    last_seen_role = None
                else:
                    # Continuation line: if we don't have a current cue yet, ignore it
                    if not current_cue:
                        continue
                    # Some continuation lines may repeat title/usage/duration; fill if missing
                    if not current_cue.get("title") and title:
                        current_cue["title"] = title
                    if not current_cue.get("usage") and usage:
                        current_cue["usage"] = usage
                    if not current_cue.get("duration") and duration:
                        current_cue["duration"] = duration
                    if reel_no and title and (not duration or not usage):
                        current_cue["notes"].append(
                            reel_no.strip() + " " + title.strip()
                        )

                # At this point we have a current cue; attach contributor if present
                if current_cue and name:
                    entry = {"name": name, "society": society}
                    r = role.strip().upper() if role else ""

                    if r in ["C", "CA", "A", "AR"]:
                        last_seen_role = r
                        current_cue["composers"].append(entry)
                    elif r == "E":
                        last_seen_role = r
                        current_cue["publishers"].append(entry)
                    elif r == "AM":
                        last_seen_role = r
                    else:
                        # Handle text line wrape with no role by using last seen role in this cue
                        if re.match(r"\d{5,}$", name):
                            # Ignore ISRC codes mistakenly parsed as names
                            continue
                        elif last_seen_role in ["C", "CA", "A", "AR"]:
                            current_cue["composers"][-1]["name"] += " " + name
                        elif last_seen_role == "E":
                            current_cue["publishers"][-1]["name"] += " " + name

        flush_current()

    if log_fn:
        log_fn(
            "soundmouse parser: "
            f"pages={pages_scanned} | pages_with_table={pages_with_tables} | "
            f"words={words_total} | rows={rows_total} | cues={len(cues)}"
        )
        # log_fn(f"soundmouse text: cues={cues}")
    return cues


# =================================================
# WB
# =================================================

WB_COLUMNS: List[Tuple[str, float, float]] = [
    ("no", 0, 35),
    ("selection", 40, 140),
    ("composer", 145, 275),
    ("publisher", 280, 480),
    ("how used", 485, 555),
    ("time", 558, 600),
]

WB_HEADER_RE = re.compile(
    r"no\s*selection\s*composer\s*publisher\s*how\sused\s*time",
    re.IGNORECASE,
)


def is_wb_pdf(text: str) -> bool:
    if not text:
        return False

    score = 0

    # Strong signal: exact column header sequence
    if WB_HEADER_RE.search(text):
        score += 3

    # Supporting signals
    if re.search(r"\bcomposer\b", text, re.IGNORECASE) and re.search(
        r"\bpublisher\b", text, re.IGNORECASE
    ):
        score += 1
    if re.search(r"\bselection\b", text, re.IGNORECASE):
        score += 1
    if re.search(r"\bhow\s*used\b", text, re.IGNORECASE):
        score += 1

    return score >= 3


def wb_extract_words(page):
    return page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=1,
        y_tolerance=3,
    )


def wb_cluster_rows(words, y_tol: float = 3):
    rows = []
    for w in sorted(words, key=lambda w: w["top"]):
        for row in rows:
            if abs(row[0]["top"] - w["top"]) <= y_tol:
                row.append(w)
                break
        else:
            rows.append([w])

    for row in rows:
        row.sort(key=lambda w: w["x0"])
    return rows


def wb_assign_column(x: float) -> Optional[str]:
    for name, x0, x1 in WB_COLUMNS:
        if x0 <= x <= x1:
            return name
    return None


def wb_find_table_start_y(rows) -> Optional[float]:
    for row in rows:
        text = " ".join(w["text"] for w in row)
        if WB_HEADER_RE.search(text):
            return row[0]["top"]
    return None


def parse_wb(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> list[Dict[str, Any]]:
    """
    Parse WB cue sheets into a list of cue dictionaries.

    Grouping rule:
      - A row with a non-empty "no" starts a new cue dict.
      - Rows without "no" are continuations of the current cue.
      - publisher names are all combined in a single string in publisher -> name.

    Output schema:
      {
        "no": str,
        "selection": str | None,
        "composers": str | None,
        "publishers": [{"name": ...}],
        "duration": str | None,
        "usage": str | None,
      }
    """
    cues: list[Dict[str, Any]] = []
    current_cue: Optional[Dict[str, Any]] = None

    def flush_current():
        nonlocal current_cue
        if current_cue:
            # Optional: drop completely empty shells
            has_identity = any(
                current_cue.get(k) for k in ("selection", "usage", "duration")
            )
            has_people = bool(
                current_cue.get("composers") or current_cue.get("publishers")
            )
            if has_identity or has_people:
                cues.append(current_cue)
        current_cue = None

    def norm_join(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    pages_scanned = 0
    pages_with_tables = 0
    words_total = 0
    rows_total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            pages_scanned += 1
            words = wb_extract_words(page)
            if not words:
                continue
            words_total += len(words)

            rows = wb_cluster_rows(words)

            table_start_y = wb_find_table_start_y(rows)
            if table_start_y is None:
                continue

            pages_with_tables += 1
            table_words = [w for w in words if w["top"] > table_start_y]
            table_rows = wb_cluster_rows(table_words)

            for row in table_rows:
                if check_cancel is not None:
                    check_cancel()
                rows_total += 1
                record = {name: [] for name, *_ in WB_COLUMNS}
                for w in row:
                    col = wb_assign_column(w["x0"])
                    if col:
                        record[col].append(w["text"])

                # compact: only keep non-empty joined fields
                compact = {k: norm_join(" ".join(v)) for k, v in record.items() if v}
                if not compact:
                    continue

                cue_no = compact.get("no", "")
                selection = compact.get("selection", "")
                composer = compact.get("composer", "")
                publisher = compact.get("publisher", "")
                usage = compact.get("how used", "")
                time = compact.get("time", "")
                # New cue boundary: cue # present
                if cue_no:
                    flush_current()
                    current_cue = {
                        "no": cue_no,
                        "selection": selection,
                        "composers": composer,
                        "publishers": [{"name": publisher}] if publisher else [],
                        "usage": usage,
                        "time": time,
                    }
                else:
                    # Continuation line: if we don't have a current cue yet, ignore it
                    if not current_cue:
                        continue

                # At this point we have a current cue; attach contributor if present
                if current_cue and composer and not cue_no:
                    current_cue["composers"] = current_cue["composers"] + " " + composer

                if current_cue and publisher and not cue_no:
                    if current_cue["publishers"]:
                        current_cue["publishers"][0]["name"] += " " + publisher
                    else:
                        current_cue["publishers"] = [{"name": publisher}]

                if current_cue and usage and not cue_no:
                    current_cue["usage"] = current_cue["usage"] + " " + usage

                if current_cue and selection and not cue_no:
                    current_cue["selection"] = (
                        current_cue["selection"] + " " + selection
                    )

        flush_current()

    if log_fn:
        log_fn(
            "wb parser: "
            f"pages={pages_scanned} | pages_with_table={pages_with_tables} | "
            f"words={words_total} | rows={rows_total} | cues={len(cues)}"
        )
        # log_fn(f"wb text: cues={cues}")
    return cues


# =================================================
# RapidCue
# =================================================

RAPIDCUE_COPYRIGHT_RE = re.compile(r"©\s*\d{4}\s*RapidCue", re.IGNORECASE)
# RAPIDCUE_PAGE_RE = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
RAPIDCUE_HEADER_HINT_RE = re.compile(
    r"SEQ#\s*CUE#\s*TITLE\s*COMPOSER\/PUBLISHER\s*AFFILIATION\s*\%\s*USAGE\s*TIME",
    re.IGNORECASE,
)
RAPIDCUE_HELPER_HING_RE = re.compile(
    r"Roles:\s+C\s*-\s*Composer\s+E\s*-\s*Publisher\s+AM\s*-\s*Administrator\s+SE",
    re.IGNORECASE,
)

# Cue header (SEQ TITLE USAGE TIME)
RC_CUE_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?P<seq>\d+)\s+
    (?P<title>.+?)\s+
    (?P<usage>VI|VV|BI|BV|SRC)\s+
    (?P<time>\d+:\d+)
    \s*$
    """,
    re.VERBOSE,
)

# Role line start (after descriptor peeling)
RC_ROLE_START_RE = re.compile(r"^(C|A|CA|AR|E|AM)\s+(.+)$")

# Extract society + percent from END of role block
RC_SOCIETY_PERCENT_RE = re.compile(r"(.*?)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)$")

# Known RapidCue descriptor prefixes
RC_DESCRIPTOR_TERMS = {
    "THEME",
    "MAIN TITLE",
    "OPENING",
    "CLOSING",
    "SEGMENT",
    "BUMPER",
    "LOGO",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


def is_rapidcue_pdf(text: str) -> bool:
    if not text:
        return False
    if RAPIDCUE_COPYRIGHT_RE.search(text):
        return True
    if RAPIDCUE_HEADER_HINT_RE.search(text):
        return True
    if RAPIDCUE_HELPER_HING_RE.search(text):
        return True
    return False


def is_rapidcue_clutter(line: str) -> bool:
    s = _norm(line)

    # Page header/footer
    if re.search(r"© \d{4} RAPIDCUE", s):
        return True
    if re.search(r"\bPAGE \d+ OF \d+\b", s):
        return True
    if s.startswith("MUSIC CUE SHEET:"):
        return True

    # Legends / documentation blocks
    if s == "ROLES:":
        return True
    if s.startswith("USAGES:"):
        return True

    # Comment / artist notes
    if re.search(r"comments?:", s, flags=re.IGNORECASE):
        return True
    if re.search(r"artists?:", s, flags=re.IGNORECASE):
        return True

    # Role legend lines
    if re.search(r"\bC\s*-\s*COMPOSER\b", s) and re.search(r"\bE\s*-\s*PUBLISHER\b", s):
        return True
    if re.search(r"\bAD\s*-\s*ADAPTER\b", s) or re.search(r"\bAR\s*-\s*ARRANGER\b", s):
        return True
    if re.search(r"\bCA\s*-\s*COMPOSER/AUTHOR\b", s) or re.search(
        r"\bSA\s*-\s*SUBAUTHOR\b", s
    ):
        return True

    # Table header (RapidCue)
    if "SEQ# CUE# TITLE" in s:
        return True
    if (
        "COMPOSER/PUBLISHER" in s
        and "AFFILIATION" in s
        and "USAGE" in s
        and "TIME" in s
    ):
        return True

    return False


def rc_extract_lines(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> List[str]:
    """Extract RapidCue lines, stripping headers/footers and non-data clutter."""
    lines: List[str] = []
    pages_scanned = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            pages_scanned += 1
            text = page.extract_text(x_tolerance=1) or ""
            for line in text.splitlines():
                if check_cancel is not None:
                    check_cancel()
                line = line.strip()
                if not line:
                    continue
                if is_rapidcue_clutter(line):
                    continue
                lines.append(line)
    if log_fn:
        log_fn(f"rapidcue parser: pages={pages_scanned} | extracted_lines={len(lines)}")
        # log_fn(f"rapidcue parser: extracted_text={lines}")
    return lines


def rc_peel_descriptor_prefix(line: str) -> Tuple[Optional[str], str]:
    """
    If a RapidCue descriptor prefixes the line, remove it and return:
      (descriptor_or_None, remaining_line)

    IMPORTANT: Only peel if it starts with '<TERM> ' (space), to avoid false matches.
    """
    stripped = line.strip()
    upper = stripped.upper()

    for term in sorted(RC_DESCRIPTOR_TERMS, key=len, reverse=True):
        prefix = term + " "
        if upper.startswith(prefix):
            return term.title(), stripped[len(prefix) :].lstrip()

    return None, stripped


def parse_rapidcue(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> list[Dict[str, Any]]:
    """Parse RapidCue cue sheets into structured cue dictionaries."""
    lines = rc_extract_lines(pdf_path, log_fn=log_fn, check_cancel=check_cancel)
    cues: List[Dict[str, Any]] = []

    current_cue: Optional[Dict[str, Any]] = None
    current_role: Optional[str] = None
    role_buffer: List[str] = []

    def flush_role():
        nonlocal current_role, role_buffer, current_cue
        if not current_cue:
            current_role = None
            role_buffer = []
            return
        if not current_role or not role_buffer:
            current_role = None
            role_buffer = []
            return

        text = " ".join(role_buffer).strip()
        m = RC_SOCIETY_PERCENT_RE.search(text)
        if m:
            name = m.group(1).strip()
            society = m.group(2)
            percent = m.group(3)
        else:
            name = text
            society = None
            percent = None

        entry = {"name": name, "society": society, "percent": percent}

        if current_role == "C":
            current_cue["composers"].append(entry)
        elif current_role == "E":
            current_cue["publishers"].append(entry)

        current_role = None
        role_buffer = []

    for line in lines:
        if check_cancel is not None:
            check_cancel()
        # Cue header
        m = RC_CUE_HEADER_RE.match(line)
        if m:
            flush_role()
            if current_cue:
                cues.append(current_cue)

            current_cue = {
                "seq": m.group("seq"),
                "title": m.group("title"),
                "usage": m.group("usage"),
                "time": m.group("time"),
                "composers": [],
                "publishers": [],
                "notes": [],
            }
            continue

        if not current_cue:
            continue

        # Peel descriptor prefix (e.g., "Bumper E ...")
        descriptor, remainder = rc_peel_descriptor_prefix(line)
        if descriptor:
            current_cue["notes"].append(descriptor)

        # Role start
        m = RC_ROLE_START_RE.match(remainder)
        if m:
            flush_role()
            current_role = m.group(1)
            role_buffer = [m.group(2)]
            continue

        # Role continuation (wrapped lines)
        if current_role:
            role_buffer.append(remainder)
            continue

        # ignore anything else

    flush_role()
    if current_cue:
        cues.append(current_cue)

    if log_fn:
        log_fn(f"rapidcue parser: cues={len(cues)}")
        # log_fn(f"rapidcue parser: cues={cues}")
    return cues


# =================================================
# Default parser (unknown format)
# =================================================
def default_extract_lines(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> List[str]:
    """Extract readable lines for the fallback parser."""
    lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            text = page.extract_text(x_tolerance=1) or ""
            for line in text.splitlines():
                if check_cancel is not None:
                    check_cancel()
                line = line.strip()
                if not line:
                    continue
                if sum(ch.isalpha() for ch in line) < 3:
                    continue
                lines.append(line)
    if log_fn:
        log_fn(f"default parser: extracted_lines={len(lines)}")
    return lines


def parse_default(
    pdf_path: Path,
    controlled_publishers: list[str],
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
    logo_context_chars: int = 180,
) -> list[Dict[str, Any]]:
    """
    Fallback parser for unknown formats.

    Strategy:
      - Canonicalize the full extracted text.
      - Search for canonical publisher names in that canonical string.
      - Suppress matches when nearby text suggests a logo/EE context.
      - Avoid substring collisions by preferring the longest matching publisher name.

    The logo_context_chars window is intentionally wide to catch logos that appear
    near a publisher credit but not necessarily adjacent.
    """
    cues: List[Dict[str, Any]] = []
    controlled_norm = [
        (pub, canonicalize_publisher_name(pub, drop_suffixes=True))
        for pub in controlled_publishers
    ]
    canon_names = sorted({canon for _, canon in controlled_norm if canon})
    # Precompute longer-name candidates so shorter matches don't fire inside them.
    longer_by_prefix: dict[str, list[str]] = {}
    for canon in canon_names:
        prefix = canon + " "
        longer = [other for other in canon_names if other.startswith(prefix)]
        if longer:
            longer_by_prefix[canon] = sorted(longer, key=len, reverse=True)
    lines = default_extract_lines(pdf_path, log_fn=log_fn, check_cancel=check_cancel)
    full_text = " ".join(lines)
    canon_text = canonicalize_text_for_search(full_text)
    matched_publishers_count = 0
    skipped_logo_matches = 0
    seen_publishers: set[str] = set()

    def _find_phrase(text: str, phrase: str, start_at: int) -> int:
        # Require space or string boundaries so we don't match inside tokens.
        start = start_at
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                return -1
            end = idx + len(phrase)
            if (idx == 0 or text[idx - 1] == " ") and (
                end == len(text) or text[end] == " "
            ):
                return idx
            start = end

    for raw_name, norm_name in controlled_norm:
        if check_cancel is not None:
            check_cancel()
        if not norm_name or raw_name in seen_publishers:
            continue
        start = 0
        matched = False
        while True:
            idx = _find_phrase(canon_text, norm_name, start)
            if idx == -1:
                break
            end = idx + len(norm_name)
            shadowed = False
            for longer in longer_by_prefix.get(norm_name, []):
                if canon_text.startswith(longer, idx):
                    long_end = idx + len(longer)
                    if long_end == len(canon_text) or canon_text[long_end] == " ":
                        start = long_end
                        shadowed = True
                        break
            if shadowed:
                continue
            context_start = max(0, idx - logo_context_chars)
            context_end = min(len(canon_text), end + logo_context_chars)
            if context_start > 0 and canon_text[context_start - 1] != " ":
                while context_start > 0 and canon_text[context_start - 1] != " ":
                    context_start -= 1
            if context_end < len(canon_text) and canon_text[context_end] != " ":
                while context_end < len(canon_text) and canon_text[context_end] != " ":
                    context_end += 1
            context = canon_text[context_start:context_end]
            if LOGO_RE.search(context) or _CONTEXT_EE_RE.search(context):
                skipped_logo_matches += 1
            else:
                cues.append(
                    {
                        "seq": None,
                        "title": raw_name,
                        "usage": None,
                        "time": None,
                        "composers": [],
                        "publishers": [{"name": raw_name}],
                        "notes": [],
                    }
                )
                matched_publishers_count += 1
                seen_publishers.add(raw_name)
                matched = True
                break
            start = end
        if matched and log_fn:
            log_fn(f"default parser: matched_publisher={raw_name}")
    if log_fn:
        log_fn(
            "default parser: "
            f"matched_publishers={matched_publishers_count} | skipped_logo_matches={skipped_logo_matches}"
        )
        # log_fn(f"canon text: {canon_text}")
    return cues


# =================================================
# Router / entry point
# =================================================
def detect_format(
    pdf_path: Path,
) -> Literal["soundmouse", "wb", "rapidcue", "unknown", "needs_ocr"]:
    """
    "needs_ocr" is used when the first page has no extractable text.
    """
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
        if not text.strip():
            return "needs_ocr"
        if is_soundmouse_pdf(text):
            return "soundmouse"
        if is_wb_pdf(text):
            return "wb"
        if is_rapidcue_pdf(text):
            return "rapidcue"
    return "unknown"


def filter_logos(cues: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for cue in cues:
        title = cue.get("title", "") or cue.get("selection", "") or ""
        usage = cue.get("usage") or ""
        notes = cue.get("notes") or []
        if usage == "EE" or LOGO_RE.search(usage) or LOGO_RE.search(title):
            continue
        if any(LOGO_RE.search(note or "") for note in notes):
            continue
        if any(_LICENSE_NOTE_RE.search(note or "") for note in notes):
            continue
        out.append(cue)
    return out


# -----------------------------
# Normalization / tokenization
# -----------------------------

_CORP_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "LTD",
    "LIMITED",
    "LLC",
    "LLP",
    "LP",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "PLC",
    "SA",
    "SL",
}

_STOPWORDS = {
    "THE",
    "&",
    "AND",
}

_PUNCT_RE = re.compile(r"[^A-Z0-9]+")


def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def canonicalize_text_for_search(text: str) -> str:
    normalized = _strip_diacritics(text).upper()
    normalized = _PUNCT_RE.sub(" ", normalized)
    tokens = [t for t in normalized.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def canonicalize_publisher_name(name: str, *, drop_suffixes: bool = True) -> str:
    if not name:
        return ""

    s = _strip_diacritics(name).upper().strip()
    s = s.replace("&", " AND ")
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = [t for t in s.split(" ") if t and t not in _STOPWORDS]
    if drop_suffixes:
        tokens = [t for t in tokens if t not in _CORP_SUFFIXES]

    return " ".join(tokens)


def publisher_tokens(name: str) -> List[str]:
    canon = canonicalize_publisher_name(name, drop_suffixes=True)
    return canon.split(" ") if canon else []


def char_ngrams_from_canon_no_space(canon_no_space: str, n: int = 3) -> Counter:
    if not canon_no_space:
        return Counter()
    if len(canon_no_space) <= n:
        return Counter({canon_no_space: 1})
    return Counter(
        canon_no_space[i : i + n] for i in range(len(canon_no_space) - n + 1)
    )


def cosine_similarity_counts(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    dot = 0.0
    if len(a) > len(b):
        a, b = b, a
    for k, av in a.items():
        bv = b.get(k)
        if bv:
            dot += av * bv

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def damerau_levenshtein_distance_canon(a: str, b: str) -> int:
    # a and b are already canonicalized strings (with spaces)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    da = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        da[i][0] = i
    for j in range(len(b) + 1):
        da[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            da[i][j] = min(da[i - 1][j] + 1, da[i][j - 1] + 1, da[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                da[i][j] = min(da[i][j], da[i - 2][j - 2] + 1)

    return da[len(a)][len(b)]


def normalized_edit_similarity_canon(a_canon: str, b_canon: str) -> float:
    denom = max(len(a_canon), len(b_canon), 1)
    dist = damerau_levenshtein_distance_canon(a_canon, b_canon)
    return max(0.0, 1.0 - (dist / denom))


def jaccard_similarity_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_controlled_index(
    controlled_publishers: List[str],
    aliases_by_publisher: Dict[str, List[str]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Precompute canonical forms, token sets, and trigram counters for controlled publishers.
    """
    idx: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_entry(raw: str, controlled: str) -> None:
        raw = raw.strip()
        controlled = controlled.strip()
        if not raw or not controlled:
            return
        key = (raw, controlled)
        if key in seen:
            return
        seen.add(key)
        canon = canonicalize_publisher_name(raw, drop_suffixes=True)
        canon_no_space = canon.replace(" ", "")
        tok_set = set(canon.split(" ")) if canon else set()
        tri = char_ngrams_from_canon_no_space(canon_no_space, 3)
        idx.append(
            {
                "raw": raw,
                "controlled": controlled,
                "canon": canon,
                "canon_no_space": canon_no_space,
                "tokens": tok_set,
                "trigrams": tri,
            }
        )

    for pub in controlled_publishers:
        add_entry(pub, pub)
        for alias in (aliases_by_publisher or {}).get(pub, []):
            if alias and alias != pub:
                add_entry(alias, pub)
    return idx


# -----------------------------
# Best-match scoring (short-circuit friendly)
# -----------------------------
def best_controlled_match_for_cue_name(
    cue_name: str,
    controlled_index: List[Dict[str, Any]],
    *,
    remaining_only: Optional[set] = None,  # raw controlled names still not found
    review_threshold: float = 0.85,
) -> Optional[Dict[str, Any]]:
    """
    Given one cue publisher name, return the best controlled publisher match above review_threshold.
    If remaining_only is provided, restrict evaluation to controlled pubs not yet found.
    """
    cue_canon = canonicalize_publisher_name(cue_name, drop_suffixes=True)
    if not cue_canon:
        return None

    cue_tokens = set(cue_canon.split(" "))
    cue_canon_no_space = cue_canon.replace(" ", "")
    cue_trigrams = char_ngrams_from_canon_no_space(cue_canon_no_space, 3)

    best = None
    best_score = -1.0
    best_feats = None

    for c in controlled_index:
        if remaining_only is not None and c["controlled"] not in remaining_only:
            continue

        # Conservative blocking to reduce work:
        # If cue has >=2 tokens and there is zero overlap, skip.
        if (
            len(cue_tokens) >= 2
            and cue_tokens
            and c["tokens"]
            and not (cue_tokens & c["tokens"])
        ):
            continue

        j = jaccard_similarity_sets(cue_tokens, c["tokens"])
        tri = cosine_similarity_counts(cue_trigrams, c["trigrams"])
        ed = normalized_edit_similarity_canon(cue_canon, c["canon"])

        score = (0.45 * j) + (0.35 * tri) + (0.20 * ed)

        if score > best_score:
            best_score = score
            best = c
            best_feats = {"jaccard": j, "trigram_cosine": tri, "edit_sim": ed}

    if best is None or best_score < review_threshold:
        return None

    return {
        "cue_name": cue_name,
        "matched_controlled": best["controlled"],
        "matched_raw": best["raw"],
        "score": round(best_score, 6),
        "features": {k: round(v, 6) for k, v in (best_feats or {}).items()},
    }


def find_controlled_publishers_present(
    filtered_cues: List[Dict[str, Any]],
    controlled_publishers: List[str],
    aliases_by_publisher: Dict[str, List[str]] | None = None,
    *,
    auto_match_threshold: float = 0.92,
    review_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Returns a final deduped set/list of controlled publishers that appear anywhere on the cue sheet.

    Behavior:
    - Scans cue-sheet publisher names.
    - For each cue publisher name, find best controlled match.
    - If best match is >= auto_match_threshold, mark that controlled publisher as present.
    - We never "need" to match a controlled publisher more than once, so we maintain a remaining set.
    - Also returns optional evidence (best match per controlled publisher).
    """
    controlled_index = build_controlled_index(
        controlled_publishers, aliases_by_publisher
    )

    remaining = set(controlled_publishers)  # raw names
    found_best_evidence: Dict[str, Dict[str, Any]] = {}

    for cue in filtered_cues:
        for pub in cue.get("publishers", []):
            if not remaining:
                # All controlled publishers already found; full short-circuit.
                break

            cue_pub_name = pub.get("name", "")
            match = best_controlled_match_for_cue_name(
                cue_pub_name,
                controlled_index,
                remaining_only=remaining,
                review_threshold=review_threshold,
            )
            if not match:
                continue

            controlled_name = match["matched_controlled"]

            # Keep best evidence if we see multiple candidate cue strings
            prev = found_best_evidence.get(controlled_name)
            if prev is None or match["score"] > prev["score"]:
                found_best_evidence[controlled_name] = match

            # If we have high confidence, we can consider this controlled publisher "present"
            # and stop spending time matching it again.
            if match["score"] >= auto_match_threshold:
                remaining.discard(controlled_name)

        if not remaining:
            break

    found = sorted(found_best_evidence.keys())

    return {
        "found_controlled_publishers": found,
        "evidence_by_controlled": found_best_evidence,
        "unmatched_controlled_publishers": sorted(remaining),
    }


def find_controlled_publishers_present_phrase(
    filtered_cues: List[Dict[str, Any]],
    controlled_publishers: List[str],
    aliases_by_publisher: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    """
    Phrase-containment matcher for WB-like formats.
    Treats the full publisher text per cue as a blob and checks for controlled
    publisher phrases within it.
    """
    controlled_index = build_controlled_index(
        controlled_publishers, aliases_by_publisher
    )
    remaining = set(controlled_publishers)
    found_best_evidence: Dict[str, Dict[str, Any]] = {}

    def _contains_phrase(canon_text: str, phrase: str) -> bool:
        if not canon_text or not phrase:
            return False
        start = 0
        while True:
            idx = canon_text.find(phrase, start)
            if idx == -1:
                return False
            end = idx + len(phrase)
            if (idx == 0 or canon_text[idx - 1] == " ") and (
                end == len(canon_text) or canon_text[end] == " "
            ):
                return True
            start = end

    for cue in filtered_cues:
        if not remaining:
            break
        publishers = cue.get("publishers", [])
        names: list[str] = []
        for pub in publishers:
            if isinstance(pub, dict):
                name = pub.get("name", "")
            else:
                name = str(pub)
            if name:
                names.append(name)
        if not names:
            continue

        cue_blob = " ".join(names)
        cue_canon = canonicalize_publisher_name(cue_blob, drop_suffixes=True)
        if not cue_canon:
            continue

        for c in controlled_index:
            if c["controlled"] not in remaining:
                continue
            phrase = c["canon"]
            if not phrase:
                continue
            if not _contains_phrase(cue_canon, phrase):
                continue

            match = {
                "cue_name": cue_blob,
                "matched_controlled": c["controlled"],
                "matched_raw": c["raw"],
                "score": 1.0,
                "features": {"phrase": 1.0},
            }

            prev = found_best_evidence.get(c["controlled"])
            if prev is None or match["score"] > prev["score"]:
                found_best_evidence[c["controlled"]] = match
            remaining.discard(c["controlled"])

        if not remaining:
            break

    found = sorted(found_best_evidence.keys())
    return {
        "found_controlled_publishers": found,
        "evidence_by_controlled": found_best_evidence,
        "unmatched_controlled_publishers": sorted(remaining),
    }


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def normalize_text_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_unique(values: list[str], *, sort: bool) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in values:
        value = normalize_text_value(item)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if sort:
        normalized.sort()
    return normalized


def split_territories(value: str) -> list[str]:
    parts = [part for part in (value or "").split("/") if part is not None]
    return normalize_unique(parts, sort=True)


def normalize_effective_entries(rows: list[dict[str, str]]) -> list[EffectiveDateEntry]:
    by_date: dict[str, set[str]] = {}
    for row in rows:
        date_value = normalize_text_value(row.get("effective date", ""))
        if not date_value:
            continue
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError:
            continue
        territories = split_territories(str(row.get("territory", "")))
        if date_value not in by_date:
            by_date[date_value] = set()
        by_date[date_value].update(territories)

    ordered: list[EffectiveDateEntry] = []
    for date_value in sorted(by_date.keys()):
        ordered.append(
            {
                "date": date_value,
                "territories": sorted(by_date[date_value]),
            }
        )
    return ordered


def normalize_observed_name_variants(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        return []
    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def build_observed_name_variant_map(
    category_entries: list[dict[str, Any]],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if not publisher:
            continue
        variants = normalize_observed_name_variants(entry.get("observed_name_variants"))
        if not variants:
            continue
        bucket = mapping.setdefault(publisher, [])
        for variant in variants:
            if variant == publisher or variant in bucket:
                continue
            bucket.append(variant)
    return mapping


def build_xmp_metadata(
    administrator: str,
    agreements: Sequence[AgreementEntry],
) -> bytes:
    """
    Build an XMP packet containing ferp:administrator and ferp:agreements.
    Returns UTF-8 XML bytes.
    """
    admin_value = normalize_text_value(administrator)
    added_date = datetime.now(timezone.utc).date().isoformat()
    agreement_items: list[str] = []
    for agreement in agreements:
        publishers = normalize_unique(agreement.get("publishers", []), sort=False)
        if not publishers:
            continue
        publisher_items = "\n".join(
            f"                <rdf:li>{escape_xml(p)}</rdf:li>" for p in publishers
        )
        effective_entries = agreement.get("effective_dates", [])
        effective_items: list[str] = []
        for entry in effective_entries:
            date_value = normalize_text_value(entry.get("date", ""))
            if not date_value:
                continue
            territories = normalize_unique(entry.get("territories", []), sort=True)
            territory_items = "\n".join(
                f"                        <rdf:li>{escape_xml(t)}</rdf:li>"
                for t in territories
            )
            effective_items.append(
                f"""              <rdf:li rdf:parseType="Resource">
                <ferp:date>{escape_xml(date_value)}</ferp:date>
                <ferp:territories>
                  <rdf:Bag>
{territory_items}
                  </rdf:Bag>
                </ferp:territories>
              </rdf:li>"""
            )
        effective_block = ""
        if effective_items:
            effective_block = (
                "            <ferp:effectiveDates>\n"
                "              <rdf:Seq>\n"
                + "\n".join(effective_items)
                + "\n              </rdf:Seq>\n"
                "            </ferp:effectiveDates>\n"
            )
        agreement_items.append(
            f"""      <rdf:li rdf:parseType="Resource">
        <ferp:publishers>
          <rdf:Bag>
{publisher_items}
          </rdf:Bag>
        </ferp:publishers>
{effective_block}      </rdf:li>"""
        )

    agreement_block = ""
    if agreement_items:
        agreement_block = (
            "      <ferp:agreements>\n"
            "        <rdf:Bag>\n"
            + "\n".join(agreement_items)
            + "\n        </rdf:Bag>\n"
            "      </ferp:agreements>\n"
        )

    xmp = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:ferp="https://tulbox.app/ferp/xmp/1.0">
      <ferp:administrator>{escape_xml(admin_value)}</ferp:administrator>
      <ferp:dataAddedDate>{escape_xml(added_date)}</ferp:dataAddedDate>
      <ferp:stampSpecVersion>{escape_xml(STAMP_SPEC_VERSION)}</ferp:stampSpecVersion>
        {agreement_block}    
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    return xmp.encode("utf-8")


def set_xmp_metadata(
    input_pdf: Path,
    output_pdf: Path,
    xmp_bytes: bytes,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()

    for page in reader.pages:
        if check_cancel is not None:
            check_cancel()
        writer.add_page(page)

    info: dict[str, str] = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            if isinstance(k, str) and k.startswith("/") and v is not None:
                info[k] = str(v)

    if info:
        writer.add_metadata(info)

    md_stream = StreamObject()
    set_data = getattr(md_stream, "set_data", None)
    if callable(set_data):
        set_data(xmp_bytes)
    else:
        md_stream._data = xmp_bytes
        md_stream.update({NameObject("/Length"): NumberObject(len(xmp_bytes))})
    md_stream.update(
        {
            NameObject("/Type"): NameObject("/Metadata"),
            NameObject("/Subtype"): NameObject("/XML"),
        }
    )

    md_ref = writer._add_object(md_stream)
    writer._root_object.update({NameObject("/Metadata"): md_ref})

    with output_pdf.open("wb") as handle:
        writer.write(handle)


def set_xmp_metadata_inplace(
    pdf_path: Path,
    xmp_bytes: bytes,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    dir_name = os.path.dirname(pdf_path)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
    try:
        set_xmp_metadata(pdf_path, tmp_path, xmp_bytes, check_cancel=check_cancel)
        os.replace(tmp_path, pdf_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def wrap_text(text: str, font_name: str, font_size: float, max_width: float):
    """
    Greedy word-wrap based on actual font metrics.
    Returns a list of wrapped lines that each fit within max_width.
    Breaks on words only (no character-level breaking).
    """
    words = (text or "").split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        trial = current + " " + w
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def wrap_phrases(
    phrases: list[str],
    font_name: str,
    font_size: float,
    max_width: float,
    *,
    joiner: str = "and",
):
    """
    Greedy wrap that allows breaks within phrases.
    Phrases after the first are prefixed with "<joiner> ".
    """
    cleaned = [p.strip() for p in phrases if str(p or "").strip()]
    if not cleaned:
        return [""]

    tokens: list[str] = []
    for idx, phrase in enumerate(cleaned):
        words = phrase.split()
        if not words:
            continue
        if idx == 0:
            tokens.extend(words)
            continue
        tokens.append(joiner)
        tokens.extend(words)

    def concat(a: str, b: str) -> str:
        if not a:
            return b
        if not b:
            return a
        if a.endswith(" ") or b.startswith(" "):
            return a + b
        return a + " " + b

    lines = []
    current = tokens[0]
    for t in tokens[1:]:
        trial = concat(current, t)
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = t
    lines.append(current)
    return lines


def build_publisher_line_phrases(prefix: str, publishers: list[str]) -> list[str]:
    cleaned = [p.strip() for p in publishers if p.strip()]
    if not cleaned:
        return []
    first = f"{prefix} {cleaned[0]}".strip()
    return [first] + cleaned[1:]


def _clone_pdf_object(writer: PdfWriter, obj: PdfObject | None):
    if obj is None:
        return None
    clone = getattr(obj, "clone", None)
    if callable(clone):
        try:
            return clone(writer)
        except TypeError:
            return clone(writer, {})
    return obj


def _get_page_content_bytes(page: PageObject) -> bytes:
    contents = page.get_contents()
    if contents is None:
        return b""
    try:
        return contents.get_data()
    except AttributeError:
        try:
            return b"".join(c.get_data() for c in contents)
        except Exception:
            return bytes(contents)


def _add_stamp_annotation(
    writer: PdfWriter,
    page: PageObject,
    stamp_page: PageObject,
    *,
    rect_x: float,
    rect_y: float,
    rect_w: float,
    rect_h: float,
    shift_x: float,
    shift_y: float,
    stamp_name: str,
) -> None:
    resources = stamp_page.get("/Resources")
    if resources is None:
        resources_obj: PdfObject = DictionaryObject()
    else:
        try:
            resources_obj = resources.get_object()
        except AttributeError:
            resources_obj = resources
    resources_obj = cast(PdfObject, _clone_pdf_object(writer, resources_obj))

    content_bytes = _get_page_content_bytes(stamp_page)
    translate = f"q 1 0 0 1 {-shift_x:.4f} {-shift_y:.4f} cm\n".encode("ascii")
    appearance = StreamObject()
    appearance.set_data(translate + content_bytes + b"\nQ")
    appearance.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/FormType"): NumberObject(1),
            NameObject("/BBox"): ArrayObject(
                [
                    NumberObject(0),
                    NumberObject(0),
                    NumberObject(rect_w),
                    NumberObject(rect_h),
                ]
            ),
            NameObject("/Resources"): resources_obj,
        }
    )
    appearance_ref = writer._add_object(appearance)

    annot = DictionaryObject()
    annot.update(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Stamp"),
            NameObject("/Rect"): ArrayObject(
                [
                    NumberObject(rect_x),
                    NumberObject(rect_y),
                    NumberObject(rect_x + rect_w),
                    NumberObject(rect_y + rect_h),
                ]
            ),
            NameObject("/AP"): DictionaryObject({NameObject("/N"): appearance_ref}),
            NameObject("/Border"): ArrayObject(
                [NumberObject(0), NumberObject(0), NumberObject(0)]
            ),
            NameObject("/F"): NumberObject(4),
            NameObject("/T"): TextStringObject(USER_NAME),
            NameObject("/NM"): TextStringObject(stamp_name),
        }
    )
    annot_ref = writer._add_object(annot)

    annots = page.get("/Annots")
    if annots is None:
        annots_array = ArrayObject()
    else:
        try:
            annots_array = annots.get_object()
        except AttributeError:
            annots_array = annots
        if not isinstance(annots_array, ArrayObject):
            annots_array = ArrayObject([annots_array])

    cleaned = ArrayObject()
    for item in annots_array:
        try:
            item_obj = item.get_object()
        except AttributeError:
            item_obj = item
        if not isinstance(item_obj, DictionaryObject):
            cleaned.append(item)
            continue
        if item_obj.get("/Subtype") == NameObject("/Stamp"):
            continue
        cleaned.append(item)
    annots_array = cleaned

    annots_array.append(annot_ref)
    page[NameObject("/Annots")] = annots_array


def draw_top_full_badge(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    *,
    second_line_text: str,  # dynamic line immediately after the fixed first line
    second_line_phrases: list[list[str]] | None = None,  # optional grouped phrases
    deal_start_date_and_territory: list[dict[str, str]],  # row data for table
    image_path: str,  # logo image file path
    image_y_offset: float = 1.5,  # vertical adjustment for image position
    margin_right: float = 8,  # distance from page right edge to badge
    margin_top: float = 8,  # distance from page top edge to badge
    padding: float = 5,  # inner padding around image + text
    gap: float = 8,  # space between image and text area
    image_w: float = 60,  # logo width in points
    image_h: float = 33,  # logo height in points
    # Fonts
    top_font_name: str = "Helvetica-Bold",  # font for top two lines
    bottom_font_name: str = "Helvetica",  # font for table rows
    top_font_size: float = 7,  # size for top two lines
    bottom_font_size: float = 6,  # size for table rows
    # Line metrics
    top_line_height: float = 7,  # line height for top two lines
    top_line_gap: float = 4.0,  # extra vertical space between top line 1 and line 2
    bottom_line_height: float = 7.5,  # line height for table rows
    max_text_width: float = 500,  # max width for all text area right of image
    min_text_width: float = 150,  # min width for all text area right of image
    corner_radius: float = 5,  # rounded rectangle corner radius
    stroke_rgb=(0, 0, 0),  # badge outline color
    stroke_width: float = 1.0,  # badge outline width
    # Horizontal rule styling / spacing
    rule_thickness: float = 0.75,  # divider line thickness
    rule_rgb=(0, 0, 0),  # divider line color
    rule_margin_top: float = 4,  # space above divider
    rule_margin_bottom: float = 1,  # space below divider
    # Table layout (below divider)
    col_gap: float = 20,  # space between the two columns
    min_right_col_width: float = 40,  # min width for territory column
    single_row_max_right_col_width: float = 135,  # cap territory width when only one row
    header_bottom_gap: float = 0,  # space between header row and first data row
):
    """
    Draws a rounded-rectangle badge anchored to the top-right corner.
    The rounded rectangle has a fully transparent background (fill=0) and a stroke.

    Layout (right of image):
      Line 1 (fixed):  "Administration Co. admin o/b/o"    (centered)
      Line 2 (dynamic): second_line_text (centered, can wrap to multiple lines)
      Divider line
      Row 1 (headers):  "Deal Start Date"       "Controlled Territory"
      Rows (data):      <effective date>        <territory>   (territory wraps within its column)
                        <effective date>        <territory>
                        ...

    Notes:
    - This function uses a fixed text-area width governed by min_text_width/max_text_width.
    - Territory wrapping is constrained to the right column and does not widen the box.
    - Wrapping is word-based only.
    """

    fixed_line_1 = ADMINISTRATOR_NAME + " admin o/b/o"
    col_header_1 = "Deal Start Date"
    col_header_2 = "Controlled Territory"

    # --- Column widths (below divider) ---
    # Left column width must fit the header and typical date text
    left_col_w = pdfmetrics.stringWidth(
        col_header_1, bottom_font_name, bottom_font_size
    )

    # --- Measure territory text to allow a narrower box when strings are short ---
    max_territory_w = min_right_col_width
    for row in deal_start_date_and_territory or []:
        territory = str(row.get("territory", "") or "")
        if not territory:
            continue
        text_w = pdfmetrics.stringWidth(territory, bottom_font_name, bottom_font_size)
        if text_w > max_territory_w:
            max_territory_w = text_w
    if len(deal_start_date_and_territory or []) == 1:
        max_territory_w = min(max_territory_w, single_row_max_right_col_width)

    # --- Decide overall text area width (right of image) ---
    desired_text_area_w = left_col_w + col_gap + max_territory_w
    text_area_w = max(min_text_width, min(max_text_width, desired_text_area_w))

    # Ensure at least min_right_col_width remains for territory, otherwise expand the text area.
    min_total_for_cols = left_col_w + col_gap + min_right_col_width
    if text_area_w < min_total_for_cols:
        text_area_w = min_total_for_cols

    right_col_w = max(text_area_w - left_col_w - col_gap, min_right_col_width)

    # --- Wrap top block (centered) ---
    # We wrap using the full text area width (not per-column).
    if second_line_phrases:
        top_line_2_lines: list[str] = []
        for phrases in second_line_phrases:
            if not phrases:
                continue
            top_line_2_lines.extend(
                wrap_phrases(phrases, top_font_name, top_font_size, text_area_w)
            )
        if not top_line_2_lines:
            top_line_2_lines = [""]
    else:
        top_line_2_lines = wrap_text(
            second_line_text, top_font_name, top_font_size, text_area_w
        )

    # --- Prepare and wrap data rows (right column wraps; left column typically doesn't) ---
    # We also compute the total height contribution of all data rows.
    data_rows = []
    data_rows_h = 0.0

    for row in deal_start_date_and_territory or []:
        eff_date = format_effective_date(row.get("effective date", "N/A"))
        territory = row.get("territory", "N/A")

        # Left column: draw as a single line.
        eff_date_lines = [eff_date] if eff_date else [""]

        # Right column: wrap territory within right column width.
        territory_lines = wrap_text(
            territory, bottom_font_name, bottom_font_size, right_col_w
        )

        row_h = bottom_line_height * max(len(eff_date_lines), len(territory_lines), 1)
        data_rows_h += row_h

        data_rows.append(
            {
                "eff_date_lines": eff_date_lines,
                "territory_lines": territory_lines,
                "row_h": row_h,
            }
        )

    # --- Heights: top block + divider + table header + data ---
    top_block_h = (
        top_line_height  # fixed first line
        + top_line_gap  # adjustable gap
        + top_line_height * max(len(top_line_2_lines), 1)
    )

    table_header_h = bottom_line_height  # single header row

    text_stack_h = (
        top_block_h
        + rule_margin_top
        + rule_thickness
        + rule_margin_bottom
        + table_header_h
        + header_bottom_gap
        + data_rows_h
    )

    # --- Compute container size ---
    content_w = image_w + gap + text_area_w
    content_h = max(image_h, text_stack_h)

    rect_w = content_w + 2 * padding
    rect_h = content_h + 2 * padding

    r = min(corner_radius, rect_w / 2, rect_h / 2)

    rect_x = page_w - margin_right - rect_w
    rect_y = page_h - margin_top - rect_h

    c.saveState()

    # Stroke-only rounded rect (transparent background)
    c.setLineWidth(stroke_width)
    c.setStrokeColorRGB(*stroke_rgb)
    c.roundRect(rect_x, rect_y, rect_w, rect_h, r, stroke=1, fill=0)

    # Vertically center the full text stack in the available content height
    text_stack_bottom = rect_y + padding + (content_h - text_stack_h) / 2
    text_stack_top = text_stack_bottom + text_stack_h

    # Image aligned to top, matching first text line
    img_x = rect_x + padding
    img_y = text_stack_top - image_h - image_y_offset
    img = ImageReader(image_path)
    c.drawImage(img, img_x, img_y, width=image_w, height=image_h, mask="auto")

    # Text origin (to the right of the image)
    text_x = img_x + image_w + gap

    # Baseline offsets for each font size / line height
    top_baseline_offset = (top_line_height - top_font_size) * 0.8
    bottom_baseline_offset = (bottom_line_height - bottom_font_size) * 0.8

    c.setFillColorRGB(0, 0, 0)
    cursor_y = text_stack_top

    # ---- Top block (centered) ----
    c.setFont(top_font_name, top_font_size)

    # Helper: centered draw within text area width
    def draw_centered(y: float, s: str):
        w = pdfmetrics.stringWidth(s, top_font_name, top_font_size)
        x = text_x + (text_area_w - w) / 2
        c.drawString(x, y, s)

    # Line 1 (fixed, centered)
    cursor_y -= top_line_height
    draw_centered(cursor_y + top_baseline_offset, fixed_line_1)

    # Gap between line 1 and line 2
    cursor_y -= top_line_gap

    # Line 2 (dynamic, wrapped, centered line-by-line)
    for line in top_line_2_lines:
        cursor_y -= top_line_height
        draw_centered(cursor_y + top_baseline_offset, line)

    # ---- Divider ----
    cursor_y -= rule_margin_top
    c.saveState()
    c.setLineWidth(rule_thickness)
    c.setStrokeColorRGB(*rule_rgb)
    c.line(text_x, cursor_y, text_x + text_area_w, cursor_y)
    c.restoreState()
    cursor_y -= rule_thickness + rule_margin_bottom

    # ---- Table header (two columns) ----
    c.setFont(bottom_font_name, bottom_font_size)
    left_x = text_x
    right_x = text_x + left_col_w + col_gap

    cursor_y -= bottom_line_height
    c.drawString(left_x, cursor_y + bottom_baseline_offset, col_header_1)
    c.drawString(right_x, cursor_y + bottom_baseline_offset, col_header_2)

    cursor_y -= header_bottom_gap

    # ---- Data rows (variable height; right column wraps) ----
    for row in data_rows:
        eff_lines = row["eff_date_lines"]
        terr_lines = row["territory_lines"]
        max_lines = max(len(eff_lines), len(terr_lines), 1)

        # Draw each visual line of the row
        for i in range(max_lines):
            cursor_y -= bottom_line_height
            if i < len(eff_lines):
                c.drawString(left_x, cursor_y + bottom_baseline_offset, eff_lines[i])
            if i < len(terr_lines):
                c.drawString(right_x, cursor_y + bottom_baseline_offset, terr_lines[i])

    c.restoreState()

    return {
        "rect": (rect_x, rect_y, rect_w, rect_h),
        "image": (img_x, img_y, image_w, image_h),
        "text_area": (text_x, text_stack_bottom, text_area_w, text_stack_h),
        "columns": {"left_w": left_col_w, "right_w": right_col_w, "gap": col_gap},
        "top_block": {"line1": fixed_line_1, "line2_lines": top_line_2_lines},
        "row_count": len(data_rows),
        "stroke_width": stroke_width,
    }


def add_stamp(
    pdf_path: Path,
    output_path: Path,
    matched_publishers: list[str],
    multi_territory_rows: list[dict[str, str]],
    second_line_phrases: list[list[str]] | None = None,
) -> None:
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.jpg"

    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        return

    writer = PdfWriter()
    first_page = reader.pages[0]
    rotation = int(first_page.get("/Rotate", 0) or 0) % 360
    stamp_page_target = (
        normalize_page_rotation(first_page, writer) if rotation else first_page
    )
    page_w, page_h, offset_x, offset_y = get_page_box(stamp_page_target)

    temp_handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    c = canvas.Canvas(str(temp_path), pagesize=(page_w, page_h))

    badge_info = draw_top_full_badge(
        c,
        page_w,
        page_h,
        second_line_text=" and ".join(matched_publishers),
        second_line_phrases=second_line_phrases,
        deal_start_date_and_territory=multi_territory_rows,
        image_path=str(logo_path),
        top_font_name="Helvetica-Bold",
        bottom_font_name="Helvetica",
    )

    c.showPage()
    c.save()

    stamp_reader = PdfReader(str(temp_path))
    if not stamp_reader.pages:
        return
    stamp_page = stamp_reader.pages[0]

    rect_x, rect_y, rect_w, rect_h = badge_info["rect"]
    stroke_width = float(badge_info.get("stroke_width", 0.0) or 0.0)
    inset = stroke_width / 2.0
    extra_top = max(0.0, stroke_width / 2.0) + 0.25
    extra_sides = max(0.0, stroke_width / 2.0)
    cushion = 2.0
    rect_x_page = rect_x + offset_x - inset - extra_sides - cushion
    rect_y_page = rect_y + offset_y - inset - cushion
    rect_w += inset * 2 + extra_sides * 2 + cushion * 2
    rect_h += inset * 2 + extra_top + cushion * 2

    for index, page in enumerate(reader.pages):
        if index == 0:
            if rotation:
                page = stamp_page_target
            _add_stamp_annotation(
                writer,
                page,
                stamp_page,
                rect_x=rect_x_page,
                rect_y=rect_y_page,
                rect_w=rect_w,
                rect_h=rect_h,
                shift_x=rect_x - inset - extra_sides - cushion,
                shift_y=rect_y - inset - cushion,
                stamp_name="ferp:umpg-stamp",
            )
        writer.add_page(page)

    if reader.metadata:
        info: dict[str, str] = {}
        for k, v in reader.metadata.items():
            if isinstance(k, str) and k.startswith("/") and v is not None:
                info[k] = str(v)
        if info:
            writer.add_metadata(info)

    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    if temp_path.exists():
        temp_path.unlink()


def get_page_box(page) -> tuple[float, float, float, float]:
    box = getattr(page, "cropbox", None) or page.mediabox
    return (
        float(box.width),
        float(box.height),
        float(box.lower_left[0]),
        float(box.lower_left[1]),
    )


def normalize_page_rotation(page: PageObject, writer: PdfWriter) -> PageObject:
    rotation = int(page.get("/Rotate", 0) or 0) % 360
    if rotation == 0:
        return page

    box = page.mediabox
    width = float(box.width)
    height = float(box.height)
    if rotation in {90, 270}:
        new_width, new_height = height, width
    else:
        new_width, new_height = width, height

    dst_page = PageObject.create_blank_page(
        pdf=writer, width=new_width, height=new_height
    )
    annots = page.get("/Annots")
    if annots is not None:
        dst_page[NameObject("/Annots")] = annots

    if rotation == 90:
        transform = Transformation().rotate(-90).translate(0, width)
    elif rotation == 180:
        transform = Transformation().rotate(-180).translate(width, height)
    elif rotation == 270:
        transform = Transformation().rotate(-270).translate(height, 0)
    else:
        transform = Transformation()

    merge_transformed = getattr(dst_page, "merge_transformed_page", None)
    if callable(merge_transformed):
        merge_transformed(page, transform)
    else:
        add_transformation = getattr(page, "add_transformation", None)
        if callable(add_transformation):
            add_transformation(transform)
        dst_page.merge_page(page)

    return dst_page


def format_effective_date(date_text: str) -> str:
    if not date_text:
        return "N/A"
    try:
        dt = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return "N/A"
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def parse_catalog_codes(raw_value: str) -> list[str]:
    raw_parts = (raw_value or "").split(",")
    seen: set[str] = set()
    codes: list[str] = []
    for part in raw_parts:
        code = part.strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def format_rows_for_compare(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    formatted: list[tuple[str, str]] = []
    for row in rows:
        eff_date = format_effective_date(row.get("effective date", ""))
        territory = str(row.get("territory", "")).strip()
        formatted.append((eff_date, territory))
    return formatted


def sort_stamp_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def _sort_key(item: dict[str, str]) -> tuple[int, datetime | None, str]:
        date_text = str(item.get("effective date", "")).strip()
        try:
            parsed = datetime.strptime(date_text, "%Y-%m-%d") if date_text else None
        except ValueError:
            parsed = None
        has_date = 0 if parsed else 1
        territory = str(item.get("territory", "")).strip()
        return (has_date, parsed, territory)

    return sorted(rows, key=_sort_key)


def _earliest_date_value(date_values: list[str]) -> str:
    earliest: datetime | None = None
    earliest_value = ""
    for value in date_values:
        if not value:
            continue
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            parsed = None
        if parsed and (earliest is None or parsed < earliest):
            earliest = parsed
            earliest_value = value
        elif not earliest and not earliest_value:
            earliest_value = value
    return earliest_value


def reduce_rows_by_territory(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_territory: dict[str, list[str]] = {}
    for row in rows:
        territory = str(row.get("territory", "")).strip()
        date_value = str(row.get("effective date", "")).strip()
        if not territory and not date_value:
            continue
        by_territory.setdefault(territory, []).append(date_value)
    reduced: list[dict[str, str]] = []
    for territory, date_values in by_territory.items():
        reduced.append(
            {
                "effective date": _earliest_date_value(date_values),
                "territory": territory,
            }
        )
    return reduced


def resolve_publisher_fields_raw(
    matched_publishers: list[str],
    category_entries: list[dict],
) -> tuple[str, str]:
    territory = ""
    effective_date = ""
    earliest_date: datetime | None = None
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher in matched_set:
            territory_value = str(entry.get("territory", "")).strip()
            if territory_value and not territory:
                territory = territory_value
            date_value = str(entry.get("effective date", "")).strip()
            if date_value:
                try:
                    parsed = datetime.strptime(date_value, "%Y-%m-%d")
                except ValueError:
                    parsed = None
                if parsed and (earliest_date is None or parsed < earliest_date):
                    earliest_date = parsed
                    effective_date = date_value
                elif not earliest_date and not effective_date:
                    effective_date = date_value
    return effective_date, territory


def scale_from_top_space(top_space_pts: float) -> float:
    if top_space_pts <= 50:
        return 0.95
    if top_space_pts <= 100:
        return 0.9
    if top_space_pts <= 150:
        return 0.8
    if top_space_pts <= 200:
        return 0.78
    return 1.0


def make_top_space_first_page_inplace(
    pdf_path: Path,
    top_space_pts: float,
    *,
    scale: float | None = None,
) -> None:
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    scale_value = scale if scale is not None else scale_from_top_space(top_space_pts)

    for i, src_page in enumerate(reader.pages):
        if i != 0:
            writer.add_page(src_page)
            continue

        # Use MediaBox for physical page size
        box = src_page.mediabox
        width = float(box.width)
        height = float(box.height)

        top_space = float(top_space_pts)
        dx = (1.0 - scale_value) * width / 2.0
        dy = (1.0 - scale_value) * height - top_space

        dst_page = PageObject.create_blank_page(pdf=writer, width=width, height=height)
        if src_page.cropbox:
            dst_page.cropbox = src_page.cropbox
        if getattr(src_page, "bleedbox", None):
            dst_page.bleedbox = src_page.bleedbox
        if getattr(src_page, "trimbox", None):
            dst_page.trimbox = src_page.trimbox
        if getattr(src_page, "artbox", None):
            dst_page.artbox = src_page.artbox
        transform = Transformation().scale(scale_value, scale_value).translate(dx, dy)
        merge_transformed = getattr(dst_page, "merge_transformed_page", None)
        if callable(merge_transformed):
            merge_transformed(src_page, transform)
        else:
            add_transformation = getattr(src_page, "add_transformation", None)
            if callable(add_transformation):
                add_transformation(transform)
            dst_page.merge_page(src_page)
        writer.add_page(dst_page)

    # ---- Copy Document Info metadata ----
    if reader.metadata:
        md: dict[str, str] = {}
        for k, v in reader.metadata.items():
            if isinstance(k, str) and k.startswith("/") and v is not None:
                md[str(k)] = str(v)
        if md:
            writer.add_metadata(md)

    # ---- Safe in-place overwrite ----
    dir_name = os.path.dirname(pdf_path)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "wb") as handle:
            writer.write(handle)
        os.replace(tmp_path, pdf_path)  # atomic on most platforms
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def resolve_multi_territory_rows(
    matched_publishers: list[str],
    category_entries: list[dict],
    *,
    territory_key: str = "multi_territory",
    selected_codes: set[str] | None = None,
) -> list[dict[str, str]]:
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher not in matched_set:
            continue
        territory_rows = entry.get(territory_key) or []
        if not isinstance(territory_rows, list):
            continue
        for row in territory_rows:
            if not isinstance(row, dict):
                continue
            if selected_codes is not None:
                code = str(row.get("territory_code", "")).strip()
                if not code or code not in selected_codes:
                    continue
            effective_date = str(row.get("effective date", "")).strip()
            territory = str(row.get("territory", "")).strip()
            if not (effective_date or territory):
                continue
            key = (effective_date, territory)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "effective date": effective_date,
                    "territory": territory,
                    "status": str(row.get("status", "")).strip(),
                }
            )

    def _sort_key(item: dict[str, str]) -> tuple[int, datetime | None, str, str]:
        date_text = item.get("effective date", "")
        try:
            parsed = datetime.strptime(date_text, "%Y-%m-%d") if date_text else None
        except ValueError:
            parsed = None
        has_date = 0 if parsed else 1
        return (has_date, parsed, item.get("territory", ""), item.get("status", ""))

    rows.sort(key=_sort_key)
    return rows


def normalize_territory_mode(value: str) -> str:
    normalized = (value or "").strip()
    if normalized.lower() == "multiple":
        return "multiple"
    if normalized.lower() == "split":
        return "split"
    return "single"


def resolve_territory_mode(
    matched_publishers: list[str],
    category_entries: list[dict],
) -> str:
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    modes: set[str] = set()
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher not in matched_set:
            continue
        mode = normalize_territory_mode(str(entry.get("territory", "")))
        modes.add(mode)
    if not modes:
        return "single"
    if len(modes) > 1:
        return "mixed"
    return next(iter(modes))


def collect_split_territory_options(
    matched_publishers: list[str],
    category_entries: list[dict],
    *,
    max_territory_len: int = 90,
) -> list[str]:
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    options: list[str] = []
    seen: set[str] = set()
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher not in matched_set:
            continue
        split_rows = entry.get("split_territory") or []
        if not isinstance(split_rows, list):
            continue
        for row in split_rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("territory_code", "")).strip()
            if not code or code in seen:
                continue
            territory = normalize_text_value(str(row.get("territory", "") or ""))
            label = code
            if territory:
                snippet = territory[:max_territory_len].rstrip()
                if len(territory) > max_territory_len:
                    snippet = f"{snippet}..."
                label = f"{code} - {snippet}"
            seen.add(code)
            options.append(label)
    return options


def split_territory_code_label(value: str) -> str:
    return value.split(" - ", 1)[0].strip()


def build_agreements(
    matched_publishers: list[str],
    category_entries: list[dict],
    multi_territory_rows: list[dict[str, str]],
) -> list[AgreementEntry]:
    if not matched_publishers:
        return []
    effective_rows = multi_territory_rows
    if not effective_rows:
        raw_date, raw_territory = resolve_publisher_fields_raw(
            matched_publishers, category_entries
        )
        if raw_date or raw_territory:
            effective_rows = [{"effective date": raw_date, "territory": raw_territory}]
    effective_entries = normalize_effective_entries(effective_rows)
    return [
        {
            "publishers": matched_publishers,
            "effective_dates": effective_entries,
        }
    ]


def build_agreements_for_groups(
    publisher_groups: list[tuple[list[str], list[dict[str, str]]]],
    category_entries: list[dict],
) -> list[AgreementEntry]:
    agreements: list[AgreementEntry] = []
    for publishers, group_rows in publisher_groups:
        if not publishers:
            continue
        effective_rows = group_rows
        if not effective_rows:
            raw_date, raw_territory = resolve_publisher_fields_raw(
                publishers, category_entries
            )
            if raw_date or raw_territory:
                effective_rows = [
                    {"effective date": raw_date, "territory": raw_territory}
                ]
        effective_entries = normalize_effective_entries(effective_rows)
        agreements.append(
            {
                "publishers": publishers,
                "effective_dates": effective_entries,
            }
        )
    return agreements


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    env_paths = ctx.environment.get("paths", {})
    cache_dir = Path(env_paths.get("cache_dir"))
    cache_path = cache_dir / "publishers_cache.json"
    if not cache_path.exists():
        api.emit_result(
            {
                "_status": "error",
                "_title": "Error: Missing Cache File",
                "Info": "Pull your Monday board data to create a cache before running this script.",
            }
        )
        return

    target_dir = ctx.target_path
    global USER_NAME
    try:
        USER_NAME = getpass.getuser()
    except Exception:
        USER_NAME = "FERP"

    with cache_path.open("r", encoding="utf-8") as handle:
        pubs = json.load(handle)
    pub_keys = [key for key in pubs.keys() if not str(key).startswith("__")]
    pub_key_set = set(pub_keys)
    meta = pubs.get("__meta__", {}) if isinstance(pubs, dict) else {}
    board_description = ""
    if isinstance(meta, dict):
        board_description = str(meta.get("board_description") or "")
    description_map = parse_board_description_yaml(board_description)
    admin_text = description_map.get("admin-text") or ""
    stamp_spec = description_map.get("stamping-spec") or ""
    global ADMINISTRATOR_NAME, STAMP_SPEC_VERSION
    if admin_text:
        ADMINISTRATOR_NAME = admin_text
    if stamp_spec:
        STAMP_SPEC_VERSION = stamp_spec

    payload = api.request_input_json(
        "Publisher catalog code(s) (comma separated, e.g., 'uvs, amz')",
        id="ferp_process_cue_sheets_cat_code",
        suggestions=[pub for pub in pub_keys],
        fields=[
            {
                "id": "recursive",
                "type": "bool",
                "label": "Recursive",
                "default": False,
            },
            {
                "id": "in_place",
                "type": "bool",
                "label": "Overwrite files",
                "default": False,
            },
            {
                "id": "adjust_header",
                "type": "bool",
                "label": "Add header",
                "default": False,
            },
            {
                "id": "manual_select",
                "type": "bool",
                "label": "Select publishers",
                "default": False,
            },
            {
                "id": "custom_stamp",
                "type": "bool",
                "label": "Custom stamp",
                "default": False,
            },
        ],
        payload_type=UserResponse,
    )

    custom_stamp = payload["custom_stamp"]
    codes = parse_catalog_codes(payload["value"])
    if not codes and not custom_stamp:
        api.emit_result(
            {
                "_status": "error",
                "_title": "Error: Missing Catalog Codes",
                "Info": "Enter at least one catalog code.",
            }
        )
        return
    main_con_pubs: List[str] = []
    copub_con_pubs: List[str] = []
    if not custom_stamp:
        invalid_codes = [code for code in codes if code not in pub_key_set]
        if invalid_codes:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Invalid Catalog Codes",
                    "Info": f"Invalid catalog codes: {', '.join(invalid_codes)}",
                }
            )
            return

        main_code = codes[0]
        copub_codes = codes[1:]
        cat_objects: list[list[dict]] = [pubs[main_code]] + [
            pubs[c] for c in copub_codes
        ]
        cat_object: List[dict] = [entry for group in cat_objects for entry in group]

        main_con_pubs = [entry["publisher"] for entry in pubs[main_code]]
        copub_con_pubs = [
            entry["publisher"] for code in copub_codes for entry in pubs[code]
        ]
        con_pubs: List[str] = [*main_con_pubs, *copub_con_pubs]

        codes_label = ", ".join(code.upper() for code in codes)
    else:
        copub_codes: list[str] = []
        cat_object = []
        con_pubs = []
        codes_label = ""

    observed_name_variants = build_observed_name_variant_map(cat_object)
    con_pubs_with_aliases = con_pubs
    if observed_name_variants:
        con_pubs_with_aliases = list(con_pubs)
        for variants in observed_name_variants.values():
            for variant in variants:
                if variant not in con_pubs_with_aliases:
                    con_pubs_with_aliases.append(variant)

    recursive = payload["recursive"]
    in_place = payload["in_place"]
    adjust_header = payload["adjust_header"]
    manual_select = payload["manual_select"]

    if custom_stamp:
        main_pub_set: set[str] = set()
        copub_pub_set: set[str] = set()
    else:
        main_pub_set = {p.strip() for p in main_con_pubs if p.strip()}
        copub_pub_set = {p.strip() for p in copub_con_pubs if p.strip()}

    selected_pubs: list[str] = []
    if manual_select and con_pubs and not custom_stamp:
        selection_payload = api.request_input_json(
            "Select controlled publishers",
            id="ferp_process_cue_sheets_selected_pubs",
            fields=[
                {
                    "id": "selected_pubs",
                    "type": "multi_select",
                    "label": f"Publisher catalog codes: {codes_label}",
                    "options": con_pubs,
                    "default": [],
                }
            ],
            show_text_input=False,
            payload_type=PublisherSelectionResponse,
        )
        selected_pubs = selection_payload.get("selected_pubs", [])
        if selected_pubs:
            con_pubs = selected_pubs

    header_top_space = 50.0
    if adjust_header:
        header_response = api.request_input(
            "PDF Header Adjustment: Top Space Amount (points)",
            id="ferp_process_cue_sheets_header_space",
            default=str(header_top_space),
        )
        header_response = header_response.strip() if header_response else ""
        if header_response:
            try:
                header_top_space = float(header_response)
            except ValueError:
                api.emit_result(
                    {
                        "_status": "error",
                        "_title": "Error: Invalid Input",
                        "Info": "Top space must be a number.",
                    }
                )
                return
        if header_top_space <= 0 or header_top_space > 200:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Invalid Input",
                    "Info": "Top space must be greater than 0 and less than or equal to 200.",
                }
            )
            return

    custom_publishers = ""
    custom_territory = ""
    custom_effective_date = ""
    if custom_stamp:
        publishers_response = api.request_input(
            "Enter exact publishers string",
            id="ferp_process_cue_sheets_custom_publishers",
        )
        custom_publishers = publishers_response.strip() if publishers_response else ""
        if not custom_publishers:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Missing Publishers",
                    "Info": "Publishers string is required for custom stamp.",
                }
            )
            return

        territory_response = api.request_input(
            "Enter the exact territory string",
            id="ferp_process_cue_sheets_custom_territory",
        )
        custom_territory = territory_response.strip() if territory_response else ""
        if not custom_territory:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Missing Territory",
                    "Info": "Territory string is required for custom stamp.",
                }
            )
            return

        date_response = api.request_input(
            "Enter the effective date (D-M-YYYY)",
            id="ferp_process_cue_sheets_custom_effective_date",
        )
        date_response = date_response.strip() if date_response else ""
        if not date_response:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Missing Effective Date",
                    "Info": "Effective date is required for custom stamp.",
                }
            )
            return
        try:
            parsed_date = datetime.strptime(date_response, "%d-%m-%Y")
        except ValueError:
            api.emit_result(
                {
                    "_status": "error",
                    "_title": "Error: Invalid Effective Date",
                    "Info": "Enter the effective date in D-M-YYYY format.",
                }
            )
            return
        custom_effective_date = parsed_date.date().isoformat()

    raw_cues = []
    temp_paths: list[Path] = []
    default_split_selection: list[str] | None = None

    def _cleanup() -> None:
        for path in temp_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    api.register_cleanup(_cleanup)

    def _resolve_group_rows(
        publishers: list[str],
        territory_mode: str,
        *,
        selected_codes: set[str] | None,
    ) -> list[dict[str, str]]:
        if not publishers:
            return []
        if territory_mode in {"multiple", "split"}:
            territory_key = (
                "split_territory" if territory_mode == "split" else "multi_territory"
            )
            rows = resolve_multi_territory_rows(
                publishers,
                cat_object,
                territory_key=territory_key,
                selected_codes=selected_codes,
            )
        else:
            rows = []
        if not rows:
            raw_date, raw_territory = resolve_publisher_fields_raw(
                publishers,
                cat_object,
            )
            if raw_date or raw_territory:
                rows = [{"effective date": raw_date, "territory": raw_territory}]
        return rows

    api.check_cancel()
    pdf_files = collect_files(
        target_dir,
        "*.pdf",
        recursive,
        check_cancel=api.check_cancel,
    )
    total_files = len(pdf_files)
    created_dirs: set[str] = set()
    api.log(
        "info",
        f"Cue sheet PDFs found={total_files} | category_codes={', '.join(codes)} | recursive={recursive}",
    )

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        api.progress(
            current=index,
            total=total_files,
            unit="files",
            message=f"Processing Cue Sheets in '{pdf_path.parent.name}'",
        )
        if custom_stamp:
            fmt = "custom"
            raw_cues = []
            filtered_cues = []
            matched_publishers = [custom_publishers]
            accuracy_audits = {}

            work_path = pdf_path
            if adjust_header:
                temp_path = pdf_path.with_suffix(".header.tmp.pdf")
                temp_paths.append(temp_path)
                shutil.copyfile(pdf_path, temp_path)
                make_top_space_first_page_inplace(temp_path, header_top_space)
                work_path = temp_path

            stamp_rows = [
                {
                    "effective date": custom_effective_date,
                    "territory": custom_territory,
                }
            ]
            agreements: list[AgreementEntry] = [
                {
                    "publishers": [custom_publishers],
                    "effective_dates": [
                        {
                            "date": custom_effective_date,
                            "territories": [custom_territory],
                        }
                    ],
                }
            ]
            xmp_bytes = build_xmp_metadata(ADMINISTRATOR_NAME, agreements)
            if in_place:
                add_stamp(
                    work_path,
                    pdf_path,
                    matched_publishers,
                    stamp_rows,
                )
                set_xmp_metadata_inplace(
                    pdf_path,
                    xmp_bytes,
                    check_cancel=api.check_cancel,
                )
            else:
                out_dir = pdf_path.parent / "_stamped"
                out_path = out_dir / pdf_path.name
                add_stamp(
                    work_path,
                    out_path,
                    matched_publishers,
                    stamp_rows,
                )
                set_xmp_metadata_inplace(
                    out_path,
                    xmp_bytes,
                    check_cancel=api.check_cancel,
                )
                created_dirs.add("_stamped")
            api.log(
                "info",
                f"Processed '{pdf_path.relative_to(target_dir)}' | format={fmt} | total_cues={len(raw_cues)} | filtered_cues={len(filtered_cues)} | matched_controlled_publishers={', '.join(matched_publishers)} | accuracy_audits={str(accuracy_audits)}",
            )
            continue
        if selected_pubs:
            fmt = "manual"
            raw_cues = []
            filtered_cues = []
            matched_publishers = selected_pubs
            accuracy_audits = {}
        else:
            fmt = detect_format(pdf_path)
            if fmt == "soundmouse":
                api.log("debug", f"{pdf_path.name}: detected Soundmouse format")
                raw_cues = parse_soundmouse(
                    pdf_path,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )
            elif fmt == "wb":
                api.log("debug", f"{pdf_path.name}: detected WB format")
                raw_cues = parse_wb(
                    pdf_path,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )
            elif fmt == "rapidcue":
                api.log("debug", f"{pdf_path.name}: detected RapidCue format")
                raw_cues = parse_rapidcue(
                    pdf_path,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )
            elif fmt == "needs_ocr":
                ocr_dir = pdf_path.parent / "_needs_ocr"
                ocr_dir.mkdir(exist_ok=True)
                pdf_path.replace(ocr_dir / pdf_path.name)
                created_dirs.add("_needs_ocr")
                api.log(
                    "warn",
                    f"{pdf_path.name}: no extractable text; moved to _needs_ocr",
                )
                continue
            else:
                api.log(
                    "debug", f"{pdf_path.name}: unknown format; using default parser"
                )
                raw_cues = parse_default(
                    pdf_path,
                    con_pubs_with_aliases,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )

            api.check_cancel()
            if fmt == "unknown":
                filtered_cues = raw_cues
            else:
                filtered_cues = filter_logos(raw_cues)
                skipped_logos = len(raw_cues) - len(filtered_cues)
                if skipped_logos:
                    api.log(
                        "debug",
                        f"{pdf_path.name}: skipped_logo_cues={skipped_logos}",
                    )
                # api.log("debug", f"filtered cues:{filtered_cues}")
            if fmt == "wb":
                result = find_controlled_publishers_present_phrase(
                    filtered_cues,
                    con_pubs,
                    aliases_by_publisher=observed_name_variants,
                )
            else:
                result = find_controlled_publishers_present(
                    filtered_cues,
                    con_pubs,
                    aliases_by_publisher=observed_name_variants,
                    auto_match_threshold=0.92,
                    review_threshold=0.85,
                )
            matched_publishers = result.get("found_controlled_publishers", [])
            accuracy_audits = result.get("evidence_by_controlled", {})
        matched_main = [p for p in matched_publishers if p in main_pub_set]
        matched_copub = [p for p in matched_publishers if p in copub_pub_set]
        has_both_groups = bool(matched_main) and bool(matched_copub)
        second_line_phrases = None
        if copub_codes and has_both_groups:
            grouped_phrases: list[list[str]] = []
            main_phrases = build_publisher_line_phrases("(1)", matched_main)
            if main_phrases:
                grouped_phrases.append(main_phrases)
            copub_phrases = build_publisher_line_phrases("(2)", matched_copub)
            if copub_phrases:
                grouped_phrases.append(copub_phrases)
            if grouped_phrases:
                second_line_phrases = grouped_phrases
        if matched_publishers:
            main_territory_mode = resolve_territory_mode(matched_main, cat_object)
            copub_territory_mode = resolve_territory_mode(matched_copub, cat_object)
            if main_territory_mode == "mixed" or copub_territory_mode == "mixed":
                err_dir = pdf_path.parent / "_error"
                err_dir.mkdir(exist_ok=True)
                pdf_path.replace(err_dir / pdf_path.name)
                created_dirs.add("_error")
                api.log(
                    "error",
                    f"{pdf_path.name}: mixed territory modes within a publisher group; moved to _error",
                )
                continue

            selected_codes: set[str] | None = None
            needs_split = (
                main_territory_mode == "split" or copub_territory_mode == "split"
            )
            if needs_split:
                split_options = collect_split_territory_options(
                    matched_publishers, cat_object
                )
                available_codes = {
                    split_territory_code_label(value) for value in split_options
                }
                if not split_options:
                    api.emit_result(
                        {
                            "_status": "error",
                            "_title": "Error: Missing Split Territories",
                            "Info": (
                                "Split territory rows are missing for one or more matched publishers; "
                                "cannot prompt for territory selection."
                            ),
                        }
                    )
                    return
                if default_split_selection is not None:
                    if not set(default_split_selection).issubset(available_codes):
                        err_dir = pdf_path.parent / "_error"
                        err_dir.mkdir(exist_ok=True)
                        pdf_path.replace(err_dir / pdf_path.name)
                        created_dirs.add("_error")
                        api.log(
                            "error",
                            f"{pdf_path.name}: default split selection not valid for this file; moved to _error",
                        )
                        continue
                    selected_codes = set(default_split_selection)
                else:
                    while True:
                        selection_payload = api.request_input_json(
                            "Select split territory codes",
                            id="ferp_process_cue_sheets_split_codes",
                            fields=[
                                {
                                    "id": "territory_codes",
                                    "type": "multi_select",
                                    "label": f"Publisher catalog codes: {codes_label}\n\nFile: {pdf_path.stem}\n",
                                    "options": split_options,
                                    "default": [],
                                },
                                {
                                    "id": "use_default_split_selection",
                                    "type": "bool",
                                    "label": "Use this selection for remaining files",
                                    "default": False,
                                },
                            ],
                            show_text_input=False,
                            payload_type=SplitTerritorySelectionResponse,
                        )
                        selected = selection_payload.get("territory_codes", [])
                        if selected:
                            if selection_payload.get("use_default_split_selection"):
                                default_split_selection = [
                                    split_territory_code_label(value)
                                    for value in selected
                                ]
                            selected_codes = {
                                split_territory_code_label(value) for value in selected
                            }
                            break
                        api.log(
                            "warn",
                            "Select at least one territory code to continue.",
                        )

            work_path = pdf_path
            if adjust_header:
                temp_path = pdf_path.with_suffix(".header.tmp.pdf")
                temp_paths.append(temp_path)
                shutil.copyfile(pdf_path, temp_path)
                make_top_space_first_page_inplace(temp_path, header_top_space)
                work_path = temp_path
            api.check_cancel()
            main_rows = _resolve_group_rows(
                matched_main,
                main_territory_mode,
                selected_codes=selected_codes,
            )
            copub_rows = _resolve_group_rows(
                matched_copub,
                copub_territory_mode,
                selected_codes=selected_codes,
            )
            if needs_split and selected_codes and not (main_rows or copub_rows):
                err_dir = pdf_path.parent / "_error"
                err_dir.mkdir(exist_ok=True)
                pdf_path.replace(err_dir / pdf_path.name)
                created_dirs.add("_error")
                api.log(
                    "error",
                    f"{pdf_path.name}: split selection produced no territory rows; moved to _error",
                )
                continue

            stamp_rows: list[dict[str, str]] = []
            if copub_codes and has_both_groups:
                main_reduced = reduce_rows_by_territory(main_rows)
                copub_reduced = reduce_rows_by_territory(copub_rows)
                for row in main_reduced:
                    territory = str(row.get("territory", "")).strip()
                    date_value = str(row.get("effective date", "")).strip()
                    stamp_rows.append(
                        {
                            "effective date": date_value,
                            "territory": f"(1) {territory}".strip(),
                        }
                    )
                for row in copub_reduced:
                    territory = str(row.get("territory", "")).strip()
                    date_value = str(row.get("effective date", "")).strip()
                    stamp_rows.append(
                        {
                            "effective date": date_value,
                            "territory": f"(2) {territory}".strip(),
                        }
                    )
            else:
                stamp_rows = main_rows

            # Ensure the stamp displays N/A when date/territory are missing.
            if not stamp_rows:
                stamp_rows = [{"effective date": "", "territory": ""}]
            else:
                stamp_rows = sort_stamp_rows(stamp_rows)

            agreements = build_agreements_for_groups(
                [
                    (matched_main, main_rows),
                    (matched_copub, copub_rows),
                ],
                cat_object,
            )
            xmp_bytes = build_xmp_metadata(ADMINISTRATOR_NAME, agreements)
            if in_place:
                add_stamp(
                    work_path,
                    pdf_path,
                    matched_publishers,
                    stamp_rows,
                    second_line_phrases=second_line_phrases,
                )
                set_xmp_metadata_inplace(
                    pdf_path,
                    xmp_bytes,
                    check_cancel=api.check_cancel,
                )
            else:
                out_dir = pdf_path.parent / "_stamped"
                out_path = out_dir / pdf_path.name
                add_stamp(
                    work_path,
                    out_path,
                    matched_publishers,
                    stamp_rows,
                    second_line_phrases=second_line_phrases,
                )
                set_xmp_metadata_inplace(
                    out_path,
                    xmp_bytes,
                    check_cancel=api.check_cancel,
                )
                created_dirs.add("_stamped")
        else:
            api.check_cancel()
            nop_dir = pdf_path.parent / "_nop"
            nop_dir.mkdir(exist_ok=True)
            pdf_path.replace(nop_dir / pdf_path.name)
            created_dirs.add("_nop")

        api.log(
            "info",
            f"Processed '{pdf_path.relative_to(target_dir)}' | format={fmt} | total_cues={len(raw_cues)} | filtered_cues={len(filtered_cues)} | matched_controlled_publishers={', '.join(matched_publishers)} | accuracy_audits={str(accuracy_audits)}",
        )

    api.emit_result(
        {
            "_title": "Cue Sheet Processing Finished",
            "Created Directories": ", ".join(sorted(created_dirs)),
            "Total Files": total_files,
            "Category Codes": ", ".join(codes),
        }
    )


if __name__ == "__main__":
    main()
