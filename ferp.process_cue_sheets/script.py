from __future__ import annotations

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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import pdfplumber
from pdfplumber.page import Page
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from pypdf.errors import PdfReadWarning
from pypdf.generic import NameObject, NumberObject, StreamObject

# from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from typing_extensions import Literal

from ferp.fscp.scripts import sdk

warnings.filterwarnings("ignore", category=PdfReadWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

LOGO_RE = re.compile(r"(?<![A-Z0-9])logos?(?![A-Z0-9])", re.IGNORECASE)
_CONTEXT_EE_RE = re.compile(r"(?<![A-Z0-9])EE(?![A-Z0-9])")
ADMINISTRATOR_NAME = "Universal Music Publishing"


class UserResponse(TypedDict):
    value: str
    recursive: bool
    in_place: bool
    adjust_header: bool
    manual_select: bool


class PublisherSelectionResponse(TypedDict):
    value: str
    selected_pubs: list[str]


class EffectiveDateEntry(TypedDict):
    date: str
    territories: list[str]


class AgreementEntry(TypedDict):
    publishers: list[str]
    effective_dates: list[EffectiveDateEntry]


# =================================================
# Soundmouse
# =================================================

SOUNDMOUSE_COLUMNS: List[Tuple[str, float, float]] = [
    ("cue #", 0, 35),
    ("reel #", 35, 60),
    ("title", 60, 225),
    ("role", 225, 250),
    ("name", 250, 340),
    ("society", 340, 450),
    ("usage", 450, 505),
    ("duration", 505, 595),
]

SOUNDMOUSE_HEADER_RE = re.compile(
    r"#\s*Reel\s*No\s*Cue\s*Title\s*Role\s*Name\s*Society\s*Usage\s*Duration",
    re.IGNORECASE,
)


def is_soundmouse_pdf(page: Page) -> bool:
    text = page.extract_text() or ""
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
                    if not current_cue.get("reel") and reel_no:
                        current_cue["reel"] = reel_no

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
RC_ROLE_START_RE = re.compile(r"^(C|A|CA|AR|E)\s+(.+)$")

# Extract society + percent from END of role block
RC_SOCIETY_PERCENT_RE = re.compile(r"(.*?)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)$")

# Known RapidCue descriptor prefixes
RC_DESCRIPTOR_TERMS = {
    "THEME",
    "MAIN TITLE",
    "OPENING",
    "CLOSING",
    "BUMPER",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


def is_rapidcue_pdf_first_page(page: Page) -> bool:
    text = page.extract_text() or ""
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
    lines: List[str] = []
    pages_scanned = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            pages_scanned += 1
            text = page.extract_text() or ""
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
                "descriptors": [],
                "composers": [],
                "publishers": [],
            }
            continue

        if not current_cue:
            continue

        # Peel descriptor prefix (e.g., "Bumper E ...")
        descriptor, remainder = rc_peel_descriptor_prefix(line)
        if descriptor:
            current_cue["descriptors"].append(descriptor)

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
    return cues


# =================================================
# Default parser (unknown format)
# =================================================
def default_extract_lines(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    check_cancel: Callable[[], None] | None = None,
) -> List[str]:
    lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if check_cancel is not None:
                check_cancel()
            text = page.extract_text() or ""
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
    cues: List[Dict[str, Any]] = []
    controlled_norm = [
        (pub, canonicalize_publisher_name(pub, drop_suffixes=True))
        for pub in controlled_publishers
    ]
    lines = default_extract_lines(pdf_path, log_fn=log_fn, check_cancel=check_cancel)
    full_text = " ".join(lines)
    canon_text = canonicalize_text_for_search(full_text)
    matched_publishers_count = 0
    skipped_logo_matches = 0
    seen_publishers: set[str] = set()

    for raw_name, norm_name in controlled_norm:
        if check_cancel is not None:
            check_cancel()
        if not norm_name or raw_name in seen_publishers:
            continue
        start = 0
        matched = False
        while True:
            idx = canon_text.find(norm_name, start)
            if idx == -1:
                break
            context_start = max(0, idx - logo_context_chars)
            context_end = min(
                len(canon_text), idx + len(norm_name) + logo_context_chars
            )
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
                        "descriptors": [],
                        "composers": [],
                        "publishers": [{"name": raw_name}],
                    }
                )
                matched_publishers_count += 1
                seen_publishers.add(raw_name)
                matched = True
                break
            start = idx + len(norm_name)
        if matched and log_fn:
            log_fn(f"default parser: matched_publisher={raw_name}")
    if log_fn:
        log_fn(
            "default parser: "
            f"matched_publishers={matched_publishers_count} | skipped_logo_matches={skipped_logo_matches}"
        )
    return cues


# =================================================
# Router / entry point
# =================================================
def detect_format(pdf_path: Path) -> Literal["soundmouse", "rapidcue", "unknown"]:
    """
    Returns: "soundmouse" | "rapidcue" | "unknown"
    """
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        if is_soundmouse_pdf(first_page):
            return "soundmouse"
        if is_rapidcue_pdf_first_page(first_page):
            return "rapidcue"
    return "unknown"


def filter_logos(cues: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for cue in cues:
        title = cue.get("title", "") or ""
        usage = cue.get("usage")
        if usage == "EE" or LOGO_RE.search(title):
            continue
        out.append(cue)
    return out


def filter_logos_unknown_format(
    cues: list[Dict[str, Any]],
    *,
    lookback: int = 3,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for index, cue in enumerate(cues):
        title = cue.get("title", "") or ""
        usage = cue.get("usage")
        if usage == "EE" or LOGO_RE.search(title):
            continue
        start = max(0, index - lookback)
        should_skip = False
        for prev in cues[start:index]:
            prev_title = prev.get("title", "") or ""
            prev_usage = prev.get("usage")
            if prev_usage == "EE" or LOGO_RE.search(prev_title):
                should_skip = True
                break
        if should_skip:
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


def build_controlled_index(controlled_publishers: List[str]) -> List[Dict[str, Any]]:
    """
    Precompute canonical forms, token sets, and trigram counters for controlled publishers.
    """
    idx: List[Dict[str, Any]] = []
    for pub in controlled_publishers:
        canon = canonicalize_publisher_name(pub, drop_suffixes=True)
        canon_no_space = canon.replace(" ", "")
        tok_set = set(canon.split(" ")) if canon else set()
        tri = char_ngrams_from_canon_no_space(canon_no_space, 3)
        idx.append(
            {
                "raw": pub,
                "canon": canon,
                "canon_no_space": canon_no_space,
                "tokens": tok_set,
                "trigrams": tri,
            }
        )
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
        if remaining_only is not None and c["raw"] not in remaining_only:
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
        "matched_controlled": best["raw"],
        "score": round(best_score, 6),
        "features": {k: round(v, 6) for k, v in (best_feats or {}).items()},
    }


def find_controlled_publishers_present(
    filtered_cues: List[Dict[str, Any]],
    controlled_publishers: List[str],
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
    controlled_index = build_controlled_index(controlled_publishers)

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


def collect_pdfs(
    root: Path, recursive: bool, check_cancel: Callable[[], None] | None = None
) -> list[Path]:
    def _is_in_underscore_dir(path: Path) -> bool:
        return any(parent.name.startswith("_") for parent in path.parents)

    if recursive:
        files = []
        for path in root.rglob("*.pdf"):
            if check_cancel is not None:
                check_cancel()
            if path.is_file() and not _is_in_underscore_dir(path):
                files.append(path)
        return sorted(files)
    files = []
    for path in root.glob("*.pdf"):
        if check_cancel is not None:
            check_cancel()
        if path.is_file():
            files.append(path)
    return sorted(files)


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


def build_xmp_metadata(
    administrator: str,
    agreements: Sequence[AgreementEntry],
) -> bytes:
    """
    Build an XMP packet containing ferp:administrator and ferp:agreements.
    Returns UTF-8 XML bytes.
    """
    admin_value = normalize_text_value(administrator)
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
{agreement_block}    </rdf:Description>
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


def draw_top_right_badge(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    *,
    second_line_text: str,  # dynamic line immediately after the fixed first line
    deal_start_date_text: str,  # dynamic value for "Deal Start Date" right column
    controlled_territory_text: str,  # dynamic value for "Controlled Territory" right column (wraps)
    image_path: str,
    margin_right: float = 5,
    margin_top: float = 5,
    padding: float = 5,
    gap: float = 5,
    image_w: float = 65,
    image_h: float = 45,
    top_font_name: str = "Calibri Bold",
    bottom_font_name: str = "Calibri",
    top_font_size: float = 8,
    bottom_font_size: float = 6,
    top_line_height: float = 8,
    top_line_gap: float = 2.0,  # extra vertical space between top line 1 and line 2
    bottom_line_height: float = 7.5,
    max_text_width: float = 170,  # max width for ALL text area to the right of the image
    min_text_width: float = 170,  # minimum width for ALL text area to the right of the image
    corner_radius: float = 5,
    stroke_rgb=(0, 0, 0),
    stroke_width: float = 1.0,
    # Horizontal rule styling / spacing
    rule_thickness: float = 0.75,
    rule_rgb=(0, 0, 0),
    rule_margin_top: float = 4,
    rule_margin_bottom: float = 1,
    # Two-column layout controls (below divider)
    col_gap: float = 20,  # space between left and right "columns"
    min_right_col_width: float = 40,  # ensures a sane wrap width even if left labels are wide
):
    """
    Draws a rounded-rectangle badge anchored to the top-right corner.
    The rounded rectangle has a fully transparent background (fill=0) and a stroke.

    Layout (right of image):
      Line 1 (fixed):  "Universal Music Publishing admin o/b/o"
      Line 2 (dynamic): second_line_text (can wrap to multiple lines)
      Divider line
      Row 1:  "Deal Start Date"       <deal_start_date_text>
      Row 2:  "Controlled Territory"  <controlled_territory_text>  (wraps within right column)

    Notes:
    - The box width is controlled by max_text_width/min_text_width (text area width).
    - Controlled territory wraps within the right column and will NOT expand the box width.
    - Wrapping is word-based only.
    """

    fixed_line_1 = "Universal Music Publishing admin o/b/o"
    left_label_1 = "Deal Start Date"
    left_label_2 = "Controlled Territory"

    # --- Column widths (below divider) --- (measured in bottom font size)
    left_col_w = max(
        pdfmetrics.stringWidth(left_label_1, bottom_font_name, bottom_font_size),
        pdfmetrics.stringWidth(left_label_2, bottom_font_name, bottom_font_size),
    )

    # --- Decide overall text area width (right of image) ---
    # Start by clamping the total text area width to min/max.
    text_area_w = max(min_text_width, min(max_text_width, max_text_width))

    # Ensure we always have room for the right column; if not, expand text area to accommodate minimum right col.
    # Note: expanding may exceed max_text_width;
    min_total_for_cols = left_col_w + col_gap + min_right_col_width
    if text_area_w < min_total_for_cols:
        text_area_w = min_total_for_cols

    right_col_w = max(text_area_w - left_col_w - col_gap, min_right_col_width)

    # --- Wrap top block (2 lines) ---
    top_line_1 = fixed_line_1
    top_line_2_lines = wrap_text(
        second_line_text, top_font_name, top_font_size, text_area_w
    )

    # --- Wrap right-column values for the two rows (below divider) ---
    deal_date_lines = wrap_text(
        deal_start_date_text, bottom_font_name, bottom_font_size, right_col_w
    )
    territory_lines = wrap_text(
        controlled_territory_text, bottom_font_name, bottom_font_size, right_col_w
    )

    # --- Heights: compute using separate line heights ---
    top_block_h = (
        top_line_height  # first fixed line
        + top_line_gap  # adjustable gap
        + top_line_height * max(len(top_line_2_lines), 1)
    )

    row1_h = bottom_line_height * max(len(deal_date_lines), 1)
    row2_h = bottom_line_height * max(len(territory_lines), 1)

    text_stack_h = (
        top_block_h
        + rule_margin_top
        + rule_thickness
        + rule_margin_bottom
        + row1_h
        + row2_h
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
    img_y = text_stack_top - image_h

    img = ImageReader(image_path)
    c.drawImage(img, img_x, img_y, width=image_w, height=image_h, mask="auto")

    # Text origin
    text_x = img_x + image_w + gap

    # Baseline offsets for each font size / line height
    top_baseline_offset = (top_line_height - top_font_size) * 0.8
    bottom_baseline_offset = (bottom_line_height - bottom_font_size) * 0.8

    c.setFillColorRGB(0, 0, 0)
    cursor_y = text_stack_top
    # ---- Top block (top font size) ----
    c.setFont(top_font_name, top_font_size)

    # Line 1 (fixed)
    cursor_y -= top_line_height
    c.drawString(text_x, cursor_y + top_baseline_offset, top_line_1)

    # Adjustable gap between line 1 and line 2
    cursor_y -= top_line_gap

    # Line 2 (dynamic, wrapped)
    for line in top_line_2_lines:
        cursor_y -= top_line_height
        c.drawString(text_x, cursor_y + top_baseline_offset, line)

    # ---- Divider ----
    cursor_y -= rule_margin_top
    c.saveState()
    c.setLineWidth(rule_thickness)
    c.setStrokeColorRGB(*rule_rgb)
    c.line(text_x, cursor_y, text_x + text_area_w, cursor_y)
    c.restoreState()
    cursor_y -= rule_thickness + rule_margin_bottom

    # ---- Two-column rows (bottom font size) ----
    c.setFont(bottom_font_name, bottom_font_size)

    left_x = text_x
    right_x = text_x + left_col_w + col_gap

    # Row 1: Deal Start Date
    cursor_y -= bottom_line_height
    c.drawString(left_x, cursor_y + bottom_baseline_offset, left_label_1)
    c.drawString(
        right_x,
        cursor_y + bottom_baseline_offset,
        deal_date_lines[0] if deal_date_lines else "",
    )
    for extra in deal_date_lines[1:] if deal_date_lines else []:
        cursor_y -= bottom_line_height
        c.drawString(right_x, cursor_y + bottom_baseline_offset, extra)

    # Row 2: Controlled Territory
    cursor_y -= bottom_line_height
    c.drawString(left_x, cursor_y + bottom_baseline_offset, left_label_2)
    c.drawString(
        right_x,
        cursor_y + bottom_baseline_offset,
        territory_lines[0] if territory_lines else "",
    )
    for extra in territory_lines[1:] if territory_lines else []:
        cursor_y -= bottom_line_height
        c.drawString(right_x, cursor_y + bottom_baseline_offset, extra)

    c.restoreState()

    return {
        "rect": (rect_x, rect_y, rect_w, rect_h),
        "image": (img_x, img_y, image_w, image_h),
        "text_area": (text_x, text_stack_bottom, text_area_w, text_stack_h),
        "columns": {"left_w": left_col_w, "right_w": right_col_w, "gap": col_gap},
        "top_block": {"line1": top_line_1, "line2_lines": top_line_2_lines},
        "rows": {
            "deal_start_date_lines": deal_date_lines,
            "territory_lines": territory_lines,
        },
    }


def draw_top_full_badge(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    *,
    second_line_text: str,  # dynamic line immediately after the fixed first line
    deal_start_date_and_territory: list[dict[str, str]],
    image_path: str,
    margin_right: float = 8,
    margin_top: float = 8,
    padding: float = 5,
    gap: float = 5,
    image_w: float = 65,
    image_h: float = 45,
    # Fonts
    top_font_name: str = "Calibri Bold",
    bottom_font_name: str = "Calibri",
    top_font_size: float = 8,
    bottom_font_size: float = 6,
    # Line metrics
    top_line_height: float = 8,
    top_line_gap: float = 4.0,  # extra vertical space between top line 1 and line 2
    bottom_line_height: float = 7.5,
    # Width controls (this stamp is intended to be "full width" of the header area)
    max_text_width: float = 500,  # max width for ALL text area to the right of the image
    min_text_width: float = 500,  # minimum width for ALL text area to the right of the image
    corner_radius: float = 5,
    stroke_rgb=(0, 0, 0),
    stroke_width: float = 1.0,
    # Horizontal rule styling / spacing
    rule_thickness: float = 0.75,
    rule_rgb=(0, 0, 0),
    rule_margin_top: float = 4,
    rule_margin_bottom: float = 1,
    # Table layout (below divider)
    col_gap: float = 20,  # space between the two columns
    min_right_col_width: float = 40,
    header_bottom_gap: float = 2.0,  # space between column header row and first data row
):
    """
    Draws a rounded-rectangle badge anchored to the top-right corner.
    The rounded rectangle has a fully transparent background (fill=0) and a stroke.

    Layout (right of image):
      Line 1 (fixed):  "Universal Music Publishing admin o/b/o"    (centered)
      Line 2 (dynamic): second_line_text (centered, can wrap to multiple lines)
      Divider line
      Row 1 (headers):  "Deal Start Date"       "Controlled Territory"
      Rows (data):      <effective date>        <territory>   (territory wraps within its column)
                        <effective date>        <territory>
                        ...

    Notes:
    - This function uses a fixed text-area width governed by min_text_width/max_text_width (often equal).
    - Territory wrapping is constrained to the right column and does not widen the box.
    - Wrapping is word-based only.
    """

    fixed_line_1 = "Universal Music Publishing admin o/b/o"
    col_header_1 = "Deal Start Date"
    col_header_2 = "Controlled Territory"

    # --- Decide overall text area width (right of image) ---
    # Clamp total text area width to [min_text_width, max_text_width].
    # Defaults set them equal to create a fixed width.
    text_area_w = max(min_text_width, min(max_text_width, max_text_width))

    # --- Column widths (below divider) ---
    # Left column width must fit the header and typical date text
    left_col_w = pdfmetrics.stringWidth(
        col_header_1, bottom_font_name, bottom_font_size
    )

    # Ensure at least min_right_col_width remains for territory, otherwise expand the text area.
    min_total_for_cols = left_col_w + col_gap + min_right_col_width
    if text_area_w < min_total_for_cols:
        text_area_w = min_total_for_cols

    right_col_w = max(text_area_w - left_col_w - col_gap, min_right_col_width)

    # --- Wrap top block (centered) ---
    # We wrap using the full text area width (not per-column).
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
    img_y = text_stack_top - image_h
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
    }


def add_stamp(
    pdf_path: Path,
    output_path: Path,
    matched_publishers: list[str],
    deal_start_date_text: str,
    controlled_territory_text: str,
    multi_territory_rows: list[dict[str, str]],
) -> None:
    font_path_b = Path(__file__).resolve().parent / "assets" / "Calibrib.ttf"
    font_path = Path(__file__).resolve().parent / "assets" / "Calibri.ttf"
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.jpg"
    pdfmetrics.registerFont(TTFont("Calibri Bold", str(font_path_b)))
    pdfmetrics.registerFont(TTFont("Calibri", str(font_path)))

    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        return

    first_page = reader.pages[0]
    page_w, page_h, offset_x, offset_y = get_page_box(first_page)

    temp_handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    c = canvas.Canvas(str(temp_path), pagesize=(page_w, page_h))

    if multi_territory_rows:
        draw_top_full_badge(
            c,
            page_w,
            page_h,
            second_line_text=" and ".join(matched_publishers),
            deal_start_date_and_territory=multi_territory_rows,
            image_path=str(logo_path),
        )
    else:
        draw_top_right_badge(
            c,
            page_w,
            page_h,
            second_line_text=" and ".join(matched_publishers),
            deal_start_date_text=deal_start_date_text,
            controlled_territory_text=controlled_territory_text,
            image_path=str(logo_path),
        )

    c.showPage()
    c.save()

    stamp_reader = PdfReader(str(temp_path))
    if not stamp_reader.pages:
        return
    stamp_page = stamp_reader.pages[0]

    writer = PdfWriter()
    for index, page in enumerate(reader.pages):
        if index == 0:
            transform = Transformation().translate(offset_x, offset_y)
            merge_transformed = getattr(page, "merge_transformed_page", None)
            if callable(merge_transformed):
                merge_transformed(stamp_page, transform)
            else:
                add_transformation = getattr(stamp_page, "add_transformation", None)
                if callable(add_transformation):
                    add_transformation(transform)
                page.merge_page(stamp_page)
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


def format_effective_date(date_text: str) -> str:
    if not date_text:
        return "N/A"
    try:
        dt = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return "N/A"
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def resolve_publisher_fields(
    matched_publishers: list[str],
    category_entries: list[dict],
) -> tuple[str, str]:
    territory = "N/A"
    effective_date = "N/A"
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher in matched_set:
            territory_value = str(entry.get("territory", "")).strip()
            if territory_value and territory == "N/A":
                territory = territory_value
            date_value = str(entry.get("effective date", "")).strip()
            if date_value and effective_date == "N/A":
                effective_date = format_effective_date(date_value)
        if territory != "N/A" and effective_date != "N/A":
            break
    return effective_date, territory


def resolve_publisher_fields_raw(
    matched_publishers: list[str],
    category_entries: list[dict],
) -> tuple[str, str]:
    territory = ""
    effective_date = ""
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher in matched_set:
            territory_value = str(entry.get("territory", "")).strip()
            if territory_value and not territory:
                territory = territory_value
            date_value = str(entry.get("effective date", "")).strip()
            if date_value and not effective_date:
                effective_date = date_value
        if territory and effective_date:
            break
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
) -> list[dict[str, str]]:
    matched_set = {p.strip() for p in matched_publishers if p.strip()}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in category_entries:
        publisher = str(entry.get("publisher", "")).strip()
        if publisher not in matched_set:
            continue
        multi_territory = entry.get("multi_territory") or []
        if not isinstance(multi_territory, list):
            continue
        for row in multi_territory:
            if not isinstance(row, dict):
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
    return rows


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
            effective_rows = [
                {"effective date": raw_date, "territory": raw_territory}
            ]
    effective_entries = normalize_effective_entries(effective_rows)
    return [
        {
            "publishers": matched_publishers,
            "effective_dates": effective_entries,
        }
    ]


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    env_paths = ctx.environment.get("paths", {})
    cache_dir = Path(env_paths.get("cache_dir"))
    cache_path = cache_dir / "publishers_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file '{cache_path}' not found")

    target_dir = ctx.target_path
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Target '{target_dir}' is not a directory.")

    with cache_path.open("r", encoding="utf-8") as handle:
        pubs = json.load(handle)

    payload = api.request_input_json(
        "Publisher catalog code (e.g., 'uvs', 'amz', etc.)",
        id="ferp_process_cue_sheets_cat_code",
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
        ],
        payload_type=UserResponse,
    )

    cat_code = payload["value"].lower()
    if cat_code not in pubs.keys():
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: Invalid Catalog",
                "Info": f"The code {cat_code} does not exist.",
            }
        )
        return

    cat_object: List[dict] = pubs[cat_code]
    con_pubs: List[str] = [entry["publisher"] for entry in cat_object]

    recursive = payload["recursive"]
    in_place = payload["in_place"]
    adjust_header = payload["adjust_header"]
    manual_select = payload["manual_select"]

    selected_pubs: list[str] = []
    if manual_select and con_pubs:
        selection_payload = api.request_input_json(
            "Select controlled publishers",
            id="ferp_process_cue_sheets_selected_pubs",
            fields=[
                {
                    "id": "selected_pubs",
                    "type": "multi_select",
                    "label": f"Publisher catalog code: {cat_code.upper()}",
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

    raw_cues = []
    temp_paths: list[Path] = []

    def _cleanup() -> None:
        for path in temp_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    api.register_cleanup(_cleanup)

    api.check_cancel()
    pdf_files = collect_pdfs(
        target_dir, recursive=recursive, check_cancel=api.check_cancel
    )
    total_files = len(pdf_files)
    created_dirs: set[str] = set()
    api.log(
        "info",
        f"Cue sheet PDFs found={total_files} | category_code={cat_code} | recursive={recursive}",
    )

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.check_cancel()
        api.progress(current=index, total=total_files, unit="files")
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
            elif fmt == "rapidcue":
                api.log("debug", f"{pdf_path.name}: detected RapidCue format")
                raw_cues = parse_rapidcue(
                    pdf_path,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )
            else:
                api.log(
                    "debug", f"{pdf_path.name}: unknown format; using default parser"
                )
                raw_cues = parse_default(
                    pdf_path,
                    con_pubs,
                    log_fn=lambda msg: api.log("debug", f"{pdf_path.name}: {msg}"),
                    check_cancel=api.check_cancel,
                )

            api.check_cancel()
            if fmt == "unknown":
                filtered_cues = raw_cues
            else:
                filtered_cues = filter_logos(raw_cues)
            result = find_controlled_publishers_present(
                filtered_cues,
                con_pubs,
                auto_match_threshold=0.92,
                review_threshold=0.85,
            )
            matched_publishers = result.get("found_controlled_publishers", [])
            accuracy_audits = result.get("evidence_by_controlled", {})
        if matched_publishers:
            work_path = pdf_path
            if adjust_header:
                temp_path = pdf_path.with_suffix(".header.tmp.pdf")
                temp_paths.append(temp_path)
                shutil.copyfile(pdf_path, temp_path)
                make_top_space_first_page_inplace(temp_path, header_top_space)
                work_path = temp_path
            api.check_cancel()
            deal_start_date_text, controlled_territory_text = resolve_publisher_fields(
                matched_publishers,
                cat_object,
            )
            multi_territory_rows = resolve_multi_territory_rows(
                matched_publishers,
                cat_object,
            )
            agreements = build_agreements(
                matched_publishers,
                cat_object,
                multi_territory_rows,
            )
            xmp_bytes = build_xmp_metadata(ADMINISTRATOR_NAME, agreements)
            if in_place:
                add_stamp(
                    work_path,
                    pdf_path,
                    matched_publishers,
                    deal_start_date_text,
                    controlled_territory_text,
                    multi_territory_rows,
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
                    deal_start_date_text,
                    controlled_territory_text,
                    multi_territory_rows,
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
            "Category Code": cat_code,
        }
    )


if __name__ == "__main__":
    main()
