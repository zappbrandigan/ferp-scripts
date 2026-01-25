from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Callable, Sequence

import pdfplumber

from ferp.fscp.scripts import sdk


@dataclass(frozen=True)
class PdfText:
    lines: list[str]
    pages: int
    text_lines: int


def _resolve_input_path(target_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = target_dir / path
    if path.suffix == "":
        path = target_dir / path.with_suffix(".pdf")
    return path


def _extract_pdf_text(
    pdf_path: Path,
    api: sdk.ScriptAPI,
    check_cancel: Callable[[], None] | None = None,
) -> PdfText:
    lines: list[str] = []
    text_line_count = 0
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages, start=1):
                if check_cancel is not None:
                    check_cancel()
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    api.log(
                        "warn",
                        (
                            "Text extraction failed for "
                            f"'{pdf_path.name}' (page {page_index}): {exc}"
                        ),
                    )
                    text = ""

                lines.append(f"--- Page {page_index} ---")
                if text:
                    page_lines = [line.rstrip() for line in text.splitlines()]
                    text_line_count += len(page_lines)
                    lines.extend(page_lines)
                else:
                    lines.append("")
    except Exception as exc:  # noqa: BLE001
        api.log("warn", f"Failed to read '{pdf_path}': {exc}")
        raise

    return PdfText(lines=lines, pages=total_pages, text_lines=text_line_count)


def _diff_counts(opcodes: Sequence[tuple[str, int, int, int, int]]) -> dict[str, int]:
    inserted = 0
    deleted = 0
    replaced_a = 0
    replaced_b = 0
    change_blocks = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        change_blocks += 1
        if tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            replaced_a += i2 - i1
            replaced_b += j2 - j1

    return {
        "change_blocks": change_blocks,
        "inserted_lines": inserted,
        "deleted_lines": deleted,
        "replaced_lines_a": replaced_a,
        "replaced_lines_b": replaced_b,
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = ctx.target_path

    file_a = api.request_input(
        "First PDF file (relative to target directory)",
        id="pdf_text_diff_file_a",
    )
    file_b = api.request_input(
        "Second PDF file (relative to target directory)",
        id="pdf_text_diff_file_b",
    )
    if not file_a or not file_b:
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: Script Canceled",
                "Info": "Both files are required.",
            }
        )
        return

    a_path = _resolve_input_path(target_dir, file_a)
    b_path = _resolve_input_path(target_dir, file_b)
    if not a_path.exists():
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: Script Canceled",
                "Info": "File does not exist.",
                "File Path": str(a_path),
            }
        )
        return
    if not b_path.exists():
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: Script Canceled",
                "Info": "File does not exist.",
                "File Path": str(b_path),
            }
        )
        return

    api.log("info", f"Extracting text from '{a_path.name}' and '{b_path.name}'")

    api.check_cancel()
    a_text = _extract_pdf_text(a_path, api, check_cancel=api.check_cancel)
    api.check_cancel()
    b_text = _extract_pdf_text(b_path, api, check_cancel=api.check_cancel)

    api.log(
        "info",
        (
            f"{a_path.name}: {a_text.pages} pages, {a_text.text_lines} text lines | "
            f"{b_path.name}: {b_text.pages} pages, {b_text.text_lines} text lines"
        ),
    )

    api.check_cancel()
    matcher = SequenceMatcher(a=a_text.lines, b=b_text.lines)
    opcodes = matcher.get_opcodes()
    counts = _diff_counts(opcodes)
    diff_lines = list(
        unified_diff(
            a_text.lines,
            b_text.lines,
            fromfile=a_path.name,
            tofile=b_path.name,
            lineterm="",
            n=3,
        )
    )

    summary_lines = [
        "PDF Text Diff Summary",
        f"File A: {a_path}",
        f"File B: {b_path}",
        f"Pages A: {a_text.pages} | Pages B: {b_text.pages}",
        f"Text lines A: {a_text.text_lines} | Text lines B: {b_text.text_lines}",
        (
            "Inserted lines: {inserted_lines} | Deleted lines: {deleted_lines} | "
            "Replaced lines A->B: {replaced_lines_a}->{replaced_lines_b} | "
            "Change blocks: {change_blocks}"
        ).format(**counts),
        "",
        "Unified Diff:",
    ]
    if diff_lines:
        summary_lines.extend(diff_lines)
    else:
        summary_lines.append("(no differences found)")

    output_dir = a_path.parent
    if a_path.parent != b_path.parent:
        api.log(
            "warn",
            (
                "Input files are in different directories; writing diff output next to "
                f"the first file: {a_path.parent}"
            ),
        )

    summary_path = output_dir / f"{a_path.stem}_vs_{b_path.stem}_pdf_text_diff.txt"
    _write_text(summary_path, "\n".join(summary_lines).rstrip() + "\n")

    api.emit_result(
        {
            "_title": "PDF Diff Created",
            "Summary Path": str(summary_path),
            "Changes Found": counts["change_blocks"],
            "Results": "\n".join(summary_lines).rstrip(),
        }
    )


if __name__ == "__main__":
    main()
