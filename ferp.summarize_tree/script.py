from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterator, cast

from ferp.fscp.scripts import sdk

MAX_DEPTH = 4
MAX_OUTPUT_LINES = 500


def _walk(root: Path) -> Iterator[tuple[Path, str, Counter[str]]]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        counter: Counter[str] = Counter()
        try:
            with os.scandir(directory) as entries:
                subdirs = []
                for entry in entries:
                    name = entry.name
                    if name.startswith("."):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth < MAX_DEPTH:
                                subdirs.append(Path(entry.path))
                            continue
                        if entry.is_file(follow_symlinks=False):
                            ext = Path(name).suffix.lower()
                            counter[ext] += 1
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            continue

        rel = str(directory.relative_to(root)) if directory != root else ""
        yield directory, rel, counter

        for subdir in sorted(subdirs, reverse=True):
            stack.append((subdir, depth + 1))


def _build_summary_table(entries: list[dict[str, object]]) -> str:
    headers = ["Directory", "Files", "Extension", "Count"]
    rows: list[list[str]] = []
    for entry in entries:
        path = cast(str, entry["relative_path"] or ".")
        total = entry["total_files"]
        extensions = cast(dict, entry["extensions"])
        if extensions:
            sorted_exts = sorted(
                extensions.items(),
                key=lambda item: (-item[1], item[0] or ""),
            )
            for index, (ext, count) in enumerate(sorted_exts):
                label = ext or "<no extension>"
                if index == 0:
                    rows.append([path, str(total), label, str(count)])
                else:
                    rows.append(["", "", label, str(count)])
        else:
            rows.append([path, str(total), "(no files)", "0"])

    widths = []
    for col_index, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            if col_index < len(row):
                max_len = max(max_len, len(row[col_index]))
        widths.append(max(max_len, 3))

    align_right = {1, 3}

    def _format_row(values: list[str]) -> str:
        padded = []
        for idx, value in enumerate(values):
            width = widths[idx]
            if idx in align_right:
                padded.append(value.rjust(width))
            else:
                padded.append(value.ljust(width))
        return "| " + " | ".join(padded) + " |"

    def _format_separator() -> str:
        parts = []
        for idx, width in enumerate(widths):
            if idx in align_right:
                parts.append("-" * (width - 1) + ":")
            else:
                parts.append(":" + "-" * (width - 1))
        return "| " + " | ".join(parts) + " |"

    lines = [
        _format_row(headers),
        _format_separator(),
    ]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def _truncate_lines(lines: list[str]) -> list[str]:
    if len(lines) <= MAX_OUTPUT_LINES:
        return lines
    remaining = MAX_OUTPUT_LINES - 1
    return lines[:remaining] + [
        f"... output truncated after {MAX_OUTPUT_LINES} lines ..."
    ]


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path
    if not root.exists():
        api.emit_result(
            {
                "root": str(root),
                "summary": "Target path does not exist.\nNo directory summary generated.",
            }
        )
        api.exit(code=0)
        return
    if not root.is_dir():
        api.emit_result(
            {
                "root": str(root),
                "summary": (
                    f"Target is a file: {root}\nSelect a directory to summarize."
                ),
            }
        )
        api.exit(code=0)
        return

    summary = []
    for directory, rel_path, file_counter in _walk(root):
        total = sum(file_counter.values())
        entry = {
            "relative_path": rel_path,
            "total_files": total,
            "extensions": dict(sorted(file_counter.items())),
        }
        summary.append(entry)
        ext_details = ", ".join(
            f"{ext or '<no ext>'}: {count}"
            for ext, count in entry["extensions"].items()
        )
        api.log(
            "info",
            f"{rel_path or '.'}: {total} file(s)"
            + (f" ({ext_details})" if ext_details else ""),
        )

    total_files = sum(entry["total_files"] for entry in summary)
    header = [
        "Directory Summary",
        f"Root: {root}",
        f"Max depth: {MAX_DEPTH}",
        f"Directories scanned: {len(summary)}",
        f"Total files: {total_files}",
        "",
    ]
    lines = header + _build_summary_table(summary).splitlines()
    lines = _truncate_lines(lines)

    api.emit_result(
        {
            "root": str(root),
            # "directories": summary,
            "summary": "\n".join(lines),
        }
    )
    api.exit(code=0)


if __name__ == "__main__":
    main()
