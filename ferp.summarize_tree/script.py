from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator, cast

from ferp.fscp.scripts import sdk

MAX_DEPTH = 4


def _walk(
    root: Path, check_cancel: Callable[[], None] | None = None
) -> Iterator[tuple[Path, str, Counter[str]]]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        if check_cancel is not None:
            check_cancel()
        directory, depth = stack.pop()
        counter: Counter[str] = Counter()
        try:
            with os.scandir(directory) as entries:
                subdirs = []
                for entry in entries:
                    if check_cancel is not None:
                        check_cancel()
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
        rel_path = cast(str, entry["relative_path"] or ".")
        if rel_path == ".":
            path_label = "."
        else:
            parts = rel_path.split(os.sep)
            depth = max(len(parts) - 1, 0)
            name = parts[-1] if parts[-1] else rel_path
            indent = "  " * depth
            prefix = "|-- " if depth > 0 else ""
            path_label = f"{indent}{prefix}{name}"
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
                    rows.append([path_label, str(total), label, str(count)])
                else:
                    rows.append(["", "", label, str(count)])
        else:
            rows.append([path_label, str(total), "(no files)", "0"])

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


def _build_destination(directory: Path, base: str, suffix: str) -> Path:
    candidate = directory.parent / f"{base}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory.parent / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_header_markdown(header: dict[str, str]) -> list[str]:
    lines = ["# Directory Summary", ""]
    lines.extend(f"**{k}**: {v}  " for k, v in header.items())
    lines.append("")
    return lines


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path
    confirm_export = api.confirm(
        "Would you like to export results to Markdown?",
        id="ferp_summarize_tree",
    )

    summary = []
    for directory, rel_path, file_counter in _walk(root, check_cancel=api.check_cancel):
        api.check_cancel()
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
    header = {
        "Root": str(root.name),
        "Max Depth": str(MAX_DEPTH),
        "Directories Scanned": str(len(summary)),
        "Total Files": str(total_files),
    }
    lines = _build_summary_table(summary).splitlines()
    export_path: Path | None = None
    if confirm_export:
        output_dir = root if root.is_dir() else root.parent
        base_name = f"{root.name or 'directory'}_summary"
        export_path = _build_destination(output_dir, base_name, ".md")
        md_wrapper = "```txt\n{table}\n```"
        try:
            markdown_lines = (
                _format_header_markdown(header)
                + md_wrapper.format(table=_build_summary_table(summary)).splitlines()
            )
            _write_text(export_path, "\n".join(markdown_lines).rstrip() + "\n")
            api.log("info", f"Markdown summary written to {export_path}")
        except OSError as exc:
            api.log("error", f"Failed to write markdown summary: {exc}")
            export_path = None

    api.emit_result(
        {
            "_title": "Directory Summary",
            **(header),
            "Markdown Path": str(export_path.relative_to(root.parents[1]))
            if export_path
            else "Not exported.",
            "Extension Details": "\n" + "\n".join(lines),
        }
    )


if __name__ == "__main__":
    main()
