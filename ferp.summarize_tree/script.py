from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterator, cast

from ferp.fscp.scripts import sdk

MAX_DEPTH = 4


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
                    if name.startswith('.'):
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

        rel = str(directory.relative_to(root)) if directory != root else ''
        yield directory, rel, counter

        for subdir in sorted(subdirs, reverse=True):
            stack.append((subdir, depth + 1))


def _format_entry(entry: dict[str, object]) -> str:
    path = entry["relative_path"] or "."
    total = entry["total_files"]
    extensions = cast(dict, entry["extensions"])
    ext_lines = []
    for ext, count in extensions.items():
        label = ext or "<no extension>"
        ext_lines.append(f"    - {label}: {count}")
    ext_block = "\n".join(ext_lines) if ext_lines else "    - (no files)"
    return f"\n▶ {path} — {total} file(s)\n{ext_block}"


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    root = ctx.target_path
    if not root.exists() or not root.is_dir():
        raise ValueError("Select a directory before running this script.")

    summary = []
    for directory, rel_path, file_counter in _walk(root):
        total = sum(file_counter.values())
        entry = {
            "relative_path": rel_path,
            "total_files": total,
            "extensions": dict(sorted(file_counter.items())),
        }
        summary.append(entry)
        ext_details = ", ".join(f"{ext or '<no ext>'}: {count}" for ext, count in entry["extensions"].items())
        api.log(
            "info",
            f"{rel_path or '.'}: {total} file(s)" + (f" ({ext_details})" if ext_details else ""),
        )

    lines = [_format_entry(entry) for entry in summary]

    api.emit_result({
        "root": str(root),
        # "directories": summary,
        "summary": "\n".join(lines),
    })
    api.exit(code=0)


if __name__ == "__main__":
    main()
