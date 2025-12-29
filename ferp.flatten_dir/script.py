from __future__ import annotations

import shutil

from ferp.fscp.scripts import sdk


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    confirm = api.confirm(
        f"This will flatten all subdirectories in {ctx.target_path.stem}. Continue?",
        default=False
    )
    if not confirm:
        api.emit_result(
            {
                "message": "Script cancelled",
            }
        )
        return
    target = ctx.target_path
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Target must be a directory. Received: {target}")

    api.log("info", f"Flattening {target}")

    files_moved = 0
    dirs_removed = 0
    entries = list(target.rglob("*"))
    total = max(len(entries), 1)

    for idx, entry in enumerate(entries, start=1):
        if entry.is_file():
            destination = target / entry.name
            counter = 1
            while destination.exists():
                destination = target / f"{entry.stem}_{counter}{entry.suffix}"
                counter += 1

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), destination)
            files_moved += 1

        elif entry.is_dir():
            # Skip the root directory itself--we only want subdirectories.
            if entry == target:
                continue
            try:
                entry.rmdir()
                dirs_removed += 1
            except OSError:
                # Directory not empty (files yet to be processed); skip for now.
                pass

        if idx == 1 or idx == total or idx % 20 == 0:
            api.progress(current=idx, total=total, unit="items")

    # Clean up any empty directories left after moving files.
    for dir_path in sorted(target.rglob("*"), reverse=True):
        if dir_path.is_dir():
            try:
                dir_path.rmdir()
                dirs_removed += 1
            except OSError:
                pass

    api.emit_result(
        {
            "message": "Directory flattened",
            "target": str(target),
            "files_moved": files_moved,
            "directories_removed": dirs_removed,
        }
    )


if __name__ == "__main__":
    main()
