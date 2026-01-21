from __future__ import annotations

import zipfile
from pathlib import Path

from ferp.fscp.scripts import sdk


def _collect_entries(target_dir: Path) -> list[Path]:
    """Return all descendants to include in the backup zip."""
    entries: list[Path] = []
    for item in target_dir.rglob("*"):
        entries.append(item)
    return entries


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    """Create a .backup.zip alongside the target directory."""
    target_dir = ctx.target_path

    backup_path = target_dir.parent / f"{target_dir.name}.backup.zip"
    if backup_path.exists():
        overwrite = api.confirm(
            f"'{backup_path.name}' already exists. Overwrite?",
            id="ferp_backup_dir",
        )
        if not overwrite:
            api.emit_result(
                {
                    "_title": "Backup Canceled by User",
                    "_status": "warn",
                    "Info": "No file operations were performed.",
                }
            )
            return

    # Include directories to preserve empty folders in the archive.
    entries = _collect_entries(target_dir)
    total = len(entries) or 1

    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(entries, start=1):
            arcname = item.relative_to(target_dir)
            zf.write(item, arcname=arcname)
            api.progress(current=idx, total=total, unit="files", every=10)

    api.emit_result(
        {
            "_title": "Backup Created",
            "Items Included": total,
            "Backup Name": str(backup_path.name),
        }
    )


if __name__ == "__main__":
    main()
