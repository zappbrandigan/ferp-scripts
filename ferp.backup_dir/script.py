from __future__ import annotations

from pathlib import Path
import zipfile

from ferp.fscp.scripts import sdk


def _collect_entries(target_dir: Path) -> list[Path]:
    entries: list[Path] = []
    for item in target_dir.rglob("*"):
        entries.append(item)
    return entries


def _backup(target_dir: Path, api: sdk.ScriptAPI) -> None:
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"{target_dir} is not a directory")

    backup_path = target_dir.parent / f"{target_dir.name}.backup.zip"
    if backup_path.exists():
        overwrite = api.confirm(
            f"'{backup_path.name}' already exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "message": "Backup cancelled",
                    "backup_path": str(backup_path),
                }
            )
            return

    entries = _collect_entries(target_dir)
    total = len(entries) or 1

    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(entries, start=1):
            arcname = item.relative_to(target_dir)
            zf.write(item, arcname=arcname)
            if idx == 1 or idx == total or idx % 10 == 0:
                api.progress(current=idx, total=total, unit="files")

    api.emit_result(
        {
            "message": "Backup created",
            "backup_path": str(backup_path),
            "entries": total,
        }
    )


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    _backup(ctx.target_path, api)


if __name__ == "__main__":
    main()
