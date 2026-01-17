from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ferp.fscp.scripts import sdk

BACKUP_SUFFIX = ".backup.zip"


def _backup_stem(backup_path: Path) -> str | None:
    name = backup_path.name
    if name.endswith(BACKUP_SUFFIX):
        return name[: -len(BACKUP_SUFFIX)]
    return None


def _restore(backup_path: Path, api: sdk.ScriptAPI) -> None:
    if not backup_path.exists() or not backup_path.is_file():
        raise ValueError("Selected file is not a backup archive.")

    base_name = _backup_stem(backup_path)
    if not base_name:
        raise ValueError("Selected file is not a backup archive.")

    output_dir = backup_path.parent / base_name
    if output_dir.exists():
        overwrite = api.confirm(
            f"Directory '{output_dir.name}' exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "message": "Restore cancelled",
                    "output_dir": str(output_dir),
                }
            )
            return

        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    with zipfile.ZipFile(backup_path, "r") as zf:
        members = zf.infolist()
        output_dir.mkdir(parents=True, exist_ok=True)

        total = len(members) or 1
        for idx, member in enumerate(members, start=1):
            zf.extract(member, output_dir)
            if idx == 1 or idx == total or idx % 25 == 0:
                api.progress(current=idx, total=total, unit="files")

    api.emit_result(
        {
            "message": "Backup restored",
            "backup_path": str(backup_path),
            "output_dir": str(output_dir),
        }
    )


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    _restore(ctx.target_path, api)


if __name__ == "__main__":
    main()
