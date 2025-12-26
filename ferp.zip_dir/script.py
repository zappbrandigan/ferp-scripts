from __future__ import annotations

from pathlib import Path
import zipfile

from ferp.fscp.scripts import sdk


def _collect_entries(target_dir: Path, zip_path: Path) -> list[Path]:
    entries: list[Path] = []
    for item in target_dir.rglob("*"):
        if item == zip_path:
            continue
        if item.suffix == ".zip" and item.parent == target_dir:
            # Skip sibling zip files to avoid recursion
            continue
        entries.append(item)
    return entries


@sdk.script
def main(context: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = context.target_path

    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"{target_dir} is not a directory")

    zip_path = target_dir.parent / f"{target_dir.name}.zip"
    if zip_path.exists():
        overwrite = api.confirm(
            f"'{zip_path.name}' already exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "message": "Zip creation cancelled",
                    "zip_path": str(zip_path),
                }
            )
            return

    entries = _collect_entries(target_dir, zip_path)
    total = len(entries) or 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(entries, start=1):
            arcname = item.relative_to(target_dir)
            zf.write(item, arcname=arcname)
            if idx == 1 or idx == total or idx % 10 == 0:
                api.progress(current=idx, total=total, unit="files")

    api.emit_result(
        {
            "message": "Created zip archive",
            "zip_path": str(zip_path),
            "entries": total,
        }
    )


if __name__ == "__main__":
    main()
