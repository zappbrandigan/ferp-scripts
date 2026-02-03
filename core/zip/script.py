from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

from ferp.fscp.scripts import sdk


def _collect_entries(
    target_dir: Path,
    zip_path: Path,
    check_cancel: Callable[[], None] | None = None,
) -> list[Path]:
    entries: list[Path] = []
    for item in target_dir.rglob("*"):
        if check_cancel is not None:
            check_cancel()
        if item == zip_path:
            continue
        if item.suffix == ".zip" and item.parent == target_dir:
            # Skip sibling zip files to avoid recursion
            continue
        entries.append(item)
    return entries


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = ctx.target_path

    zip_path = target_dir.parent / f"{target_dir.name}.zip"
    created = False

    def _cleanup() -> None:
        if created:
            return
        if zip_path.exists():
            zip_path.unlink()

    api.register_cleanup(_cleanup)
    if zip_path.exists():
        overwrite = api.confirm(
            f"'{zip_path.name}' already exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "_status": "warn",
                    "_title": "Zip Creation Canceled by User",
                    "Info": "No file operations were performed.",
                }
            )
            return

    entries = _collect_entries(
        target_dir, zip_path, check_cancel=api.check_cancel
    )
    total = len(entries) or 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(entries, start=1):
            api.check_cancel()
            arcname = item.relative_to(target_dir)
            zf.write(item, arcname=arcname)
            api.progress(current=idx, total=total, unit="files", every=10)

    created = True
    api.emit_result(
        {
            "_title": "Zip Archive Created",
            "Zip Location": str(zip_path),
            "Entries": total,
        }
    )


if __name__ == "__main__":
    main()
