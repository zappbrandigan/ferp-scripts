import shutil
import zipfile
from pathlib import Path, PurePosixPath

import py7zr

from ferp.fscp.scripts import sdk


def _common_root(paths: list[str]) -> str | None:
    root: str | None = None
    for path in paths:
        parts = [part for part in PurePosixPath(path).parts if part not in {"", "."}]
        if not parts:
            continue
        head = parts[0]
        if head in {".."}:
            return None
        if root is None:
            root = head
        elif root != head:
            return None
    return root


def _flatten_nested_root(root: Path) -> None:
    nested = root / root.name
    if not nested.exists() or not nested.is_dir():
        return

    for child in nested.iterdir():
        destination = root / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        child.rename(destination)

    nested.rmdir()


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_path = ctx.target_path

    output_dir = target_path.parent / target_path.stem
    if output_dir.exists():
        overwrite = api.confirm(
            f"Directory '{output_dir.name}' exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "_status": "warn",
                    "_title": "Extraction Canceled by User",
                    "Info": "No file operations were performed.",
                }
            )
            return

    extract_root = output_dir
    output_dir.mkdir(exist_ok=True)

    if target_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(target_path, "r") as zf:
            members = zf.infolist()
            root_name = _common_root([member.filename for member in members])

            total = len(members) or 1
            for idx, member in enumerate(members, start=1):
                zf.extract(member, extract_root)
                api.progress(current=idx, total=total, unit="files", every=25)
    else:
        with py7zr.SevenZipFile(target_path, mode="r") as zf:
            names = zf.getnames()
            root_name = _common_root(names)
            total = len(names) or 1
            api.progress(current=0, total=total, unit="files")
            zf.extractall(path=extract_root)
            api.progress(current=total, total=total, unit="files")

    if root_name == output_dir.name:
        _flatten_nested_root(output_dir)

    api.emit_result(
        {
            "_title": "Archive Extracted",
            "Target Path": str(target_path),
            "Output Location": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
