import shutil
import zipfile
from pathlib import Path, PurePosixPath

from ferp.fscp.scripts import sdk


def _common_root(members: list[zipfile.ZipInfo]) -> str | None:
    root: str | None = None
    for info in members:
        parts = [
            part for part in PurePosixPath(info.filename).parts if part not in {"", "."}
        ]
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


def _extract(zip_path: Path, api: sdk.ScriptAPI) -> None:
    if not zip_path.exists() or zip_path.suffix.lower() != ".zip":
        raise ValueError("Selected file is not a zip archive.")

    output_dir = zip_path.parent / zip_path.stem
    if output_dir.exists():
        overwrite = api.confirm(
            f"Directory '{output_dir.name}' exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            api.emit_result(
                {
                    "message": "Extraction cancelled",
                    "output_dir": str(output_dir),
                }
            )
            return

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        root_name = _common_root(members)
        extract_root = output_dir
        output_dir.mkdir(exist_ok=True)

        total = len(members) or 1
        for idx, member in enumerate(members, start=1):
            zf.extract(member, extract_root)
            if idx == 1 or idx == total or idx % 25 == 0:
                api.progress(current=idx, total=total, unit="files")

    if root_name == output_dir.name:
        _flatten_nested_root(output_dir)

    api.emit_result(
        {
            "message": "Archive extracted",
            "zip_path": str(zip_path),
            "output_dir": str(output_dir),
        }
    )


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    _extract(ctx.target_path, api)


if __name__ == "__main__":
    main()
