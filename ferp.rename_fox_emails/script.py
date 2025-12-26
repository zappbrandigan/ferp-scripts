from __future__ import annotations

from pathlib import Path

from ferp.fscp.scripts import sdk

_SUFFIX = ".pdf"
_MAX_CHARS = 12


def _derive_name(path: Path) -> str:
    stem = path.stem
    tail = stem[-_MAX_CHARS:] if len(stem) > _MAX_CHARS else stem
    tail = tail or stem or "document"
    return tail


def _build_destination(directory: Path, base: str, suffix: str) -> Path:
    candidate = directory / f"{base}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


@sdk.script
def main(context: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = Path(context.target_path)

    if not target_dir.exists() or not target_dir.is_dir():
        raise RuntimeError(f"Target directory '{target_dir}' is not accessible.")

    pdf_files = sorted(
        [path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == _SUFFIX],
        key=lambda item: item.name.lower(),
    )

    if not pdf_files:
        api.log("info", "No PDF files found to rename.")
        api.emit_result({"renamed": 0})
        return

    renamed = 0
    for index, pdf in enumerate(pdf_files, start=1):
        new_base = _derive_name(pdf)
        new_name = f"{new_base}{_SUFFIX}"
        if pdf.name == new_name:
            api.log("debug", f"Skipping {pdf.name}: already matches target pattern.")
            continue

        destination = _build_destination(target_dir, new_base, _SUFFIX)

        try:
            pdf.rename(destination)
            renamed += 1
            api.log("info", f"Renamed '{pdf.name}' -> '{destination.name}'")
        except OSError as exc:
            api.log("error", f"Failed to rename '{pdf.name}': {exc}")

        api.progress(current=index, total=len(pdf_files), unit="files")

    api.emit_result({"renamed": renamed, "total": len(pdf_files)})


if __name__ == "__main__":
    main()
