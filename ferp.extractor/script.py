from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import extract_msg

from ferp.fscp.scripts import sdk

SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}


def _extract_from_msg(msg_file: Path, output_dir: Path) -> int:
    msg = extract_msg.Message(str(msg_file))
    attachments = list(msg.attachments)

    email_output_dir = output_dir / msg_file.stem
    email_output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for attachment in attachments:
        name = attachment.longFilename or attachment.shortFilename
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        attachment.save(customPath=email_output_dir)
        saved += 1

    return saved


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    zip_path = ctx.target_path

    output_dir = zip_path.parent / f"{zip_path.stem}_attachments"
    temp_dir = output_dir / "_unzipped"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    api.log("info", f"Extracting {zip_path.name} to {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(temp_dir)

    msg_files = list(temp_dir.rglob("*.msg"))
    if not msg_files:
        api.log("warn", "No .msg files found in archive.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        api.emit_result(
            {
                "_status": "warn",
                "_title": "Warning: No .msg Files Found",
                "Output": None,
                "Processed Messages": 0,
            }
        )
        return

    total_saved = 0

    for index, msg_file in enumerate(msg_files, start=1):
        api.progress(current=index, total=len(msg_files), unit="messages", every=10)
        saved = _extract_from_msg(msg_file, output_dir)
        total_saved += saved
        api.log("info", f"{msg_file.name}: saved {saved} attachment(s)")

    shutil.rmtree(temp_dir, ignore_errors=True)

    api.emit_result(
        {
            "_title": "Attachment Extraction Finished",
            "Output Directory": str(output_dir),
            "Messages Processed": len(msg_files),
            "Attachments Extracted": total_saved,
        }
    )


if __name__ == "__main__":
    main()
