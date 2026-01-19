import json
import os
import tempfile
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2._page import PageObject
from PyPDF2.generic import DictionaryObject, NameObject

from ferp.fscp.scripts import sdk


def _make_top_space_first_page_inplace(
    pdf_path: Path,
    top_space_pts: float,
    scale: float = 0.95,
):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for i, src_page in enumerate(reader.pages):
        if i != 0:
            writer.add_page(src_page)
            continue

        # Prefer CropBox visually; fall back to MediaBox
        box = src_page.cropbox or src_page.mediabox
        W = float(box.width)
        H = float(box.height)

        T = float(top_space_pts)
        s = float(scale)
        dx = (1.0 - s) * W / 2.0
        dy = (1.0 - s) * H - T

        dst_page = PageObject.create_blank_page(width=W, height=H)
        transform = Transformation().scale(s, s).translate(dx, dy)
        merge_transformed = getattr(dst_page, "merge_transformed_page", None)
        if callable(merge_transformed):
            merge_transformed(src_page, transform)
        else:
            add_transformation = getattr(src_page, "add_transformation", None)
            if callable(add_transformation):
                add_transformation(transform)
            dst_page.merge_page(src_page)
        writer.add_page(dst_page)

    # ---- Copy Document Info metadata ----
    if reader.metadata:
        md = {}
        for k, v in reader.metadata.items():
            if k and v is not None:
                md[str(k)] = str(v)
        if md:
            writer.add_metadata(md)

    # ---- Best-effort XMP metadata preservation ----
    try:
        root_obj = reader.trailer.get("/Root")
        root_dict = root_obj.get_object() if root_obj else None
        if isinstance(root_dict, DictionaryObject):
            xmp = root_dict.get("/Metadata")
            if xmp is not None:
                writer._root_object[NameObject("/Metadata")] = xmp
    except Exception:
        pass

    # ---- Safe in-place overwrite ----
    dir_name = os.path.dirname(pdf_path)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "wb") as f:
            writer.write(f)
        os.replace(tmp_path, pdf_path)  # atomic on most platforms
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _collect_pdfs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in root.rglob("*.pdf") if path.is_file())
    return sorted(path for path in root.glob("*.pdf") if path.is_file())


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = ctx.target_path
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError("Select a directory before running this script.")

    scale = 0.90  # default scale factor for content resizing
    top_space_pts = 50.0  # default top space in points

    response = api.request_input(
        "PDF Header Adjustment: Top Space Amount (points)",
        id="ferp_pdf_header_options",
        default=str(top_space_pts),
        fields=[
            {
                "id": "recursive",
                "type": "bool",
                "label": "Scan subdirectories",
                "default": False,
            },
        ],
    )
    response = response.strip() if response else ""

    payload: dict[str, object]
    try:
        payload = json.loads(response) if response else {}
    except json.JSONDecodeError:
        payload = {"value": response}

    top_space_response = str(payload.get("value", "")).strip()
    if top_space_response:
        try:
            top_space_pts = float(top_space_response)
        except ValueError as exc:
            raise ValueError("Top space must be a number.") from exc
        if top_space_pts < 0:
            raise ValueError("Top space must be zero or greater.")

    recursive = bool(payload.get("recursive", False))
    pdf_files = _collect_pdfs(target_dir, recursive=recursive)
    total_files = len(pdf_files)
    api.log(
        "info",
        f"PDFs found={total_files} | recursive={recursive}",
    )

    for index, pdf_path in enumerate(pdf_files, start=1):
        api.progress(current=index, total=total_files or 1, unit="files")
        _make_top_space_first_page_inplace(
            pdf_path,
            top_space_pts=top_space_pts,
            scale=scale,
        )


if __name__ == "__main__":
    main()
