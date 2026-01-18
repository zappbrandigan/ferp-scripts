from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from ferp.fscp.scripts import sdk


def _norm(v: object) -> object:
    """Normalize values so trivial differences don't create noise."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return v


def _sheet_cells(ws_values, ws_source) -> dict[tuple[int, int], object]:
    """
    Return dict mapping (row, col) -> normalized value
    limited to ws.calculate_dimension() (used range).
    """
    cells = {}
    max_row = max(ws_values.max_row or 0, ws_source.max_row or 0)
    max_col = max(ws_values.max_column or 0, ws_source.max_column or 0)
    if max_row == 0 or max_col == 0:
        return cells
    for row_source in ws_source.iter_rows(
        min_row=1, max_row=max_row, min_col=1, max_col=max_col
    ):
        for cell_source in row_source:
            cell_values = ws_values.cell(row=cell_source.row, column=cell_source.column)
            value = cell_values.value
            if value is None and cell_source.value is not None:
                value = cell_source.value
            cells[(cell_source.row, cell_source.column)] = _norm(value)
    return cells


def compare_excel(
    a_path: Path, b_path: Path, sheets: list[str] | None = None
) -> tuple[list[tuple[str, int, int, object, object]], list[str], bool]:
    wb_a_values = load_workbook(a_path, data_only=True)
    wb_b_values = load_workbook(b_path, data_only=True)
    wb_a_source = load_workbook(a_path, data_only=False)
    wb_b_source = load_workbook(b_path, data_only=False)
    missing: list[str] = []
    compared_by_index = False
    try:
        pairs: list[tuple[str, str, str]] = []
        if sheets is None:
            common = sorted(set(wb_a_values.sheetnames) & set(wb_b_values.sheetnames))
            if common:
                pairs = [(name, name, name) for name in common]
            else:
                compared_by_index = True
                for a_name, b_name in zip(
                    wb_a_values.sheetnames, wb_b_values.sheetnames, strict=False
                ):
                    label = f"{a_name} vs {b_name}"
                    pairs.append((a_name, b_name, label))
        else:
            pairs = [(name, name, name) for name in sheets]
        diffs: list[tuple[str, int, int, object, object]] = []

        for a_name, b_name, label in pairs:
            if (
                a_name not in wb_a_values.sheetnames
                or b_name not in wb_b_values.sheetnames
            ):
                missing.append(label)
                continue

            ws_a_values = wb_a_values[a_name]
            ws_b_values = wb_b_values[b_name]
            ws_a_source = wb_a_source[a_name]
            ws_b_source = wb_b_source[b_name]

            a = _sheet_cells(ws_a_values, ws_a_source)
            b = _sheet_cells(ws_b_values, ws_b_source)

            coords = sorted(set(a.keys()) | set(b.keys()))

            for r, c in coords:
                va = a.get((r, c), "")
                vb = b.get((r, c), "")
                if va != vb:
                    diffs.append((label, r, c, va, vb))
    finally:
        wb_a_values.close()
        wb_b_values.close()
        wb_a_source.close()
        wb_b_source.close()

    return diffs, missing, compared_by_index


def _write_csv(
    out_csv_path: Path, diffs: list[tuple[str, int, int, object, object]]
) -> None:
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sheet", "row", "col", "a_value", "b_value"])
        writer.writerows(diffs)


def _format_diffs(diffs: list[tuple[str, int, int, object, object]]) -> str:
    max_display = 100
    lines: list[str] = []
    for index, diff in enumerate(diffs):
        if index >= max_display:
            lines.append(
                f"(truncated to {max_display} differences; export CSV for full data)"
            )
            break
        sheet, row, col, a_value, b_value = diff
        col_letter = get_column_letter(col)
        lines.append(f"{sheet}!{col_letter}{row}: {a_value} -> {b_value}")
    return "\n".join(lines).rstrip()


def _sample_cells(path: Path, max_cells: int = 5) -> list[tuple[str, str, object]]:
    wb = load_workbook(path, data_only=True)
    samples: list[tuple[str, str, object]] = []
    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            max_row = ws.max_row or 0
            max_col = ws.max_column or 0
            if max_row == 0 or max_col == 0:
                continue
            for row in ws.iter_rows(
                min_row=1, max_row=max_row, min_col=1, max_col=max_col
            ):
                for cell in row:
                    if cell.column is None:
                        continue
                    value = cell.value
                    if value is None or (isinstance(value, str) and not value.strip()):
                        continue
                    samples.append(
                        (
                            sheet_name,
                            f"{get_column_letter(int(cell.column))}{cell.row}",
                            value,
                        )
                    )
                    if len(samples) >= max_cells:
                        return samples
    finally:
        wb.close()
    return samples


def _ensure_xlsx(path: Path, label: str) -> None:
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"{label} must be a .xlsx file: {path}")


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    target_dir = ctx.target_path
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Target '{target_dir}' is not a directory.")

    file_a = api.request_input(
        "First Excel file (relative to target directory)",
        id="excel_diff_file_a",
    ).strip()
    file_b = api.request_input(
        "Second Excel file (relative to target directory)",
        id="excel_diff_file_b",
    ).strip()
    sheets_raw = api.request_input(
        "Sheets to compare (comma-separated, optional)",
        id="excel_diff_sheets",
        default="",
    ).strip()

    if not file_a or not file_b:
        raise ValueError("Both file_a and file_b are required.")

    a_path = Path(file_a)
    b_path = Path(file_b)
    if not a_path.is_absolute():
        a_path = target_dir / a_path
    if not b_path.is_absolute():
        b_path = target_dir / b_path

    if not a_path.exists():
        raise FileNotFoundError(a_path)
    if not b_path.exists():
        raise FileNotFoundError(b_path)
    _ensure_xlsx(a_path, "file_a")
    _ensure_xlsx(b_path, "file_b")

    sheets = [s.strip() for s in sheets_raw.split(",") if s.strip()] or None

    for label, path in (("file_a", a_path), ("file_b", b_path)):
        samples = _sample_cells(path)
        if samples:
            api.log("info", f"{label} sample cells: {samples}")
        else:
            api.log("warn", f"{label} has no non-empty cells detected")

    api.log(
        "info",
        f"Comparing '{a_path.name}' vs '{b_path.name}' | sheets={sheets or 'auto'}",
    )

    diffs, missing, compared_by_index = compare_excel(a_path, b_path, sheets=sheets)

    csv_path: Path | None = None
    output_dir = a_path.parent
    if a_path.parent != b_path.parent:
        api.log(
            "warn",
            (
                "Input files are in different directories; writing CSV next to "
                f"the first file: {a_path.parent}"
            ),
        )
    csv_path = output_dir / f"{a_path.stem}_vs_{b_path.stem}_excel_diff.csv"
    _write_csv(csv_path, diffs)

    if missing:
        api.log("warn", f"Skipped missing sheets: {', '.join(missing)}")
    if compared_by_index:
        api.log(
            "warn",
            (
                "No matching sheet names found; compared sheets by index order instead. "
                "Provide explicit sheet names to override."
            ),
        )

    api.emit_result(
        {
            "diffs_found": len(diffs),
            "missing_sheets": missing,
            "compared_by_index": compared_by_index,
            "csv_path": str(csv_path) if csv_path else None,
            "results": _format_diffs(diffs),
        }
    )


if __name__ == "__main__":
    main()
