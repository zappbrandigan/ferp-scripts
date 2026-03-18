# Convert Excel to PDF

Converts Excel workbooks to PDF using the local Excel application.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlighted file` of type `.xls*` in the File Navigator.
- Emits per-file progress and logs failures without stopping the batch.

## Usage

1. Navigate to the directory containing Excel files (or select a single file).
2. Run **Convert: Excel to PDF** from the Scripts panel.
3. Confirm options when prompted:
   - `Recursive` to scan subdirectories.
   - `Portrait orientation` to switch layout from the default landscape orientation.
   - `Autofit columns` to autofit column widths (off by default).
   - `Autofit rows` to autofit row heights (off by default).
   - `Convert all tabs` to export one PDF per worksheet in the workbook.
   - Optional text input to target a specific sheet by name or 1-based index when `Convert all tabs` is off.

## Behavior

- Finds `*.xls*` files in the target directory (optionally recursive).
- Exports the active sheet by default, or a specific sheet when provided in the text input.
- When `Convert all tabs` is enabled, exports every worksheet as its own PDF named `<workbook> - <sheet>.pdf`.
- Adds a numeric suffix if a PDF with the same name already exists.
- Reports failures while continuing the batch.

## Notes

- Requires Windows with Excel installed (uses `win32com`).
- On non-Windows systems the script runs in dry mode and reports what it would do.
