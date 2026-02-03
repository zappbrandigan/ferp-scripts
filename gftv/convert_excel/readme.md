# Convert Excel to PDF

Converts Excel workbooks to PDF using the local Excel application.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for options unless arguments are preconfigured.
- Emits per-file progress and logs failures without stopping the batch.

## Usage

1. Navigate to the directory containing Excel files (or select a single file).
2. Run **Convert: Excel to PDF** from the Scripts panel.
3. Confirm options when prompted:
   - `recursive` to scan subdirectories.
   - `test` to process only the first file.

## Behavior

- Finds `*.xls*` files in the target directory (optionally recursive).
- Exports each workbook as a PDF next to the source file.
- Adds a numeric suffix if a PDF with the same name already exists.
- Reports failures while continuing the batch.

## Notes

- Requires Windows with Excel installed (uses `win32com`).
- On non-Windows systems the script runs in dry mode and reports what it would do.
