# Compare Excel Diff

Compares two Excel files and reports cell-level differences, with optional
CSV output.

---

## FERP Integration

- Operates on the `current directory` (the current working path shown in the app
  title bar).
- Prompts for the two Excel files to compare and optional sheet names.
- Emits a summary of differences and can write a CSV report.

## Usage

1. Navigate to the directory containing the Excel files in FERP (this becomes the current working path shown in
   the app title bar).
2. Run `Compare Excel Diff` from the Scripts panel.
3. Enter the two file names (relative to the current working directory).
4. Optionally enter comma-separated sheet names to compare.

## Behavior

- If sheet names are provided, only those sheets are compared.
- If no matching sheet names are found, the script compares sheets by index.
- Differences are reported as `Sheet!A1: old -> new`.
- CSV output is saved next to the first file as
  `<file_a>_vs_<file_b>_excel_diff.csv`.

## Notes

- Whitespace is trimmed for string values before comparison.
- If the two files live in different folders, the CSV is written next to
  the first file and a warning is logged.
