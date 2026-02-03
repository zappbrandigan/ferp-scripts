# Convert Moonbug

Converts Moonbug Excel workbooks to PDF by applying a standard page setup and
naming output files from specific worksheet cells.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for options unless arguments are preconfigured.
- Emits per-file progress and logs failures without stopping the batch.

## Usage

1. Navigate to the directory containing Moonbug Excel files.
2. Run **Convert Moonbug** from the Scripts panel.
3. Confirm options when prompted:
   - `autofitcolumn` (default true) to auto-size columns before export.
   - `test` to process only the first file.

## Behavior

- Recursively scans `*.xls*` files.
- Applies print setup (landscape, fit-to-page width, computed print area).
- Builds PDF file names from worksheet cells:
  - `I4` for PD title (uppercased).
  - `I6` for episode title (title-cased).
  - `D15` for episode number/version.
- Writes PDFs into the target directory, appending a counter if a name exists.
- Prompts to confirm Excel files are closed and suggests running in test mode.
- If a naming collision occurs, moves related PDFs and Excel files to `_check` in the same directory.
- Moves remaining Excel files into `_og` within their original directories after conversion.
- If a filename includes `None Vrsn`, moves related PDFs and Excel files to `_none_vrsn` in the same directory.

## Notes

- Requires Windows with Excel installed (uses `win32com`).
- On non-Windows systems the script runs in dry mode and reports what it would do.
- Empty episode numbers default to `None Vrsn` in the output filename.
