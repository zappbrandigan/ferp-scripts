# Convert VIA to PDF

Converts a VIA Excel workbook into per-sheet PDFs with standard cleanup and
page setup.

---

## FERP Integration

- Operates on a `highlighted file` of type `.xlsx` in the File Navigator.
- Emits per-sheet progress and logs failures without stopping the batch.

## Usage

1. Select a VIA Excel file.
2. Run **Convert: Viacom to PDF** from the Scripts panel.
3. Confirm the Excel close prompt.

## Behavior

- Iterates through each worksheet in the workbook.
- Skips sheets where `D12` indicates `Show Type: Digital` or `Show Type: Podcast`.
- Cleans each sheet before export:
  - Deletes columns `L:N`.
  - Deletes column `I`.
  - Autofits column `E`.
  - Autofits all rows.
- Computes print area automatically and applies standard page setup.
- Exports each worksheet as a separate PDF named `via_{Sheet}`.
- Writes PDFs into a `via_converted/{group}` subfolder next to the source workbook.
  - The group is read from `P9` or `O9` (e.g. `Production Group: <name>`), normalized to lowercase with `-` separators.
  - Missing or empty `P9` values go to a `_unknown` folder.
- Adds XMP metadata with a newly generated `ferp:DocumentID` for each exported PDF.

## Notes

- Requires Windows with Excel installed (uses `win32com`).
- On non-Windows systems the script runs in dry mode and reports what it would do.
