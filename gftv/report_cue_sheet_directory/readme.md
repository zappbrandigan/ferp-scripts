# Report Cue Sheet Directory

Creates an Excel summary of cue sheet PDFs stored in the expected GFTV folder layout.

---

## FERP Integration

- Operates on a `highlighted directory`.
- Prompts for a report type before processing.
- Scans PDF files recursively under the selected directory.
- Writes an `.xlsx` report next to the selected directory.

## Report Types

- `Monthly Report`
  Uses the current cue sheet summary workbook format.
- `Weekly Report`
  Uses the same PDF discovery and parsing flow, but writes a reduced column set for weekly reporting.

## Monthly Report Columns

- `Deal`
- `Catalog`
- `PD Code`
- `Film or Series`
- `Revision Status`
- `Production Title`
- `Episode Title`
- `Season Number`
- `Episode Number`
- `Territory Code`
- `Territory`

## Weekly Report Columns

- `Deal`
- `Catalog`
- `Film or Series`
- `Revision Status`
- `Cue Sheet`
- `Territory Code`
- `Territory`

## Expected Layout

The script recognizes cue sheet PDFs stored in either of these path shapes:

- `date_range/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/territory_codes/tv/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/tv/catalog_code/cue_sheets.pdf`

## Behavior

- Prompts the user to choose `Weekly Report` or `Monthly Report`.
- Finds all `.pdf` files under the selected directory.
- Parses each relative path to extract the report columns.
- Marks `_REV` paths as `Revision`; all other recognized paths are marked as `New`.
- Writes one row per recognized PDF to either `<selected-directory>_cue_sheet_monthly_report.xlsx` or `<selected-directory>_cue_sheet_weekly_report.xlsx` next to the selected directory.
- Logs and counts PDFs that do not match the expected folder structure.
