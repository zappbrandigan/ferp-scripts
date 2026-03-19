# Report Cue Sheet Directory

Creates an Excel summary of cue sheet PDFs stored in the expected GFTV folder layout.

---

## FERP Integration

- Operates on a `highlighted directory`.
- Scans PDF files recursively under the selected directory.
- Writes an `.xlsx` report next to the selected directory.

## Expected Layout

The script recognizes cue sheet PDFs stored in either of these path shapes:

- `date_range/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/territory_codes/tv/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/tv/catalog_code/cue_sheets.pdf`

## Report Columns

- `Date Range`
- `Territory Code`
- `Revision Status`
- `Film or TV`
- `Catalog Code`
- `Cue Sheet PDF`

## Behavior

- Finds all `.pdf` files under the selected directory.
- Parses each relative path to extract the report columns.
- Marks `_REV` paths as `Revision`; all other recognized paths are marked as `New`.
- Writes one row per recognized PDF to `<selected-directory>_cue_sheet_summary.xlsx` next to the selected directory.
- Logs and counts PDFs that do not match the expected folder structure.
