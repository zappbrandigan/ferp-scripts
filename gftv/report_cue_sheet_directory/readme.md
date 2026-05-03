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
- `Statistical Report`
  Combines the cue sheet folder structure with FERP XMP metadata embedded in the PDFs and writes a multi-sheet workbook for business and QA reporting.

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

## Statistical Report Sheets

- `Summary`
  High-level totals for recognized vs skipped PDFs, revisions, metadata completeness, publishers, catalogs, and date ranges.
- `Cue Sheets`
  One row per recognized PDF, combining path-derived fields with FERP XMP fields such as catalog code, stamp date, document IDs, publishers, territories, and metadata health flags.
- `Catalog Insights`
  Aggregates cue sheet volume, revisions, film vs series mix, production counts, and publisher counts by path catalog and XMP catalog code.
- `Publisher Insights`
  Aggregates cue sheet volume, revisions, and production counts by publisher.
- `Quality Checks`
  Lists skipped layouts, missing XMP fields, metadata read failures, catalog mismatches, duplicate document IDs, and invalid dates.

## Expected Layout

The script recognizes cue sheet PDFs stored in either of these path shapes:

- `date_range/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/territory_codes/tv/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/film/catalog_code/cue_sheets.pdf`
- `date_range/_REV/territory_codes/tv/catalog_code/cue_sheets.pdf`

## Behavior

- Prompts the user to choose `Weekly Report`, `Monthly Report`, or `Statistical Report`.
- Finds all `.pdf` files under the selected directory.
- Parses each relative path to extract the report columns.
- Marks `_REV` paths as `Revision`; all other recognized paths are marked as `New`.
- For `Statistical Report`, reads FERP XMP metadata from each recognized PDF and builds multiple workbook sheets for analysis.
- Writes one `.xlsx` workbook next to the selected directory using the suffix matching the selected report type.
- Logs and counts PDFs that do not match the expected folder structure.
