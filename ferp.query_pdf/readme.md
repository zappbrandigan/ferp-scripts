# Query PDF

Searches every PDF beneath the highlighted directory for a text pattern or
regular expression, producing a CSV summary of all matches.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for the search query unless arguments are preconfigured.
- Emits per-file progress and logs whenever a file is skipped or fails.

## Usage

1. Navigate into the directory tree you want to search.
2. Run **Query PDF** from the Scripts panel.
3. Provide a search term (literal or regex).
4. Optional flags (configure via `config.json` or bundle args):
   - `--regex` to interpret the query as a regular expression.
   - `--case-sensitive` for case-sensitive matches.
   - `--context-chars N` to adjust the context snippet (default 80).

## Behavior

- Recursively scans `*.pdf` files.
- Extracts plain text (skipping encrypted/unreadable PDFs with a warning).
- Records every match with file name, relative path, page number, match text,
  and surrounding context.
- Writes deterministic CSV results to the parent directory of the search root,
  e.g. `../project_query_results.csv`.

## Notes

- Uses PyPDF2’s text extraction; scanned PDFs without text won’t yield results
  unless OCR is performed beforehand.
- Regex errors, extraction failures, or individual file issues are logged but do
  not abort the entire search.
