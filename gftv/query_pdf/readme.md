# Query PDF

Searches PDFs for a text pattern or regular expression, producing a CSV summary
of all matches.

---

## FERP Integration

- Operates on a highlighted directory or a single `.pdf` file in the File Navigator.
- Prompts for the search query and options.
- Emits per-file progress and logs whenever a file is skipped or fails.

## Usage

1. Navigate into the directory tree you want to search (or select a single PDF).
2. Run **Query PDF** from the Scripts panel.
3. Provide a search term (literal or regex).
4. Optional settings (configure in the prompt):
   - Use regex to interpret the query as a regular expression.
   - Case sensitive for case-sensitive matches.
   - Context chars to adjust the context snippet (default 80).

## Behavior

- Recursively scans `*.pdf` files when a directory is selected.
- Queries the selected PDF when a single file is selected.
- Extracts plain text (skipping encrypted/unreadable PDFs with a warning).
- Records every match with file name, relative path, page number, match text,
  and surrounding context.
- Writes deterministic CSV results to the same directory as the target file/directory,
  e.g. `<dir_name>_query_results.csv` or `<file_name>_query_results.csv`.

## Notes

- Scanned PDFs without text won’t yield results unless OCR is performed beforehand.
- Regex errors, extraction failures, or individual file issues are logged but do
  not abort the entire search.
