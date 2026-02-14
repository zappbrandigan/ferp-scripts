# Query PDF

Searches PDFs for one or more text queries (with optional inline regex),
producing an XLSX summary of all matches.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlighted file` of type `.pdf` in the File Navigator.
- Prompts for the search query and options.
- Emits per-file progress and logs whenever a file is skipped or fails.

## Usage

1. Navigate into the directory tree you want to search (or select a single PDF).
2. Run **Query PDF** from the Scripts panel.
3. Provide one or more search terms (one per line). Wrap a line in `/.../` to
   treat it as regex.
4. Optional settings (configure in the prompt):
   - Recursive to scan directories sub folders.
   - Case sensitive for case-sensitive matches.
   - Context chars to adjust the context snippet (default 80).

## Behavior

- Scans `*.pdf` files in the selected directory (optionally recursive).
- Queries the selected PDF when a single file is selected.
- Extracts plain text (skipping encrypted/unreadable PDFs with a warning).
- Records every match with file name, relative path, query, page number, match text,
  and surrounding context.
- Writes deterministic XLSX results (including a Summary sheet) to the directory
  containing the target, e.g. `<dir_name>_query_results.xlsx` or
  `<file_name>_query_results.xlsx`.

## Notes

- Scanned PDFs without text won’t yield results unless OCR is performed beforehand.
- Invalid regex patterns abort the run with an error message.
- Extraction failures or individual file issues are logged but do not abort the
  entire search.
