# Extract PDF Metadata

Collects metadata tags from PDF files in the highlighted directory, with
optional recursive scanning and CSV output.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for options to enable recursion and CSV output.
- Streams per-file metadata to the output panel.

## Usage

1. Highlight a directory containing PDFs in FERP.
2. Run `Extract PDF Metadata` from the Scripts panel.
3. Toggle `Scan subdirectories` or `Write CSV summary` as needed.
4. Review metadata in the output panel and optional CSV file.

## Behavior

- Default scan is non-recursive; only PDFs in the highlighted folder are read.
- Logs metadata tags per file (or notes when none are present).
- Writes a CSV summary to the parent directory as
  `<dir_name>_pdf_metadata.csv` when enabled.

## Notes

- Encrypted or unreadable PDFs are skipped with a warning.
- Some PDFs may not include metadata tags; those files still appear in output.
