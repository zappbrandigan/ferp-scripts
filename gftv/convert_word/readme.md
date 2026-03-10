# Convert Word to PDF

Converts Word `.doc` and `.docx` files to `.pdf` and archives the original
files into an `_og/` folder after processing.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlighted file` of type `.doc` or `.docx` in the File Navigator.
- Converts every `.doc` and `.docx` file inside that directory; other files are ignored.
- Streams progress and result summaries to the output panel.

## Usage

1. Highlight a `.doc`/`.docx` file or a folder containing the Word files.
2. Run `Convert: Word to PDF` from the Scripts panel.

## Behavior

- Requires Windows with Microsoft Word installed.
- On non-Windows systems, runs in dry mode and reports totals only.
- Writes PDFs next to the source files and overwrites existing PDFs.
- Moves processed `.doc` and `.docx` files into `_og/`.

## Notes

- If Word cannot open a file or export fails, the file is logged under
  `Failures` in the summary.
