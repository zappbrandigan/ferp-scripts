# Compare PDF Text Diff

Compares the extracted text of two PDF files and produces a unified diff
summary saved to disk (with a truncated snippet shown in FERP).

---

## FERP Integration

- Operates on the `current directory` (the current working path shown in the app
  title bar).
- Prompts for the two PDF files to compare.
- Emits a diff summary snippet in the output panel and writes the full diff
  report to disk.

## Usage

1. Navigate to the directory containing the PDF files in FERP (this becomes the current working path shown in the app title bar).
2. Run `Compare PDF Text Diff` from the Scripts panel.
3. Enter the two file names (relative to the current directory).

## Behavior

- Extracts text per page with page markers for context.
- Computes a unified diff and saves it as
  `<file_a>_vs_<file_b>_pdf_text_diff.txt` next to the first file.
- Logs a warning if the files are in different directories and writes output
  next to the first file.
- Displays a truncated diff snippet in the output panel to keep results readable.

## Notes

- Encrypted PDFs or extraction failures are logged and will raise errors.
- PDFs that contain only images may yield empty text unless OCR is applied first.
