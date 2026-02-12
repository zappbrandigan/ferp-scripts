# Rename Fox Sports

Renames Fox Sports cue sheet PDFs in the highlighted directory using the program
title, episode title, air date, and episode number extracted from page 1.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlightd file` of type `.pdf` in the File Navigator.
- Renames every `.pdf` file inside that directory; other files are ignored.
- Streams progress and result summaries to the output panel.

## Usage

1. Highlight PDF file or a folder containing the relevant cue sheet PDFs.
2. Run `Rename: Fox Sports` from the Scripts panel.
3. Wait for the script to finish; refreshed filenames will appear in the tree.

## Behavior

- Extracts `Initial Air Date`, `Program Title`, `Episode Title`, and `Episode Number`
  from the first page of the cue sheet.
- Accepts several air date formats (e.g., `December 2, 2025`, `12/2/2025`,
  `2025-12-02`) and formats the date as `DDMMYYYY`.
- Produces `PROGRAM   EPISODE  Ep No. DDMMYYYY-<episode_number>.pdf` when
  the episode title differs from the program title.
- Produces `PROGRAM   Ep No. DDMMYYYY-<episode_number>.pdf` when the episode
  title matches the program title.
- Moves files with no extractable text into `_needs_ocr/`.
- When a name collision occurs, moves both the incoming file and the existing
  target file into `_check/`.
- Moves any other skipped files into `_check/`.

## Notes

- Episode numbers may be alphanumeric; the script reads the first token after
  `Episode Number:` and ignores the following labels.
- If files in `_check/` include `_02`, `_03`, etc., the file may be a duplicate or
  there was insufficient information to rename automatically.
