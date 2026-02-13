# Rename UVS

Renames UVS PDFs in the highlighted directory using the production title
from the filename plus the episode number or air date found on page 1.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Renames every `.pdf` file inside that directory; other files are ignored.
- Streams progress and result summaries to the output panel.

## Usage

1. Highlight the folder containing the UVS TEL PDFs you want to rename.
2. Run `Rename UVS Tel` from the Scripts panel.
3. Wait for the script to finish; refreshed filenames will appear in the tree.

## Behavior

- Derives the production title from the filename pattern `<title> - ...`.
- Uses a combination of `Air Date` and `Episode Number`:
  - Uses `Air Date` or `Air/Release Date` from cue sheet contents
  - `Air Date` accepts `MM/DD/YY` or `MM/DD/YYYY` and is formatted as `DDMMYYYY`
  - Adds episode number as suffix when present, distinct from air date, and is 5 characters or less
  - Removes special characters form episode number
- Produces `PRODUCTION   Ep No. <air_date>.pdf` or `PRODUCTION   Ep No. <air_date>-<episode_number>.pdf` 
- Moves files with no extractable text into `_needs_ocr/`.
- When a name collision occurs, moves both the incoming file and the existing target file into `_check/`.
- Moves any other skipped files into `_check/`.

## Notes

- If files in `_check/` include `_02`, `_03`, etc., the file may be a duplicate or there was insufficient information to rename automatically.
- The selected directory must be accessible; the script aborts if it cannot read or write in place.
