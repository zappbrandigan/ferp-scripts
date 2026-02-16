# Organize: Group Cue Sheets

Groups PDF cue sheets into subfolders based on either the production title in
the filename or the publisher names found in the FERP XMP stamp metadata.

---

## FERP Integration

- Operates on a `highlighted directory` in the File Navigator.
- Groups only the PDFs in the selected directory (non-recursive).
- Streams progress and summaries to the output panel.

## Usage

1. Highlight a PDF file or folder containing cue sheet PDFs.
2. Run `Organize: Group Cue Sheets` from the Scripts panel.
3. Choose whether to group by `production` or `publishers`.

## Behavior

- **Production mode**
  - Uses the filename segment before the first three-space delimiter (`"   "`).
  - Lowercases the folder name and replaces spaces with `-`.
  - Skips files without the delimiter and logs a warning.
- **Publishers mode**
  - Reads FERP XMP metadata (`ferp:agreements -> publishers`).
  - Lowercases each publisher, replaces spaces with `-`, and joins publishers with `_`.
  - Skips PDFs without publisher metadata or encrypted PDFs and logs a warning.
- Folder names are truncated to 60 characters with `...` if needed.

## Notes

- Files are moved into the new subfolders in the selected directory.
- Name collisions are resolved by appending a numeric suffix to the filename.
