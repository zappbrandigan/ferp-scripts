# Process: Stamp Cue Sheets

Processes cue sheet PDFs in the highlighted directory, detects supported
formats, and tags matched publishers with XMP metadata and a stamped badge.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for a category code and optional recursive scan.
- Logs per-file processing details and emits a final summary.

## Usage

1. Highlight a directory containing cue sheet PDFs in FERP.
2. Run `Process: Stamp Cue Sheets` from the Scripts panel.
3. Enter a category code (for example, `uvs`).
4. (Optional) Select `Add header` to add blank space at the top of the PDF.
5. (Optional) Select `Recursive` if you want to process subdirectories.
6. (Optional) Toggle `Select publishers` to bypass parsing and manually choose publishers for the stamp.
7. Review logs and the output folders.

## Behavior

- Detects Soundmouse, RapidCue, or unknown formats and parses cue data.
- Matches controlled publishers from the cached category list.
- Adds XMP metadata for matched publishers and stamps the first page.
- By default, writes stamped PDFs to an `_stamped` folder next to the source file.
- Optional `Overwrite` updates the original files instead of writing to `_stamped`.
- Moves unmatched PDFs into a `_nop` folder.

## Notes

- Requires a publisher cache file.
- When overwriting is disabled (default), the original PDF remains unchanged and
  metadata is only written to the stamped copy.
