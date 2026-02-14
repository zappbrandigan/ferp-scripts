# Process: Stamp Cue Sheets

Processes cue sheet PDFs in the highlighted directory, detects supported
formats, and tags matched publishers with XMP metadata and a stamped badge.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlighted file` of type `.pdf` in the File Navigator.
- Prompts for a category code and optional recursive scan.
- Logs per-file processing details and emits a final summary.

## Usage

1. Highlight a directory containing cue sheet PDFs in FERP.
2. Run `Process: Stamp Cue Sheets` from the Scripts panel.
3. Enter a category code. 
4. (Optional) Select `Overwrite` to update the original files instead of creating copies.
5. (Optional) Select `Add header` to add blank space at the top of the PDF.
6. (Optional) Select `Recursive` if you want to process subdirectories.
7. (Optional) Toggle `Select publishers` to bypass parsing and manually choose publishers for the stamp.
8. (Optional) Select `Custom stamp` to enter custom text for the publisher names, territory, and start date.
9. Review logs and the output folders.

> For co-pub situations, enter a comma separated list of catalog codes. The first catalog code will be considered
> `primary` publishers and the rest will be considered the `co-`publishers.

## Behavior

- Detects Soundmouse, RapidCue, Cuetrak, WB or unknown formats and parses cue data.
- If the first page has no extractable text, the PDF is moved to `_needs_ocr`.
- Matches controlled publishers from the cached category list.
- For non-default parsers, co-publishers are only matched when they appear in the
  same cue as a matched main publisher. Co-publishers are ignored if no main
  publisher is found in that cue.
- If controlled publisher has "split" territories, a territory selection prompt will appear.
- Adds XMP metadata for matched publishers and stamps the first page. (It will remove any existing stamps before adding a new stamp.)
- By default, writes stamped PDFs to an `_stamped` folder next to the source file.
- Optional `Overwrite` updates the original files instead of writing to `_stamped`.
- Moves unmatched PDFs into a `_nop` folder.
- Moves mixed-territory PDFs into `_error`.

## Parser Differences

Different cue sheet formats yield different parsing granularity. This affects how
publishers are matched and how co-publishers are detected.

- Soundmouse, RapidCue, WB, Cuetrak: These parsers operate on per-cue rows, so
  main/co-publisher matching happens within the same cue. Co-publishers are only
  considered if a main publisher is matched in that cue.
- Default/unknown: This parser does not have a per-cue notion. It scans the full
  document text to find publisher names and cannot apply per-cue co-publisher
  logic.

## Notes

- Requires a publisher cache file.
- When overwriting is disabled (default), the original PDF remains unchanged and
  metadata is only written to the stamped copy.

## TODO
- Extend support for master/sync license detection to more cue sheet formats. (Currently only supported in Soundmouse formats.)
- Add custom parsers for Soundmouse (landscape orientation) and Silvermouse