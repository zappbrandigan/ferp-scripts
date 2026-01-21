# Summarize Directory Tree

Counts files per extension for each directory (up to four levels deep) beneath
the selected folder.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Walks the directory tree up to four levels below the root.
- Ignores hidden files/folders and never modifies the filesystem.

## Usage

1. Highlight the directory you want to analyze in FERP.
2. Run `Summarize Directory Tree` from the Scripts panel.
3. Confirm whether you'd like to export a Markdown summary.
4. Review the output panel for per-directory totals and extension breakdowns.

## Behavior

- Groups only the files directly inside each directory (child folders are counted
  separately).
- Reports one entry per directory showing total files and counts per extension.
- Skips unreadable paths (permission errors, broken symlinks) without aborting.

## Notes

- The script can optionally export a Markdown summary next to the target folder.
- Large trees will take slightly longer but remain bounded by the depth limit.
