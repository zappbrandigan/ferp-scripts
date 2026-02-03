# Rename FOX Emails

Renames exported FOX email PDFs so that only the last 12 characters of each
filename (typically the message identifier) are preserved.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Renames every `.pdf` file inside that directory; other files are ignored.
- Streams progress and a renamed-count summary to the output panel.

## Usage

1. Highlight the folder containing the FOX email PDFs you want to clean up.
2. Run `Rename FOX Emails` from the Scripts panel.
3. Wait for the script to finish; refreshed filenames will appear in the tree.

## Behavior

- Keeps only the final 12 characters of each filename's stem and re-attaches
  the `.pdf` suffix.
- Leaves PDFs that already match the target pattern unchanged.
- Appends numeric suffixes (`_01`, `_02`, …) when duplicates would otherwise
  collide.
- Logs per-file rename activity plus a final `{renamed, total}` summary.

## Notes

- The selected directory must be accessible; the script aborts if it cannot
  read or write in place.
- Large directories may take a moment; avoid closing FERP until the script
  reports completion.
