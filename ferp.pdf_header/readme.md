# Process: PDF Header

Adds top margin to the first page of each PDF by scaling and shifting the
content to create room for a header.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Prompts for the top-space size (points) and whether to scan subdirectories.
- Emits per-file progress and an info log of the total PDFs found.

## Usage

1. Select the directory containing the PDFs you want to adjust.
2. Run **Process: PDF Header** from the Scripts panel.
3. Enter the top-space amount in points (default 50).
4. Toggle subdirectory scanning if needed.

## Behavior

- Scales and shifts the first page content to create top space.
- Keeps the first-page content horizontally centered after scaling.
- Uses a fixed scale factor of `0.95`.
- Preserves the rest of the document pages unchanged.
- Overwrites the original files in place using a safe temp-file swap.

## Notes

- The top-space value must be a number and cannot be negative.
- Page-level annotations, links, and form fields on the first page are not preserved.
- Consider keeping a backup of the originals before running in bulk.
