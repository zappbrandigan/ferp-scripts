# Normalize PDF Names

Validates and normalizes PDF filenames in the highlighted directory so they conform to the studio naming convention.

---

## FERP Integration

- Runs against the `highlighted directory` only (no recursion).
- Emits structured logs and result summaries to the output panel.
- Moves any ambiguous files into `_check/` inside the target directory.

## Usage

1. Highlight the directory that contains the PDFs you want to review.
2. Run `Normalize PDF Names` from the Scripts list.
3. Monitor the output panel for the summary of valid, renamed, and `_check/` files.

## Behavior

- Supports three shapes: `PRODUCTION`, `PRODUCTION   EPISODE INFO`, and `PRODUCTION   Episode Title  EPISODE INFO`.
- Repairs double/incorrect delimiters when a deterministic fix is possible.
- Repositions leading articles per language detection (English, Spanish, French, etc.).
- Forces casing: production titles → ALL CAPS, episode titles → Title Case, episode info unchanged.
- Ensures names stay ≤ 60 chars by truncating episode title first, then production title; trailing punctuation is trimmed before appending `...`.
- Uses safe renaming with numeric suffixes to avoid overwrites; any collision goes to `_check/`.

## Notes

- Files already in compliance are reported but untouched.
- Broken symlinks, permission errors, or unparsable names are routed to `_check/` with a reason tag.
