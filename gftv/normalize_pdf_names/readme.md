# Normalize PDF Names

Validates and normalizes PDF filenames in the highlighted directory so they conform to the studio naming convention.

---

## FERP Integration

- Runs against the `highlighted directory` by default (no recursion).
- Optional recursive mode prompts on launch.
- Emits structured logs and result summaries to the output panel.
- Moves any ambiguous files into `_check/` inside the target directory (or
  each subdirectory when running recursively).

## Usage

1. Highlight the directory that contains the PDFs you want to review.
2. Run `Normalize PDF Names` from the Scripts list.
3. Monitor the output panel for the summary of valid, renamed, and `_check/` files.

## Behavior

- Supports three shapes: `PRODUCTION`, `PRODUCTION   EPISODE INFO`, and `PRODUCTION   Episode Title  EPISODE INFO`.
- Repairs double/incorrect delimiters when a deterministic fix is possible.
- Repositions leading articles per language detection (English, Spanish, French, etc.).
- Forces casing: production titles → ALL CAPS, episode titles → Title Case, episode info normalized.
- Normalizes episode tokens (case/spacing of `Ep No.`; supports `101A - 101B`; normalizes `version`/`vrsn` to `Vrsn`).
- Uses ASCII-only filenames (accents removed; smart punctuation normalized).
- Ensures names stay ≤ 60 chars by truncating episode title first, then production title; trailing `- _ , . ' ` and whitespace are trimmed before appending `. . .`.
- Uses safe renaming with numeric suffixes to avoid overwrites; any collision moves both the existing and incoming files to `_check/`.

## Notes

- Files already in compliance are reported but untouched.
- Broken symlinks, permission errors, or unparsable names are routed to `_check/` with a reason tag in the log file.
