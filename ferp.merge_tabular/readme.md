# Merge Tabular Data

Combines CSV and XLSX files in the current working directory (shown in the app
title bar) when they share the same column headers.

---

## FERP Integration

- Operates on the `current directory` (the current working path shown in the app
  title bar).
- Scans only the top-level files (no subdirectories).
- Produces one merged CSV per matching header group without touching the source
  files.

## Usage

1. Navigate to a directory that contains the tabular files you want to merge
   (this becomes the current working path shown in the app title bar).
2. Run `Merge Tabular Data` from the Scripts panel.
3. Review the output panel summary for merged files, skipped files, and schemas
   that had no partner.

## Behavior

- Supports `.csv` and `.xlsx` files (first worksheet only).
- Files are grouped strictly by identical header names and order; mismatched
  schemas are not mixed.
- Each group creates `merged_<signature>.csv` (or a numbered variant if the file
  already exists), preserving column order and cell contents.
- Files that cannot be parsed are skipped with an explanation in the summary.

## Notes

- The script never modifies or deletes source files.
- Large datasets may take a moment; progress and results appear in FERP's output
  panel.
