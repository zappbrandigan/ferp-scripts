# Flatten Directory

Moves all files from nested subdirectories into the highlighted folder and
removes any empty directories afterward.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- Streams progress to the output panel as items are processed.

## Usage

1. Highlight the directory you want to flatten.
2. Run `Flatten Directory` from the Scripts panel.
3. Wait for the summary result showing files moved and directories removed.

## Behavior

- Recursively moves every file into the root folder.
- Appends suffixes (`_1`, `_2`, …) when filename collisions occur.
- Deletes empty directories once their contents have been relocated.

## Notes

- Existing root-level files are never overwritten.
- Large folder trees may take time; avoid closing FERP until the script finishes.
