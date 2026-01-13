# Restore Backup

Restores a highlighted `.backup.zip` archive into a sibling directory that
matches the original folder name.

---

## FERP Integration

- Operates on the `highlighted file`; the file must end with `.backup.zip`.
- The output directory lives next to the backup archive.
- Warns before overwriting existing files in the destination directory.

## Usage

1. Highlight a `.backup.zip` file in FERP.
2. Run `Restore Backup` from the Scripts panel.
3. Confirm/cancel overwrites if prompted.
4. The script creates `<archive_base>/` next to the backup and restores contents there.

## Behavior

- Recreates the original folder structure inside the output directory.
- Removes an existing destination directory when overwriting is confirmed.
- Leaves the `.backup.zip` file untouched.

## Notes

- Corrupt or encrypted archives stop the script and surface an error message in FERP.
- Large backups may take several seconds; progress updates show in FERP's output panel.
