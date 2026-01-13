# Backup Directory

Creates a `.backup.zip` archive of the highlighted directory alongside the
original folder.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- The output backup lives next to the directory.
- Warns before overwriting an existing backup archive with the same name.

## Usage

1. Highlight a directory in FERP.
2. Run `Backup Directory` from the Scripts panel.
3. Confirm/cancel overwrites if prompted.
4. The script creates `<dir_name>.backup.zip` next to the original directory.

## Behavior

- Recursively adds every file/folder under the target.
- Preserves the directory hierarchy inside the archive.
- Leaves the original directory untouched.

## Notes

- Hidden files are included.
- Large directories may take several seconds; progress updates show in FERP's
  output panel.
