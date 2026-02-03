# Zip Directory

Compresses the highlighted directory into a `.zip` archive alongside the
original folder.

---

## FERP Integration

- Operates on the `highlighted directory` in the File Navigator.
- The output `.zip` lives next to the directory.
- Warns before overwriting an existing archive with the same name.

## Usage

1. Highlight a directory in FERP.
2. Run `Zip Directory` from the Scripts panel.
3. Confirm/cancel overwrites if prompted.
3. The script creates `<dir_name>.zip` next to the original directory.

## Behavior

- Recursively adds every file/folder under the target.
- Preserves the directory hierarchy inside the archive.
- Skips other `.zip` files at the same level to avoid nesting the output
  archive into itself.

## Notes

- Hidden files are included.
- Large directories may take several seconds; progress updates show in FERP's
  output panel.
