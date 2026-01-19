# Compress: Unzip File

Extracts a highlighted `.zip` or `.7z` file into a sibling directory that matches the
archive name.

---

## FERP Integration

- Operates on the `highlighted file`; the file must end with `.zip` or `.7z`.
- The output directory lives next to the archive.
- Warns before overwriting an existing directory with the same name.

## Usage

1. Highlight a `.zip` or `.7z` file in FERP.
2. Run `Compress: Unzip File` from the Scripts panel.
3. Confirm/cancel overwrites if prompted.
3. The script creates `<archive_name>/` next to the original archive and extracts contents there.

## Behavior

- Warns before overwriting an existing directory with the same name.
- Preserves the directory structure found in the archive.
- Leaves the original `.zip` untouched.

## Notes

- Corrupt or encrypted archives stop the script and surface an error message in FERP.
- Large directories may take several seconds; progress updates show in FERP's
