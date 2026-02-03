# Analytics: Transliterate Excel

Transliterates text in a user-specified Excel range by removing diacritics and
converting characters to their closest ASCII equivalents. Formatting and
non-text values are preserved.

---

## FERP Integration

- Operates on the `highlighted file` in the File Navigator.
- Prompts for a cell range like `A1:C21`.
- Writes changes in place to the selected workbook.

## Usage

1. Highlight an `.xls`, `.xlsx`, `.xlsm`, or `.xlsb` file in FERP.
2. Run `Analytics: Transliterate Excel` from the Scripts panel.
3. Enter a cell range (for example, `A1:C21`).
4. Review the completion summary in the output panel.

## Behavior

- Only updates cells in the specified range that contain text.
- Skips merged cells.
- Leaves non-text cells untouched.

## Notes

- Large ranges may take a few seconds to process.
