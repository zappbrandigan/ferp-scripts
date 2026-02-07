# Convert HTML to PDF

Converts HTML files to PDF using a local Chrome/Chromium install.

---

## FERP Integration

- Operates on a `highlighted directory` or `highlighted file` of type `.html` in the File Navigator.
- Prompts for recursive scan when a directory is targeted.
- Emits per-file progress and reports failures without stopping the batch.

## Usage

1. Navigate to a directory containing HTML files (or select a single `.html` file).
2. Run **Convert: HTML to PDF** from the Scripts panel.
3. If prompted, choose whether to scan subdirectories.

## Behavior

- Finds `*.html` files in the target directory (optionally recursive).
- Renders each HTML file with headless Chrome to a PDF next to the source file.
- Adds a numeric suffix if a PDF with the same name already exists.
- Moves successfully converted HTML files into a sibling `_og` folder.
- Reports conversion and archive failures while continuing the batch.

## Notes

- Requires a valid Chrome/Chromium path (stored in the script settings).
- Typical Mac OS path: /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome
- On Windows, you can identify the path by following these steps:
    1. Click Start
    2. Type Chrome
    3. Right-click Google Chrome
    4. Click Open file location
    5. In the folder that opens, right-click Google Chrome
    6. Click Properties
    7. Copy the Target path
