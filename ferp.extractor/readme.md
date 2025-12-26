# Attachment Extractor

Pulls non-image attachments from every `.msg` file found inside a `.zip`
archive, writing each message’s files to its own folder.

---

## FERP Integration

- Requires you to highlight a `.zip` file that contains Outlook `.msg` emails.
- Extracted attachments are placed alongside the archive in
  `<zip_name>_attachments/`.

## Usage

1. Highlight the `.zip` containing `.msg` files.
2. Run **Attachment Extractor** from the Scripts panel.
3. The script unpacks the archive to a temporary folder, processes `.msg` files,
   and deletes the temporary data after completion.

## Behavior

- Saves only non-image attachments (ignores `.png`, `.jpg`, `.jpeg`, `.gif`).
- Creates one output directory per email using the message filename.
- Produces a summary result listing each processed message and the number of
  attachments saved.

## Notes

- Corrupt `.msg` files are skipped; the script continues processing the rest.
- Existing output directories are reused/merged if you run the script multiple
  times against the same archive.
