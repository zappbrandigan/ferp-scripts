# Circulation Email

Generates `.eml` cue sheet circulation drafts from the expected GFTV folder layout.

---

## FERP Integration

- Operates on a `highlighted directory`.
- Scans PDF files recursively under the selected directory.
- Writes one or more `.eml` draft files next to the selected directory.
- Uses the host-managed GFTV cues inbox setting for both the `From` and `To` fields.

## Expected Layout

The script recognizes cue sheet PDFs stored in either of these path shapes:

- `date_range/territory_codes/film/catalog_code - catalog_name/cue_sheets.pdf`
- `date_range/territory_codes/tv/catalog_code - catalog_name/cue_sheets.pdf`
- `date_range/_REV/territory_codes/film/catalog_code - catalog_name/cue_sheets.pdf`
- `date_range/_REV/territory_codes/tv/catalog_code - catalog_name/cue_sheets.pdf`

## Draft Behavior

- Groups recognized PDFs by `media type`, `territory code`, and `New` vs `Revised`.
- Uses the catalog code before ` - ` in the folder name for the email subject.
- Uses the catalog name after ` - ` in the bullet list.
- Leaves `{deal_owner}`, `{revision_notes}`, and `{link}` as manual placeholders.
- Does not add a signature block.
- Formats the HTML email body in Calibri `11pt`, uses a normal bullet list, and bolds the link-expiration note.
- Requires the GFTV cues inbox to be configured in FERP before the script is run.

## Output

- Generates one draft per grouping, for example:
  - `<selected-directory>_tv_world_new_circulation.eml`
  - `<selected-directory>_film_ca_revised_circulation.eml`
- Uses `tv-circulations.txt` or `film-circulations.txt` as the source template.
- Logs and counts PDFs that do not match the expected folder structure.
