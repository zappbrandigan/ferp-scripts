# Hello World

Demonstrates FERP’s FSCP scripting flow by asking for simple input, logging a
message, and returning the captured data.

---

## FERP Integration

- Runs against the `current directory` shown in the File Navigator.
- Prompts for a file extension (defaults to `.txt`) using FERP's dialog.

## Usage

1. Highlight any folder in FERP.
2. Run **Hello World** from the Scripts panel.
3. Enter an extension when prompted, or accept the default.

## Behavior

- Emits a structured result containing the target path, provided extension, and
  any additional arguments configured in `config.json`.
- Shows how `api.log`, `api.request_input`, and `api.emit_result` behave inside FERP.

## Notes

- Intended purely for testing/demo purposes; it does not modify files.
