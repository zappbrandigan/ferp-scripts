from __future__ import annotations

import time
from typing import TypedDict

from ferp.fscp.scripts import sdk


class SayHello(TypedDict):
    value: str
    recursive: bool
    test: bool


class TestChoices(TypedDict):
    value: str
    selections: list[str]


@sdk.script
def main(ctx: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    temp_paths: list[str] = []

    def _cleanup() -> None:
        for temp in temp_paths:
            api.log("debug", f"Cleanup: released {temp}")

    api.register_cleanup(_cleanup)
    api.log("info", f"Hello World invoked for {ctx.target_path}")

    response_text = api.request_input(
        "Test regular input",
        id="ferp_hello_world_text",
    )

    response_dict = api.request_input_json(
        "Test json input",
        default="test",
        id="ferp_hello_world_json",
        fields=[
            {
                "id": "recursive",
                "type": "bool",
                "label": "Recursive",
                "default": False,
            },
            {"id": "test", "type": "bool", "label": "Test mode", "default": False},
        ],
        payload_type=SayHello,
    )
    selection_payload = api.request_input_json(
        "Select Your Favorite Character",
        id="ferp_process_cue_sheets_selected_pubs",
        fields=[
            {
                "id": "selections",
                "type": "multi_select",
                "label": "Character List:",
                "options": [
                    "Zapp Brannigan",
                    "Phillip J. Fry",
                    "Pofessor Hubert Farnsworh",
                    "Dr. John Zoidberg",
                ],
                "default": [],
            }
        ],
        show_text_input=False,
        payload_type=TestChoices,
    )

    confirmation_bool = api.confirm(
        "Confirm Prompt Test", default=True, id="ferp_hello_world_confirm"
    )

    for i in range(10):
        time.sleep(0.1)
        api.check_cancel()
        temp_paths.append(f"temp-resource-{i}")
        api.progress(current=i, total=10, unit="Hahas")

    api.emit_result(
        {
            "Observation": "Ah, yes. Comets, the icebergs of the sky.",
            "Life Lesson": "In the game of chess, you can never let you're adversary see your pieces.",
            "Serious Stuff": "If we can hit that bullseye, the rest of the dominoes will fall like a house of cards. Checkmate.",
            "_status": "success",
            "_title": "Raw Text",
        }
    )
    api.emit_result(
        {
            "Request Input": response_text,
            "Request Input Json": response_dict,
            "Options Selected": selection_payload["selections"],
            "Confirmation Res": confirmation_bool,
            "_status": "warn",
            "_title": "SDK API",
        }
    )
    api.emit_result(
        {
            "App": ctx.environment["app"],
            "Host": ctx.environment["host"],
            "Target Path": str(ctx.target_path),
            "Target Kind": ctx.target_kind,
            "_status": "error",
            "_title": "SDK CTX",
        }
    )


if __name__ == "__main__":
    main()
