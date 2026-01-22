from __future__ import annotations

import time
from typing import TypedDict

from ferp.fscp.scripts import sdk


class SayHello(TypedDict):
    value: str
    recursive: bool
    test: bool


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
                "label": "Scan subdirectories",
                "default": False,
            },
            {"id": "test", "type": "bool", "label": "Test mode", "default": False},
        ],
        payload_type=SayHello,
    )
    # response_dict["value"]
    # response_dict["recursive"]
    # response_dict["test"]
    confirmation_bool = api.confirm(
        "Confirm Prompt Test", default=True, id="ferp_hello_world_confirm"
    )

    for i in range(10):
        time.sleep(1)
        api.check_cancel()
        temp_paths.append(f"temp-resource-{i}")
        api.progress(current=i, total=10, unit="Hahas")

    api.emit_result(
        {
            "Hello Text": "Here is a hello to you as well.",
            "Howdy Textual": "In the game of chess, you can never let you're adversary see your pieces.",
            "Serious Stuff": "If we can hit that bullseye, the rest of the dominos will fall like a house of cards. Checkmate.",
            "_status": "success",
            "_title": "Custom Text",
        }
    )
    api.emit_result(
        {
            "Request Input": response_text,
            "Request Input Json": response_dict,
            "Confirm": confirmation_bool,
            "_status": "warn",
            "_title": "Hello Inputs",
        }
    )
    api.emit_result(
        {
            "Environment": ctx.environment,
            "Target Path": str(ctx.target_path),
            "Target Kind": ctx.target_kind,
            "Params": ctx.params,
            "_status": "error",
            "_title": "Hello SDK",
        }
    )


if __name__ == "__main__":
    main()
