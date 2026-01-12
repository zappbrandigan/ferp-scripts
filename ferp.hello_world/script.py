from __future__ import annotations

from ferp.fscp.scripts import sdk
import time


@sdk.script
def main(context: sdk.ScriptContext, api: sdk.ScriptAPI) -> None:
    api.log("info", f"Hello World invoked for {context.target_path}")

    extension = api.request_input(
        "Enter the file extension",
        default=".txt",
    )

    for i in range(10):
        time.sleep(0.2)
        api.progress(current=i, total=10, unit="seconds")

    api.emit_result(
        {
            "message": "Hello from FSCP!",
            "extension": extension,
            "args": context.args,
        }
    )


if __name__ == "__main__":
    main()
