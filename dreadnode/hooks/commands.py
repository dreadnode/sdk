# packages/agent/src/yourpkg_agent/hooks/confirm.py

from dreadnode.core.agents.events import ToolStart
from dreadnode.core.agents.reactions import Reaction, RetryWithFeedback
from dreadnode.core.hook import hook
from dreadnode.core.log import confirm

DANGEROUS_COMMANDS = {"rm", "sudo", "chmod", "chown", "mkfs", "dd"}


@hook(ToolStart, condition=lambda e: e.tool_call.name == "command")
async def confirm_dangerous_commands(event: ToolStart) -> Reaction | None:
    import json

    args = json.loads(event.tool_call.function.arguments)
    cmd = args.get("cmd", [])

    if cmd and cmd[0] in DANGEROUS_COMMANDS:
        confirm(
            f"The command '{' '.join(cmd)}' is potentially dangerous. "
            "Are you sure you want to proceed?"
        )

        return RetryWithFeedback(
            feedback=f"Command '{cmd[0]}' requires confirmation. Explain why this is needed."
        )
    return None
