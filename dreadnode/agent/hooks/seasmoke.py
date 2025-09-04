import typing as t
import json

import rigging as rg
from dreadnode.agent.events import AgentError, AgentEvent, GenerationEnd, StepStart
from dreadnode.agent.hooks.summarize import (
    is_context_length_error,
    get_last_input_tokens,
)
from dreadnode.agent.types import Message
from dreadnode.agent.reactions import Retry, Reaction, ToolEnd

if t.TYPE_CHECKING:
    from dreadnode.agent.hooks.base import Hook


class Parameter(rg.Model):
    name: str = rg.attr()
    value: t.Union[str, int, float, None] = None


class ToolCallRecord(rg.Model):
    name: str = rg.element(tag="Name")
    parameters: list[Parameter] = rg.wrapped("parameters", rg.element(default=[]))
    output: t.Optional[str] = rg.element(tag="ouptut", default=None)


def _estimate_token_size(text: str) -> int:
    """rough estimator for token size of text"""
    return int(len(text) / 4)


def summarize(
    event: AgentError,
    exlude_tool_calls: list[str] = [
        "store_output",
        "get_output",
        "store_data",
        "get_data",
        "store_note",
        "get_notes",
    ],
    reduce_ratio: float = .7
) -> "Hook":
    """
    Hook is designed for context window errors.

    What will get passed in:

        AgentError(
        agent=self,
        thread=thread,
        messages=messages,
        events=events,
        error=e,
    )

    return Continue reaction with new messages
    """

    async def _summarize() -> Reaction | None:
        """summarizes prompt, notes, outputs, data objects, tool call history from a Seasmoke agent"""

        if not is_context_length_error(event.error):
            return None

        last_token_count = get_last_input_tokens(event)

        sum_context = ""
        # sum_context += event.messages[0].content  # system prompt
        sum_context += f"\n{event.messages[1].content}"  # user prompt
        sum_context += "\n\nYou have already been working on the previously described task. Below is the most recent context on your progress."
       
        sum_context += "\n\nData you have previously stored:"
        for k, v in event.agent.memory.get_data().items():
            sum_context += f"\n<key>{k}</key> <value>{v}</value"

        sum_context += "\n\nPrevious Output:"
        for k, v in event.agent.memory.get_output().items():
            for w in v:
                sum_context += f"\n\n{w}"

        sum_context += "\n\nPrevious Tool Calls:\n\n"
        for i in event.agent.memory.tool_calls:
            if i.name not in exlude_tool_calls:
                continue

            # Tool call history can fill remaining token bandwidth until token size surpasses (reduce_ratio * previous_token_count)
            if _estimate_token_size(sum_context) > int((reduce_ratio * last_token_count)):
                break

            sum_context += str(
                ToolCallRecord(
                    name=i.name,
                    parameters=[
                        Parameter(name=k, value=v)
                        for k, v in json.loads(i.function.arguments).items()
                    ],
                )
            )
            sum_context += "\n"

        new_messages = [
            event.messages[0],
            Message("user", sum_context, metadata={"summary": True}),
        ]

        return Retry(messages=new_messages)

    return _summarize


def log_tool_call(event: ToolEnd) -> "Hook":
    """records tool calls to the Seasmoke agent's memory"""

    async def _log_tool_call() -> None:
        """ """
        event.agent.memory.tool_calls.append(event.tool_call)
        return None

    return _log_tool_call
