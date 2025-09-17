import contextlib
import typing as t

import rigging as rg

from dreadnode.agent.events import AgentError, AgentEvent, GenerationEnd, StepStart
from dreadnode.agent.prompts import summarize_conversation
from dreadnode.agent.reactions import Continue, Reaction, Retry
from dreadnode.meta import Config, component

if t.TYPE_CHECKING:
    from dreadnode.agent.hooks.base import Hook

# Best-effort match for some common error patterns
CONTEXT_LENGTH_ERROR_PATTERNS = [
    "context_length_exceeded",
    "context window",
    "token limit",
    "maximum context length",
    "is too long",
]


def _is_context_length_error(error: BaseException) -> bool:
    """Checks if an exception is likely due to exceeding the context window."""
    with contextlib.suppress(ImportError):
        from litellm.exceptions import ContextWindowExceededError

        if isinstance(error, ContextWindowExceededError):
            return True

    error_str = str(error).lower()
    return any(pattern in error_str for pattern in CONTEXT_LENGTH_ERROR_PATTERNS)


def _get_last_input_tokens(event: AgentEvent) -> int:
    """
    Finds the input token count from the most recent GenerationEnd event in the thread.
    This represents the size of the context for the last successful model call.
    """
    last_generation_event = event.get_latest_event_by_type(GenerationEnd)
    if not last_generation_event:
        return 0
    return last_generation_event.usage.input_tokens if last_generation_event.usage else 0


@component
def summarize_when_long(
    model: str | rg.Generator | None = None,
    max_tokens: int = 100_000,
    min_messages_to_keep: int = 5,
) -> "Hook":
    """
    Creates a hook to manage the agent's context window by summarizing the conversation history.

    This hook operates in two ways:
    1.  **Proactively (on `StepStart`)**: Before each step, it checks the `input_tokens` from the
        last `GenerationEnd` event. If it exceeds `max_tokens`, it summarizes older messages.
    2.  **Reactively (on `AgentError`)**: If the agent fails with a context length error,
        it summarizes the history and retries the step.

    Args:
        model: The model identifier or generator to use for summarization, otherwise it will use the agent's model.
        max_tokens: The maximum number of tokens allowed in the context window before summarization is triggered
            (default is None, meaning no proactive summarization).
        min_messages_to_keep: The minimum number of messages to retain after summarization (default is 5).
    """

    if min_messages_to_keep < 2:
        raise ValueError("min_messages_to_keep must be at least 2.")

    @component
    async def summarize_when_long(  # noqa: PLR0912
        event: AgentEvent,
        *,
        model: str | rg.Generator | None = Config(  # noqa: B008
            model,
            help="Model to use for summarization - fallback to the agent model",
            expose_as=str | None,
        ),
        max_tokens: int | None = Config(
            max_tokens,
            help="Maximum number of tokens observed before summarization is triggered",
        ),
        min_messages_to_keep: int = Config(
            5, help="Minimum number of messages to retain after summarization"
        ),
    ) -> Reaction | None:
        should_summarize = False

        # Proactive check using the last known token count
        if max_tokens is not None and isinstance(event, StepStart):
            last_token_count = _get_last_input_tokens(event)
            if last_token_count > 0 and last_token_count > max_tokens:
                should_summarize = True

        # Reactive check based on the error message
        elif isinstance(event, AgentError):
            if _is_context_length_error(event.error):
                should_summarize = True

        if not should_summarize:
            return None

        summarizer_model = model or event.agent.model
        if summarizer_model is None:
            return None

        messages = list(event.messages)

        # Check if we have enough messages to summarize
        if len(messages) <= min_messages_to_keep:
            return None

        # Exclude the system message from the summarization process.
        system_message: rg.Message | None = (
            messages.pop(0) if messages and messages[0].role == "system" else None
        )

        # Find the best point to summarize by walking the message list once.
        # A boundary is valid after a simple assistant message or a finished tool block.
        best_summarize_boundary = 0
        for i, message in enumerate(messages):
            # If the remaining messages are less than or equal to our minimum, we can't slice any further.
            if len(messages) - i <= min_messages_to_keep:
                break

            # Condition 1: The message is an assistant response without tool calls.
            is_simple_assistant = message.role == "assistant" and not getattr(
                message, "tool_calls", None
            )

            # Condition 2: The message is the last in a block of tool responses.
            is_last_tool_in_block = message.role == "tool" and (
                i + 1 == len(messages) or messages[i + 1].role != "tool"
            )

            if is_simple_assistant or is_last_tool_in_block:
                best_summarize_boundary = i + 1

        if best_summarize_boundary == 0:
            return None  # No valid slice point was found.

        messages_to_summarize = messages[:best_summarize_boundary]
        messages_to_keep = messages[best_summarize_boundary:]

        if not messages_to_summarize:
            return None

        # Generate the summary and rebuild the messages
        summary = await summarize_conversation.bind(summarizer_model)(
            "\n".join(str(msg) for msg in messages_to_summarize)
        )
        summary_content = (
            f"<conversation-summary messages={len(messages_to_summarize)}>\n"
            f"{summary.summary}\n"
            "</conversation-summary>"
        )

        new_messages: list[rg.Message] = []
        if system_message:
            new_messages.append(system_message)
        new_messages.append(rg.Message("user", summary_content, metadata={"summary": True}))
        new_messages.extend(messages_to_keep)

        return (
            Continue(messages=new_messages)
            if isinstance(event, StepStart)
            else Retry(messages=new_messages)
        )

    return summarize_when_long
