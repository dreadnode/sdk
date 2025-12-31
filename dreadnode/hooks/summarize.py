import contextlib
import typing as t
from dataclasses import dataclass

from dreadnode.core.agents.events import AgentError, AgentStep
from dreadnode.core.agents.reactions import Continue, Reaction, Retry
from dreadnode.core.generators.chat import Chat
from dreadnode.core.generators.generator import Generator
from dreadnode.core.generators.message import Message
from dreadnode.core.hook import Hook, hook
from dreadnode.core.meta import Config, component

if t.TYPE_CHECKING:
    from dreadnode.core.hook import Hook

CONTEXT_LENGTH_ERROR_PATTERNS = [
    "context_length_exceeded",
    "context window",
    "token limit",
    "maximum context length",
    "is too long",
]


@dataclass
class Summary:
    analysis: str
    summary: str
    chat: Chat


async def summarize_conversation(conversation: str, *, guidance: str = "") -> Summary:  # type: ignore[empty-body]
    """
    Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
    This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

    Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

    1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
    - The user's explicit requests and intents
    - Your approach to addressing the user's requests
    - Key decisions, technical concepts and code patterns
    - Specific technical details like paths, usernames, structured objects, and code
    - Tool interactions performed with a specific focus on intent and outcome
    - Errors that you ran into and how you fixed them
    - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.

    2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

    Your summary should include the following sections:

    1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
    2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
    3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
    4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
    5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
    6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
    7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
    8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
    9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
    If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

    Here's an example of how your output should be structured:

    <example>
    <analysis>
    [Your thought process, ensuring all points are covered thoroughly and accurately]
    </analysis>

    <summary>
    1. Primary Request and Intent:
    [Detailed description]

    2. Key Technical Details:

    - [Concept 1]
    - [Concept 2]
    - [Object 1]
    - [...]

    3. Tool Interactions:

    - [Tool Call 1]: [Description of what the tool did, inputs, outputs]
    - [Tool Call 2]: ...
    - [...]

    4. Errors and fixes:

    - [Detailed description of error 1]:
        - [How you fixed the error]
        - [User feedback on the error if any]
    - [...]

    5. Problem Solving:
    [Description of solved problems and ongoing troubleshooting]

    6. All user messages:

    - [Detailed non tool use user message]
    - [...]

    7. Pending Tasks:

    - [Task 1]
    - [Task 2]
    - [...]

    8. Current Work:
    [Precise description of current work]

    9. Optional Next Step:
    [Optional Next step to take]

    </summary>
    </example>

    Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

    There may be additional summarization guidance provided. If so, remember to follow these instructions when creating the above summary.
    """


def _is_context_length_error(error: BaseException) -> bool:
    """Checks if an exception is likely due to exceeding the context window."""
    with contextlib.suppress(ImportError):
        from litellm.exceptions import ContextWindowExceededError

        if isinstance(error, ContextWindowExceededError):
            return True

    error_str = str(error).lower()
    return any(pattern in error_str for pattern in CONTEXT_LENGTH_ERROR_PATTERNS)


@hook(AgentStep, AgentError)
def summarize_when_long(
    model: str | Generator | None = None,
    max_tokens: int = 100_000,
    min_messages_to_keep: int = 10,
    guidance: str = "",
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
        guidance: Additional guidance for the summarization process (default is "").
    """

    if min_messages_to_keep < 2:
        raise ValueError("min_messages_to_keep must be at least 2.")

    @component
    async def summarize_when_long(
        step: AgentStep,
        *,
        model: str | Generator | None = Config(
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
        guidance: str = Config(
            guidance,
            help="Additional guidance for the summarization process",
        ),
    ) -> Reaction | None:
        should_summarize = False

        if max_tokens is not None and isinstance(step, AgentStep):
            last_token_count = step.usage.total_tokens
            if last_token_count > 0 and last_token_count > max_tokens:
                should_summarize = True

        elif isinstance(step, AgentError):
            if _is_context_length_error(step.error):
                should_summarize = True

        if not should_summarize:
            return None

        summarizer_model = model or step.generator
        if summarizer_model is None:
            return None

        messages = list(step.messages)

        if len(messages) <= min_messages_to_keep:
            return None

        system_message: Message | None = (
            messages.pop(0) if messages and messages[0].role == "system" else None
        )

        best_summarize_boundary = 0
        for i, message in enumerate(messages):
            if len(messages) - i <= min_messages_to_keep:
                break

            is_simple_assistant = message.role == "assistant" and not getattr(
                message, "tool_calls", None
            )

            is_last_tool_in_block = message.role == "tool" and (
                i + 1 == len(messages) or messages[i + 1].role != "tool"
            )

            if is_simple_assistant or is_last_tool_in_block:
                best_summarize_boundary = i + 1

        if best_summarize_boundary == 0:
            return None

        messages_to_summarize = messages[:best_summarize_boundary]
        messages_to_keep = messages[best_summarize_boundary:]

        if not messages_to_summarize:
            return None

        summary = await summarize_conversation.bind(summarizer_model)(
            "\n".join(str(msg) for msg in messages_to_summarize), guidance=guidance
        )
        summary_content = (
            f"<conversation-summary messages={len(messages_to_summarize)}>\n"
            f"{summary.summary}\n"
            "</conversation-summary>"
        )

        new_messages: list[Message] = []
        if system_message:
            new_messages.append(system_message)
        new_messages.append(Message("user", summary_content, metadata={"summary": True}))
        new_messages.extend(messages_to_keep)

        return (
            Continue(messages=new_messages)
            if isinstance(step, AgentStep)
            else Retry(messages=new_messages)
        )

    return summarize_when_long
