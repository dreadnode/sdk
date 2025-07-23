import typing as t

from dreadnode.metric import Metric, Scorer

if t.TYPE_CHECKING:
    from rigging.chat import Chat
    from rigging.message import Message

ChatFilterMode = t.Literal[
    "all", "last", "first", "user", "assistant", "last_user", "last_assistant"
]
ChatFilterFunction = t.Callable[["list[Message]"], list["Message"]]


def wrap_chat(
    inner_scorer: Scorer[t.Any],
    *,
    filter: ChatFilterMode | ChatFilterFunction = "last",
    name: str | None = None,
) -> "Scorer[Chat]":
    """
    Wraps a text-based scorer to work on a `rigging.Chat` object.

    This function acts as an adapter. It extracts and filters messages from a
    `Chat` object, converts them to a single string, and then passes that
    string to the `inner_scorer` for evaluation.

    Args:
        inner_scorer: The text-based Scorer instance to wrap (e.g., one from `contains` or `similarity_to`).
        filter: The strategy for filtering which messages to include:
            - "all": Use all messages in the chat.
            - "last": Use only the last message.
            - "first": Use only the first message.
            - "user": Use only user messages.
            - "assistant": Use only assistant messages.
            - "last_user": Use only the last user message.
            - "last_assistant": Use only the last assistant message.
            - A callable that takes a list of `Message` objects and returns a filtered list.
        name: An optional name for the new, wrapped scorer. If None, a descriptive name is generated.

    Returns:
        A new Scorer that takes a `Chat` object as input.
    """

    async def evaluate(chat: "Chat") -> Metric:
        from rigging.chat import Chat

        # Fall through to the inner scorer if chat is not a Chat instance
        if not isinstance(chat, Chat):
            return await inner_scorer(chat)

        messages = chat.all
        if callable(filter):
            messages = filter(messages)
        elif filter == "last":
            messages = messages[-1:] if messages else []
        elif filter == "first":
            messages = messages[:1] if messages else []
        elif filter == "user":
            messages = [m for m in messages if m.role == "user"]
        elif filter == "assistant":
            messages = [m for m in messages if m.role == "assistant"]
        elif filter == "last_user":
            user_messages = [m for m in messages if m.role == "user"]
            messages = user_messages[-1:] if user_messages else []
        elif filter == "last_assistant":
            assistant_messages = [m for m in messages if m.role == "assistant"]
            messages = assistant_messages[-1:] if assistant_messages else []

        all_text = "\n".join(msg.content for msg in messages if msg.content is not None)
        return await inner_scorer(all_text)

    if name is None:
        name = f"chat_{inner_scorer.name}"

    return Scorer.from_callable(evaluate, name=name)
