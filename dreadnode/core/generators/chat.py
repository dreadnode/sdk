"""
Chats are used pre and post generation to hold messages.

They are the primary way to interact with the generator.
"""

import asyncio
import typing as t
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    computed_field,
)

from dreadnode.core.generators.generator import GenerateParams, Generator, get_generator
from dreadnode.core.generators.generator.base import StopReason, Usage
from dreadnode.core.generators.message import (
    Message,
    MessageDict,
    Messages,
    MessageSlice,
    SliceType,
    inject_system_content,
)
from dreadnode.core.generators.tokenizer import TokenizedChat, Tokenizer, get_tokenizer
from dreadnode.core.tools.transforms import Transform, get_transform

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch  # type: ignore [import-not-found, unused-ignore]
    from rigging.data import ElasticOpType


P = t.ParamSpec("P")
R = t.TypeVar("R")

CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])

DEFAULT_MAX_ROUNDS = 5
"""Maximum number of internal callback rounds to attempt during generation before giving up."""

DEFAULT_MAX_DEPTH = 20
"""Maximum depth of nested pipeline generations to attempt before giving up."""

FailMode = t.Literal["raise", "skip", "include"]
"""
How to handle failures in pipelines.

- raise: Raise an exception when a failure is encountered.
- skip: Ignore the error and do not include the failed chat in the final output.
- include: Mark the message as failed and include it in the final output.
"""


class Chat(BaseModel):
    """
    A completed chat interaction.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra={"rigging.type": "chat"}
    )

    uuid: UUID = Field(default_factory=uuid4)
    """The unique identifier for the chat."""
    timestamp: datetime = Field(default_factory=datetime.now, repr=False)
    """The timestamp when the chat was created."""
    messages: list[Message]
    """The list of messages prior to generation."""
    generated: list[Message] = Field(default_factory=list)
    """The list of messages resulting from the generation."""
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    """Additional metadata for the chat."""

    stop_reason: StopReason = Field(default="unknown")
    """The reason the generation stopped."""
    usage: Usage | None = Field(None, repr=False)
    """The usage statistics for the generation if available."""
    extra: dict[str, t.Any] = Field(default_factory=dict, repr=False)
    """Any additional information from the generation."""

    generator: Generator | None = Field(None, exclude=True, repr=False)
    """The generator associated with the chat."""
    params: GenerateParams | None = Field(None, repr=False)
    """Any additional generation params used for this chat."""

    error: (
        t.Annotated[
            BaseException,
            PlainSerializer(
                lambda x: str(x),
                return_type=str,
                when_used="json-unless-none",
            ),
            WithJsonSchema({"type": "string", "description": "Error message"}),
        ]
        | None
    ) = Field(None, repr=False)
    """Holds any exception that was caught during the generation pipeline."""
    failed: bool = Field(default=False, exclude=False, repr=True)
    """
    Indicates whether conditions during generation were not met.
    This is typically used for graceful error handling when parsing.
    """

    @computed_field(repr=False)  # type: ignore [prop-decorator]
    @property
    def generator_id(self) -> str | None:
        """The identifier of the generator used to create the chat"""
        if self.generator is not None:
            return self.generator.to_identifier()
        return None

    def __init__(
        self,
        messages: Messages,
        generated: Messages | None = None,
        generator: Generator | None = None,
        params: GenerateParams | None = None,
        **kwargs: t.Any,
    ):
        """
        Initialize a Chat object.

        Args:
            messages: The messages for the chat.
            generated: The next messages for the chat.
            generator: The generator associated with this chat.
            **kwargs: Additional keyword arguments (typically used for deserialization)
        """

        if "generator_id" in kwargs and generator is None:
            generator_id = kwargs.pop("generator_id")
            if generator_id:
                generator = get_generator(generator_id)

        # We can't deserialize an error
        if isinstance(kwargs.get("error"), str):
            kwargs.pop("error")

        super().__init__(
            messages=Message.fit_as_list(messages),
            generated=Message.fit_as_list(generated) if generated is not None else [],
            generator=generator,
            params=params,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.all)

    @property
    def all(self) -> list[Message]:
        """Returns all messages in the chat, including the next messages."""
        return self.messages + self.generated

    @property
    def prev(self) -> list[Message]:
        """Alias for the .messages property"""
        return self.messages

    @property
    def next(self) -> list[Message]:
        """Alias for the .generated property"""
        return self.generated

    @property
    def last(self) -> Message:
        """Alias for .all[-1]"""
        return self.all[-1]

    @property
    def conversation(self) -> str:
        """Returns a string representation of the chat."""
        conversation = "\n\n".join([str(m) for m in self.all])
        if self.error:
            conversation += f"\n\n[error]: {self.error}"
        return conversation

    def __str__(self) -> str:
        formatted = f"--- Chat {self.uuid}"
        formatted += f"\n |- timestamp:   {self.timestamp.isoformat()}"
        if self.usage:
            formatted += f"\n |- usage:       {self.usage}"
        if self.generator:
            formatted += f"\n |- generator:   {self.generator.to_identifier(short=True)}"
        if self.stop_reason:
            formatted += f"\n |- stop_reason: {self.stop_reason}"
        if self.metadata:
            formatted += f"\n |- metadata:    {self.metadata}"
        formatted += f"\n\n{self.conversation}\n"
        return formatted

    @property
    def message_dicts(self) -> list[MessageDict]:
        """Returns the chat as a minimal message dictionaries."""
        return [t.cast("MessageDict", m.to_openai()) for m in self.all]

    @property
    def message_metadata(self) -> dict[str, t.Any]:
        """Returns a merged dictionary of metadata from all messages in the chat."""
        metadata: dict[str, t.Any] = {}
        for message in self.all:
            if message.metadata:
                metadata.update(message.metadata)
        return metadata

    def message_slices(
        self,
        slice_type: SliceType | None = None,
        filter_fn: t.Callable[[MessageSlice], bool] | None = None,
        *,
        reverse: bool = False,
    ) -> list[MessageSlice]:
        """
        Get all slices across all messages with optional filtering.

        See Message.find_slices() for more information.

        Args:
            slice_type: Filter by slice type
            filter_fn: A function to filter slices. If provided, only slices for which
                `filter_fn(slice)` returns True will be included.
            reverse: If True, the slices will be returned in reverse order.

        Returns:
            List of all matching slices across all messages
        """
        all_slices = []
        for message in self.messages:
            all_slices.extend(
                message.find_slices(slice_type=slice_type, filter_fn=filter_fn, reverse=reverse),
            )
        return all_slices

    def meta(self, **kwargs: t.Any) -> "Chat":
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat.
        """
        self.metadata.update(kwargs)
        return self

    def apply(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on the last message in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The updated chat.
        """
        if self.generated:
            self.generated[-1] = self.generated[-1].apply(**kwargs)
        else:
            self.messages[-1] = self.messages[-1].apply(**kwargs)
        return self

    def apply_to_all(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on all messages in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The updated chat.
        """
        self.messages = Message.apply_to_list(self.messages, **kwargs)
        self.generated = Message.apply_to_list(self.generated, **kwargs)
        return self

    def inject_system_content(self, content: str) -> "Chat":
        """
        Injects content into the chat as a system message.

        Note:
            If the chat is empty or the first message is not a system message,
            a new system message with the given content is inserted at the beginning of the chat.
            If the first message is a system message, the content is appended to it.

        Args:
            content: The content to be injected.

        Returns:
            The updated chat.
        """
        self.messages = inject_system_content(self.messages, content)
        return self

    def to_df(self) -> t.Any:
        """
        Converts the chat to a Pandas DataFrame.

        See [rigging.data.chats_to_df][] for more information.

        Returns:
            The chat as a DataFrame.
        """
        # Late import for circular
        from rigging.data import chats_to_df

        return chats_to_df(self)

    async def to_elastic(
        self,
        index: str,
        client: "AsyncElasticsearch",
        *,
        op_type: "ElasticOpType" = "index",
        create_index: bool = True,
        **kwargs: t.Any,
    ) -> int:
        """
        Converts the chat data to Elasticsearch format and indexes it.

        See [rigging.data.chats_to_elastic][] for more information.

        Returns:
            The number of chats indexed.
        """
        from rigging.data import chats_to_elastic

        return await chats_to_elastic(
            self,
            index,
            client,
            op_type=op_type,
            create_index=create_index,
            **kwargs,
        )

    def to_openai(self) -> list[dict[str, t.Any]]:
        """
        Converts the chat messages to the OpenAI-compatible JSON format.

        See Message.to_openai() for more information.

        Returns:
            The serialized chat.
        """
        return [m.to_openai() for m in self.all]

    async def to_tokens(
        self,
        tokenizer: str | Tokenizer,
        transform: str | Transform | None = None,
    ) -> TokenizedChat:
        """
        Converts the chat messages to a list of tokenized messages.

        Args:
            tokenizer: The tokenizer to use for tokenization. Can be a string identifier or a Tokenizer instance.
            transform: An optional transform to apply to the chat before tokenization. Can be a well-known transform
                identifier or a Transform instance.

        Returns:
            The serialized chat as a list of token lists.
        """

        if isinstance(tokenizer, str):
            tokenizer = get_tokenizer(tokenizer)

        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"Expected a Tokenizer instance, got {type(tokenizer).__name__}",
            )

        if isinstance(transform, str):
            transform = get_transform(transform)

        if transform and not isinstance(transform, Transform):
            raise TypeError(
                f"Expected a Transform instance, got {type(transform).__name__}",
            )

        chat = await self.transform(transform) if transform else self
        return await tokenizer.tokenize_chat(chat)

    async def transform(self, transform: Transform | str) -> "Chat":
        """
        Applies a transform to the chat.

        Args:
            transform: The transform to apply.

        Returns:
            A new chat with the transform applied to its messages and parameters.
        """
        if isinstance(transform, str):
            transform = get_transform(transform)
        messages = [m.clone() for m in self.messages]
        params = self.params.clone() if self.params else GenerateParams()
        messages, params, _ = await transform(self.messages, params)
        new = self.clone()
        new.messages = messages
        new.params = params
        return new


# List Helper Type


class ChatList(list[Chat]):
    """
    Represents a list of chat objects.

    Inherits from the built-in `list` class and is specialized for storing `Chat` objects.
    """

    def to_df(self) -> t.Any:
        """
        Converts the chat list to a Pandas DataFrame.

        See [rigging.data.chats_to_df][] for more information.

        Returns:
            The chat list as a DataFrame.
        """
        # Late import for circular
        from dreadnode.core.generators.data import chats_to_df

        return chats_to_df(self)

    async def to_elastic(
        self,
        index: str,
        client: "AsyncElasticsearch",
        *,
        op_type: "ElasticOpType" = "index",
        create_index: bool = True,
        **kwargs: t.Any,
    ) -> int:
        """
        Converts the chat list to Elasticsearch format and indexes it.

        See [rigging.data.chats_to_elastic][] for more information.

        Returns:
            The number of chats indexed.
        """
        from rigging.data import chats_to_elastic

        return await chats_to_elastic(
            self,
            index,
            client,
            op_type=op_type,
            create_index=create_index,
            **kwargs,
        )

    def to_json(self) -> list[dict[str, t.Any]]:
        """
        Helper to convert the chat list to a list of dictionaries.
        """
        return [chat.model_dump() for chat in self]

    def to_openai(self) -> list[list[dict[str, t.Any]]]:
        """
        Converts the chat list to a list of OpenAI-compatible JSON format.

        See Message.to_openai() for more information.

        Returns:
            The serialized chat list.
        """
        return [chat.to_openai() for chat in self]

    async def to_tokens(
        self,
        tokenizer: str | Tokenizer,
        transform: str | Transform | None = None,
    ) -> list[TokenizedChat]:
        """
        Converts the chat list to a list of tokenized chats.

        Args:
            tokenizer: The tokenizer to use for tokenization. Can be a string identifier or a Tokenizer instance.
            transform: An optional transform to apply to each chat before tokenization. Can be a well-known transform
                identifier or a Transform instance.

        Returns:
            A list of tokenized chats.
        """
        # Resolve the tokenizer first so we don't duplicate effort
        if isinstance(tokenizer, str):
            tokenizer = get_tokenizer(tokenizer)

        return await asyncio.gather(
            *(chat.to_tokens(tokenizer, transform) for chat in self),
        )
