"""Tests for the generators module."""

import copy
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from dreadnode.core.generators.message import (
    Message,
    MessageSlice,
    ContentText,
    ContentImageUrl,
    ContentAudioInput,
    Role,
    inject_system_content,
    strip_system_content,
)
from dreadnode.core.generators.generator.base import (
    Generator,
    GenerateParams,
    Usage,
    GeneratedMessage,
    GeneratedText,
    StopReason,
    convert_stop_reason,
    get_identifier,
    get_generator,
    register_generator,
    Fixup,
    with_fixups,
)
from dreadnode.core.generators.chat import Chat, ChatList
from dreadnode.core.generators.caching import apply_cache_mode_to_messages


# ==============================================================================
# Message Tests
# ==============================================================================


class TestMessageCreation:
    """Tests for Message creation."""

    def test_basic_creation(self):
        """Test creating a basic message."""
        msg = Message(role="user", content="Hello, world!")

        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert len(msg.content_parts) == 1
        assert isinstance(msg.content_parts[0], ContentText)

    def test_creation_with_none_content(self):
        """Test creating a message with None content defaults to empty string."""
        msg = Message(role="user", content=None)

        assert msg.content == ""

    def test_creation_with_list_content(self):
        """Test creating a message with list of content."""
        msg = Message(role="user", content=["Hello", "World"])

        assert len(msg.content_parts) == 2
        assert msg.content == "Hello\nWorld"

    def test_creation_with_cache_control(self):
        """Test creating a message with cache control."""
        msg = Message(role="user", content="Hello", cache_control="ephemeral")

        assert msg.content_parts[-1].cache_control is not None
        assert msg.content_parts[-1].cache_control["type"] == "ephemeral"

    def test_creation_with_tool_calls(self):
        """Test creating a message with tool calls."""
        from dreadnode.core.tools import ToolCall, FunctionCall

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = Message(role="assistant", content="", tool_calls=[tool_call])

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"

    def test_creation_with_tool_call_id(self):
        """Test creating a message with tool_call_id."""
        msg = Message(role="tool", content="Result", tool_call_id="call_123")

        assert msg.tool_call_id == "call_123"

    def test_uuid_generation(self):
        """Test that UUIDs are generated for messages."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")

        assert isinstance(msg1.uuid, UUID)
        assert isinstance(msg2.uuid, UUID)
        assert msg1.uuid != msg2.uuid

    def test_dedent_content(self):
        """Test that content is dedented."""
        msg = Message(role="user", content="""
            Hello
            World
        """)

        # dedent should remove leading whitespace
        assert "            Hello" not in msg.content


class TestMessageProperties:
    """Tests for Message properties."""

    def test_len_returns_content_length(self):
        """Test that len() returns content length."""
        msg = Message(role="user", content="Hello")

        assert len(msg) == 5

    def test_str_representation(self):
        """Test string representation."""
        msg = Message(role="user", content="Hello")

        result = str(msg)
        assert "[user]:" in result
        assert "Hello" in result

    def test_hash_property(self):
        """Test hash property."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")
        msg3 = Message(role="user", content="World")

        # Same content should have same hash (ignoring uuid/metadata)
        assert msg1.hash == msg2.hash
        # Different content should have different hash
        assert msg1.hash != msg3.hash


class TestMessageContentProperty:
    """Tests for Message.content property."""

    def test_content_getter_single_text(self):
        """Test getting content with single text part."""
        msg = Message(role="user", content="Hello")

        assert msg.content == "Hello"

    def test_content_getter_multiple_text(self):
        """Test getting content with multiple text parts."""
        msg = Message(role="user", content=["Hello", "World"])

        assert msg.content == "Hello\nWorld"

    def test_content_setter(self):
        """Test setting content."""
        msg = Message(role="user", content="Hello")
        msg.content = "Goodbye"

        assert msg.content == "Goodbye"

    def test_content_setter_preserves_slices(self):
        """Test that content setter preserves slices if content still contains them."""
        msg = Message(role="user", content="Hello World")
        msg.mark_slice("World", "other")

        msg.content = "Goodbye World"

        # The "World" slice should still be present
        assert len(msg.slices) == 1
        assert msg.slices[0].content == "World"


class TestMessageSliceOperations:
    """Tests for Message slice operations."""

    def test_mark_slice_with_string(self):
        """Test marking a slice with string target."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.mark_slice("World")

        assert slice_ is not None
        assert slice_.type == "other"
        assert slice_.content == "World"

    def test_mark_slice_with_tuple_range(self):
        """Test marking a slice with tuple range."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.mark_slice((0, 5))

        assert slice_ is not None
        assert slice_.content == "Hello"

    def test_mark_slice_entire_content(self):
        """Test marking entire content as slice."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.mark_slice(-1)

        assert slice_ is not None
        assert slice_.content == "Hello World"

    def test_mark_slice_with_regex(self):
        """Test marking a slice with regex pattern."""
        import re

        msg = Message(role="user", content="Hello 123 World 456")
        slices = msg.mark_slice(re.compile(r"\d+"), select="all")

        assert len(slices) == 2
        assert slices[0].content == "123"
        assert slices[1].content == "456"

    def test_mark_slice_select_first(self):
        """Test selecting first matching slice."""
        msg = Message(role="user", content="Hello Hello Hello")
        slice_ = msg.mark_slice("Hello", select="first")

        assert slice_ is not None
        assert slice_.start == 0

    def test_mark_slice_select_last(self):
        """Test selecting last matching slice."""
        msg = Message(role="user", content="Hello Hello Hello")
        slice_ = msg.mark_slice("Hello", select="last")

        assert slice_ is not None
        assert slice_.start == 12

    def test_mark_slice_select_all(self):
        """Test selecting all matching slices."""
        msg = Message(role="user", content="Hello Hello Hello")
        slices = msg.mark_slice("Hello", select="all")

        assert len(slices) == 3

    def test_mark_slice_case_insensitive(self):
        """Test case-insensitive slice marking."""
        msg = Message(role="user", content="Hello WORLD")
        slice_ = msg.mark_slice("world", case_sensitive=False)

        assert slice_ is not None
        assert slice_.content == "WORLD"

    def test_append_slice(self):
        """Test appending content with slice."""
        msg = Message(role="user", content="Hello")
        slice_ = msg.append_slice(" World")

        assert "World" in msg.content
        assert slice_.type == "other"

    def test_replace_with_slice(self):
        """Test replacing content with slice."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.replace_with_slice("Goodbye")

        assert msg.content == "Goodbye"
        assert len(msg.slices) == 1
        assert slice_.content == "Goodbye"

    def test_find_slices(self):
        """Test finding slices with filter."""
        msg = Message(role="user", content="Hello World Foo Bar")
        msg.mark_slice("Hello", "other")
        msg.mark_slice("World", "model")
        msg.mark_slice("Foo", "other")

        other_slices = msg.find_slices(slice_type="other")

        assert len(other_slices) == 2

    def test_get_slice(self):
        """Test getting a single slice."""
        msg = Message(role="user", content="Hello World")
        msg.mark_slice("Hello", "other")
        msg.mark_slice("World", "model")

        slice_ = msg.get_slice("model")

        assert slice_ is not None
        assert slice_.content == "World"

    def test_iter_slices(self):
        """Test iterating slices."""
        msg = Message(role="user", content="A B C")
        msg.mark_slice("A", "other")
        msg.mark_slice("B", "other")
        msg.mark_slice("C", "other")

        slices = list(msg.iter_slices())

        assert len(slices) == 3
        # Should be sorted by start position
        assert slices[0].content == "A"
        assert slices[2].content == "C"

    def test_remove_slices(self):
        """Test removing slices."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.mark_slice("World")

        removed = msg.remove_slices(slice_)

        assert len(removed) == 1
        assert "World" not in msg.content


class TestMessageClone:
    """Tests for Message.clone()."""

    def test_clone_creates_copy(self):
        """Test that clone creates a separate copy."""
        msg = Message(role="user", content="Hello")
        cloned = msg.clone()

        assert cloned.content == msg.content
        assert cloned.role == msg.role
        assert cloned is not msg

    def test_clone_deep_copies_metadata(self):
        """Test that metadata is deep copied."""
        msg = Message(role="user", content="Hello")
        msg.metadata["key"] = {"nested": "value"}

        cloned = msg.clone()
        cloned.metadata["key"]["nested"] = "modified"

        assert msg.metadata["key"]["nested"] == "value"


class TestMessageToOpenAI:
    """Tests for Message.to_openai()."""

    def test_to_openai_basic(self):
        """Test basic to_openai conversion."""
        msg = Message(role="user", content="Hello")

        result = msg.to_openai()

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_to_openai_with_tool_calls(self):
        """Test to_openai with tool calls."""
        from dreadnode.core.tools import ToolCall, FunctionCall

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )
        msg = Message(role="assistant", content="", tool_calls=[tool_call])

        result = msg.to_openai()

        assert "tool_calls" in result

    def test_to_openai_skip_tools_flag(self):
        """Test to_openai with skip_tools compatibility flag."""
        from dreadnode.core.tools import ToolCall, FunctionCall

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )
        msg = Message(role="assistant", content="", tool_calls=[tool_call])

        result = msg.to_openai(compatibility_flags={"skip_tools"})

        assert "tool_calls" not in result


class TestMessageMethods:
    """Tests for Message methods."""

    def test_meta(self):
        """Test meta method updates metadata."""
        msg = Message(role="user", content="Hello")
        result = msg.meta(key="value")

        assert result is msg
        assert msg.metadata["key"] == "value"

    def test_cache_method(self):
        """Test cache method."""
        msg = Message(role="user", content="Hello")
        result = msg.cache(True)

        assert result is msg
        assert msg.content_parts[-1].cache_control is not None

    def test_cache_method_disable(self):
        """Test cache method with False."""
        msg = Message(role="user", content="Hello", cache_control="ephemeral")
        msg.cache(False)

        assert msg.content_parts[-1].cache_control is None

    def test_apply_template(self):
        """Test apply method with template substitution."""
        msg = Message(role="user", content="Hello $name!")
        result = msg.apply(name="World")

        assert result.content == "Hello World!"
        assert result is not msg  # Should be a clone

    def test_truncate(self):
        """Test truncate method."""
        msg = Message(role="user", content="Hello World! This is a long message.")
        result = msg.truncate(15)

        assert len(result.content) <= 15 + len("\n[truncated]")

    def test_shorten(self):
        """Test shorten method."""
        msg = Message(role="user", content="Hello World! This is a long message.")
        result = msg.shorten(20)

        assert len(result.content) <= 20
        assert "..." in result.content


class TestMessageFromModel:
    """Tests for Message.from_model()."""

    def test_from_model_basic(self):
        """Test creating message from model."""
        from rigging.model import Model

        class TestModel(Model):
            value: str

        model = TestModel(value="test")
        msg = Message.from_model(model)

        assert msg.role == "user"
        # Model name is converted to kebab-case in XML: test-model
        assert "test-model" in msg.content
        assert len(msg.slices) == 1
        assert msg.slices[0].type == "model"


class TestMessageFit:
    """Tests for Message.fit()."""

    def test_fit_from_string(self):
        """Test fit from string."""
        msg = Message.fit("Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_fit_from_dict(self):
        """Test fit from dict."""
        msg = Message.fit({"role": "assistant", "content": "Hello"})

        assert msg.role == "assistant"
        assert msg.content == "Hello"

    def test_fit_from_message(self):
        """Test fit from existing message."""
        original = Message(role="user", content="Hello")
        msg = Message.fit(original)

        assert msg is not original
        assert msg.content == original.content

    def test_fit_as_list(self):
        """Test fit_as_list."""
        msgs = Message.fit_as_list(["Hello", "World"])

        assert len(msgs) == 2
        assert all(isinstance(m, Message) for m in msgs)


class TestSystemContentHelpers:
    """Tests for inject_system_content and strip_system_content."""

    def test_inject_into_empty(self):
        """Test injecting into empty list."""
        messages: list[Message] = []
        result = inject_system_content(messages, "System prompt")

        assert len(result) == 1
        assert result[0].role == "system"
        assert result[0].content == "System prompt"

    def test_inject_creates_new_system(self):
        """Test injecting creates new system message when first is not system."""
        messages = [Message(role="user", content="Hello")]
        result = inject_system_content(messages, "System prompt")

        assert len(result) == 2
        assert result[0].role == "system"

    def test_inject_appends_to_existing_system(self):
        """Test injecting appends to existing system message."""
        messages = [Message(role="system", content="Original")]
        result = inject_system_content(messages, "Additional")

        assert len(result) == 1
        assert "Original" in result[0].content
        assert "Additional" in result[0].content

    def test_inject_empty_content_no_op(self):
        """Test injecting empty content does nothing."""
        messages = [Message(role="user", content="Hello")]
        result = inject_system_content(messages, "   ")

        assert len(result) == 1

    def test_strip_system_content(self):
        """Test stripping system content."""
        messages = [Message(role="system", content="System prompt")]
        result = strip_system_content(messages, "System prompt")

        assert len(result) == 0

    def test_strip_partial_content(self):
        """Test stripping partial system content."""
        messages = [Message(role="system", content="Keep this. Remove this.")]
        result = strip_system_content(messages, "Remove this.")

        assert len(result) == 1
        assert result[0].content == "Keep this."


# ==============================================================================
# MessageSlice Tests
# ==============================================================================


class TestMessageSlice:
    """Tests for MessageSlice class."""

    def test_basic_creation(self):
        """Test creating a basic slice."""
        slice_ = MessageSlice(type="other", start=0, stop=5)

        assert slice_.type == "other"
        assert slice_.start == 0
        assert slice_.stop == 5

    def test_len(self):
        """Test slice length."""
        slice_ = MessageSlice(type="other", start=0, stop=5)

        assert len(slice_) == 5

    def test_slice_property(self):
        """Test slice_ property returns Python slice."""
        slice_ = MessageSlice(type="other", start=0, stop=5)

        result = slice_.slice_

        assert isinstance(result, slice)
        assert result.start == 0
        assert result.stop == 5

    def test_content_with_attached_message(self):
        """Test content property with attached message."""
        msg = Message(role="user", content="Hello World")
        slice_ = msg.mark_slice("World")

        assert slice_.content == "World"

    def test_content_detached(self):
        """Test content property when detached."""
        slice_ = MessageSlice(type="other", start=0, stop=5)

        assert slice_.content == "[detached]"

    def test_clone(self):
        """Test cloning a slice."""
        slice_ = MessageSlice(
            type="other",
            start=0,
            stop=5,
            metadata={"key": "value"},
        )

        cloned = slice_.clone()

        assert cloned.type == slice_.type
        assert cloned.start == slice_.start
        assert cloned.stop == slice_.stop
        assert cloned.metadata == slice_.metadata
        assert cloned is not slice_


# ==============================================================================
# ContentTypes Tests
# ==============================================================================


class TestContentText:
    """Tests for ContentText class."""

    def test_basic_creation(self):
        """Test creating ContentText."""
        content = ContentText(text="Hello")

        assert content.type == "text"
        assert content.text == "Hello"

    def test_str(self):
        """Test string representation."""
        content = ContentText(text="Hello")

        assert str(content) == "Hello"


class TestContentImageUrl:
    """Tests for ContentImageUrl class."""

    def test_from_url(self):
        """Test creating from URL."""
        content = ContentImageUrl.from_url("https://example.com/image.png")

        assert content.type == "image_url"
        assert content.image_url.url == "https://example.com/image.png"

    def test_from_bytes(self):
        """Test creating from bytes."""
        data = b"fake image data"
        content = ContentImageUrl.from_bytes(data, "image/png")

        assert content.type == "image_url"
        assert "base64" in content.image_url.url

    def test_to_bytes(self):
        """Test converting back to bytes."""
        import base64

        original = b"fake image data"
        content = ContentImageUrl.from_bytes(original, "image/png")
        recovered = content.to_bytes()

        assert recovered == original

    def test_to_bytes_non_base64_raises(self):
        """Test to_bytes raises for non-base64 URL."""
        content = ContentImageUrl.from_url("https://example.com/image.png")

        with pytest.raises(ValueError, match="not base64"):
            content.to_bytes()


class TestContentAudioInput:
    """Tests for ContentAudioInput class."""

    def test_from_bytes(self):
        """Test creating from bytes."""
        data = b"fake audio data"
        content = ContentAudioInput.from_bytes(data, format="wav")

        assert content.type == "input_audio"
        assert content.input_audio.format == "wav"

    def test_to_bytes(self):
        """Test converting back to bytes."""
        original = b"fake audio data"
        content = ContentAudioInput.from_bytes(original, format="wav")
        recovered = content.to_bytes()

        assert recovered == original

    def test_transcript_property(self):
        """Test transcript property."""
        content = ContentAudioInput.from_bytes(
            b"data",
            format="wav",
            transcript="Hello world",
        )

        assert content.transcript == "Hello world"


# ==============================================================================
# GenerateParams Tests
# ==============================================================================


class TestGenerateParams:
    """Tests for GenerateParams class."""

    def test_basic_creation(self):
        """Test creating basic params."""
        params = GenerateParams(temperature=0.7, max_tokens=100)

        assert params.temperature == 0.7
        assert params.max_tokens == 100

    def test_merge_with_other(self):
        """Test merging with another params object."""
        params1 = GenerateParams(temperature=0.7)
        params2 = GenerateParams(max_tokens=100)

        merged = params1.merge_with(params2)

        assert merged.temperature == 0.7
        assert merged.max_tokens == 100

    def test_merge_with_override(self):
        """Test that merge overrides existing values."""
        params1 = GenerateParams(temperature=0.7)
        params2 = GenerateParams(temperature=0.9)

        merged = params1.merge_with(params2)

        assert merged.temperature == 0.9

    def test_merge_with_none(self):
        """Test merging with None."""
        params = GenerateParams(temperature=0.7)

        merged = params.merge_with(None)

        assert merged.temperature == 0.7

    def test_merge_with_multiple(self):
        """Test merging with multiple params."""
        params1 = GenerateParams(temperature=0.7)
        params2 = GenerateParams(max_tokens=100)
        params3 = GenerateParams(top_p=0.9)

        merged = params1.merge_with(params2, params3)

        assert merged.temperature == 0.7
        assert merged.max_tokens == 100
        assert merged.top_p == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        params = GenerateParams(temperature=0.7, max_tokens=100)

        result = params.to_dict()

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert "extra" not in result  # extra should be merged

    def test_to_dict_with_extra(self):
        """Test that extra params are merged into dict."""
        params = GenerateParams(temperature=0.7, extra={"custom": "value"})

        result = params.to_dict()

        assert result["custom"] == "value"

    def test_clone(self):
        """Test cloning params."""
        params = GenerateParams(temperature=0.7)
        cloned = params.clone()

        assert cloned.temperature == params.temperature
        assert cloned is not params

    def test_hash(self):
        """Test params are hashable."""
        params1 = GenerateParams(temperature=0.7)
        params2 = GenerateParams(temperature=0.7)
        params3 = GenerateParams(temperature=0.9)

        assert hash(params1) == hash(params2)
        assert hash(params1) != hash(params3)

    def test_stop_validator_string(self):
        """Test stop sequences from string."""
        params = GenerateParams(stop="a;b;c")

        assert params.stop == ["a", "b", "c"]

    def test_stop_validator_list(self):
        """Test stop sequences from list."""
        params = GenerateParams(stop=["a", "b", "c"])

        assert params.stop == ["a", "b", "c"]


# ==============================================================================
# Usage Tests
# ==============================================================================


class TestUsage:
    """Tests for Usage class."""

    def test_basic_creation(self):
        """Test creating usage."""
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_add(self):
        """Test adding two usage objects."""
        usage1 = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        usage2 = Usage(input_tokens=20, output_tokens=10, total_tokens=30)

        result = usage1 + usage2

        assert result.input_tokens == 30
        assert result.output_tokens == 15
        assert result.total_tokens == 45

    def test_add_type_error(self):
        """Test adding non-Usage raises TypeError."""
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)

        with pytest.raises(TypeError):
            usage + 5

    def test_str(self):
        """Test string representation."""
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)

        result = str(usage)

        assert "10" in result
        assert "5" in result
        assert "15" in result


# ==============================================================================
# GeneratedMessage Tests
# ==============================================================================


class TestGeneratedMessage:
    """Tests for GeneratedMessage class."""

    def test_basic_creation(self):
        """Test creating generated message."""
        msg = Message(role="assistant", content="Hello")
        gen = GeneratedMessage(message=msg, stop_reason="stop")

        assert gen.message.content == "Hello"
        assert gen.stop_reason == "stop"

    def test_str(self):
        """Test string representation."""
        msg = Message(role="assistant", content="Hello")
        gen = GeneratedMessage(message=msg)

        assert str(gen) == str(msg)

    def test_from_text(self):
        """Test creating from text."""
        gen = GeneratedMessage.from_text("Hello", stop_reason="stop")

        assert gen.message.content == "Hello"
        assert gen.message.role == "assistant"
        assert gen.stop_reason == "stop"


# ==============================================================================
# GeneratedText Tests
# ==============================================================================


class TestGeneratedText:
    """Tests for GeneratedText class."""

    def test_basic_creation(self):
        """Test creating generated text."""
        gen = GeneratedText(text="Hello", stop_reason="stop")

        assert gen.text == "Hello"
        assert gen.stop_reason == "stop"

    def test_str(self):
        """Test string representation."""
        gen = GeneratedText(text="Hello")

        assert str(gen) == "Hello"

    def test_from_text(self):
        """Test creating from text."""
        gen = GeneratedText.from_text("Hello", stop_reason="stop")

        assert gen.text == "Hello"
        assert gen.stop_reason == "stop"

    def test_to_generated_message(self):
        """Test converting to GeneratedMessage."""
        gen = GeneratedText(
            text="Hello",
            stop_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        result = gen.to_generated_message()

        assert isinstance(result, GeneratedMessage)
        assert result.message.content == "Hello"
        assert result.stop_reason == "stop"
        assert result.usage is not None


# ==============================================================================
# StopReason Tests
# ==============================================================================


class TestConvertStopReason:
    """Tests for convert_stop_reason function."""

    def test_stop_reasons(self):
        """Test various stop reason conversions."""
        assert convert_stop_reason("stop") == "stop"
        assert convert_stop_reason("eos") == "stop"
        assert convert_stop_reason("length") == "length"
        assert convert_stop_reason("model_length") == "length"
        assert convert_stop_reason("content_filter") == "content_filter"
        assert convert_stop_reason("tool_calls") == "tool_calls"
        assert convert_stop_reason("end_tool") == "tool_calls"
        assert convert_stop_reason("unknown_value") == "unknown"
        assert convert_stop_reason(None) == "unknown"


# ==============================================================================
# Chat Tests
# ==============================================================================


class TestChatCreation:
    """Tests for Chat creation."""

    def test_basic_creation(self):
        """Test creating a basic chat."""
        messages = [Message(role="user", content="Hello")]
        chat = Chat(messages=messages)

        assert len(chat.messages) == 1
        assert chat.messages[0].content == "Hello"

    def test_creation_with_generated(self):
        """Test creating chat with generated messages."""
        messages = [Message(role="user", content="Hello")]
        generated = [Message(role="assistant", content="Hi")]
        chat = Chat(messages=messages, generated=generated)

        assert len(chat.generated) == 1
        assert chat.generated[0].content == "Hi"

    def test_creation_from_dicts(self):
        """Test creating chat from message dicts."""
        messages = [{"role": "user", "content": "Hello"}]
        chat = Chat(messages=messages)

        assert chat.messages[0].content == "Hello"


class TestChatProperties:
    """Tests for Chat properties."""

    def test_len(self):
        """Test len returns total message count."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        assert len(chat) == 2

    def test_all_property(self):
        """Test all property returns all messages."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        assert len(chat.all) == 2

    def test_prev_property(self):
        """Test prev is alias for messages."""
        chat = Chat(messages=[Message(role="user", content="Hello")])

        assert chat.prev is chat.messages

    def test_next_property(self):
        """Test next is alias for generated."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        assert chat.next is chat.generated

    def test_last_property(self):
        """Test last returns last message."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        assert chat.last.content == "Hi"

    def test_conversation_property(self):
        """Test conversation returns string representation."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        result = chat.conversation

        assert "Hello" in result
        assert "Hi" in result


class TestChatMethods:
    """Tests for Chat methods."""

    def test_meta(self):
        """Test meta method."""
        chat = Chat(messages=[Message(role="user", content="Hello")])
        result = chat.meta(key="value")

        assert result is chat
        assert chat.metadata["key"] == "value"

    def test_apply(self):
        """Test apply on last message."""
        chat = Chat(
            messages=[Message(role="user", content="Hello $name!")],
        )
        chat.apply(name="World")

        assert chat.messages[-1].content == "Hello World!"

    def test_apply_to_all(self):
        """Test apply to all messages."""
        chat = Chat(
            messages=[
                Message(role="user", content="Hello $name!"),
                Message(role="assistant", content="Hi $name!"),
            ],
        )
        chat.apply_to_all(name="World")

        assert "World" in chat.messages[0].content
        assert "World" in chat.messages[1].content

    def test_inject_system_content(self):
        """Test inject_system_content method."""
        chat = Chat(messages=[Message(role="user", content="Hello")])
        chat.inject_system_content("System prompt")

        assert chat.messages[0].role == "system"
        assert "System prompt" in chat.messages[0].content

    def test_message_slices(self):
        """Test getting slices across all messages."""
        msg1 = Message(role="user", content="Hello World")
        msg1.mark_slice("Hello", "other")
        msg2 = Message(role="assistant", content="Hi There")
        msg2.mark_slice("Hi", "other")

        chat = Chat(messages=[msg1, msg2])
        slices = chat.message_slices()

        assert len(slices) == 2

    def test_to_openai(self):
        """Test to_openai conversion."""
        chat = Chat(
            messages=[Message(role="user", content="Hello")],
            generated=[Message(role="assistant", content="Hi")],
        )

        result = chat.to_openai()

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


# ==============================================================================
# ChatList Tests
# ==============================================================================


class TestChatList:
    """Tests for ChatList class."""

    def test_to_json(self):
        """Test to_json method."""
        chat = Chat(messages=[Message(role="user", content="Hello")])
        chat_list = ChatList([chat])

        result = chat_list.to_json()

        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_to_openai(self):
        """Test to_openai method."""
        chat = Chat(messages=[Message(role="user", content="Hello")])
        chat_list = ChatList([chat])

        result = chat_list.to_openai()

        assert len(result) == 1
        assert len(result[0]) == 1


# ==============================================================================
# Caching Tests
# ==============================================================================


class TestCaching:
    """Tests for caching module."""

    def test_apply_cache_mode_none(self):
        """Test that None mode returns messages unchanged."""
        msg = Message(role="user", content="Hello")
        messages = [[msg]]

        result = apply_cache_mode_to_messages(None, messages)

        assert result is messages

    def test_apply_cache_mode_latest(self):
        """Test latest cache mode."""
        messages = [[
            Message(role="system", content="System"),
            Message(role="user", content="User1"),
            Message(role="assistant", content="Assistant"),
            Message(role="user", content="User2"),
        ]]

        result = apply_cache_mode_to_messages("latest", messages)

        # Last 2 non-assistant messages should have cache_control
        # System and User2 (User1 and Assistant don't count towards "latest 2")
        non_assistant = [m for m in result[0] if m.role != "assistant"]
        # The last 2 should have cache_control
        assert non_assistant[-1].content_parts[-1].cache_control is not None
        assert non_assistant[-2].content_parts[-1].cache_control is not None

    def test_apply_cache_mode_clears_existing(self):
        """Test that existing cache settings are cleared first."""
        msg = Message(role="user", content="Hello", cache_control="ephemeral")
        messages = [[msg]]

        # Verify it had cache control
        assert msg.content_parts[-1].cache_control is not None

        result = apply_cache_mode_to_messages("latest", messages)

        # After applying, it should still have cache control (it's the latest)
        assert result[0][0].content_parts[-1].cache_control is not None


# ==============================================================================
# Generator Registration Tests
# ==============================================================================


class TestGeneratorRegistration:
    """Tests for generator registration."""

    def test_register_generator(self):
        """Test registering a generator."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        register_generator("mock", MockGenerator)

        gen = get_generator("mock!test-model")

        assert isinstance(gen, MockGenerator)
        assert gen.model == "test-model"

    def test_get_generator_with_params(self):
        """Test getting generator with params in identifier."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        register_generator("mock2", MockGenerator)

        gen = get_generator("mock2!test-model,temperature=0.7,max_tokens=100")

        assert gen.params.temperature == 0.7
        assert gen.params.max_tokens == 100


# ==============================================================================
# Fixup Tests
# ==============================================================================


class TestFixup:
    """Tests for Fixup system."""

    @pytest.mark.asyncio
    async def test_fixup_decorator(self):
        """Test with_fixups decorator."""

        class TestFixup(Fixup):
            def __init__(self):
                self.applied = False

            def can_fix(self, exception: Exception) -> bool:
                return isinstance(exception, ValueError)

            def fix(self, messages):
                self.applied = True
                return [Message(role="user", content="fixed")]

        fixup = TestFixup()
        call_count = 0

        @with_fixups(fixup)
        async def test_func(self, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and messages[0].content != "fixed":
                raise ValueError("Need fix")
            return messages

        messages = [Message(role="user", content="original")]
        result = await test_func(None, messages)

        assert fixup.applied
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_fixup_once(self):
        """Test fixup with 'once' return value."""

        class OnceFixup(Fixup):
            def __init__(self):
                self.fix_count = 0

            def can_fix(self, exception: Exception):
                return "once"

            def fix(self, messages):
                self.fix_count += 1
                return messages

        fixup = OnceFixup()
        call_count = 0

        @with_fixups(fixup)
        async def test_func(self, messages):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Error")
            return messages

        messages = [Message(role="user", content="test")]

        # First call will fail, trigger once fixup, and then succeed on retry
        # But since once fixup is removed after success, third failure won't be fixed
        with pytest.raises(ValueError):
            await test_func(None, messages)


# ==============================================================================
# Generator Base Class Tests
# ==============================================================================


class TestGenerator:
    """Tests for Generator base class."""

    def test_to_identifier(self):
        """Test to_identifier method."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        register_generator("mockgen", MockGenerator)
        gen = MockGenerator(model="my-model", params=GenerateParams())

        identifier = gen.to_identifier(short=True)

        assert "my-model" in identifier

    def test_watch(self):
        """Test watch method adds callbacks."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())
        callback = MagicMock()

        result = gen.watch(callback)

        assert result is gen
        assert callback in gen._watch_callbacks

    def test_watch_no_duplicates(self):
        """Test watch doesn't add duplicates by default."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())
        callback = MagicMock()

        gen.watch(callback)
        gen.watch(callback)

        assert len(gen._watch_callbacks) == 1

    def test_watch_allow_duplicates(self):
        """Test watch can allow duplicates."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())
        callback = MagicMock()

        gen.watch(callback, allow_duplicates=True)
        gen.watch(callback, allow_duplicates=True)

        assert len(gen._watch_callbacks) == 2

    def test_load_unload(self):
        """Test load and unload return self."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())

        assert gen.load() is gen
        assert gen.unload() is gen

    def test_wrap(self):
        """Test wrap method."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())
        wrapper = lambda f: f

        result = gen.wrap(wrapper)

        assert result is gen
        assert gen._wrap is wrapper

    @pytest.mark.asyncio
    async def test_supports_function_calling_default(self):
        """Test default supports_function_calling returns None."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())

        result = await gen.supports_function_calling()

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_messages_not_implemented(self):
        """Test generate_messages raises NotImplementedError."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())

        with pytest.raises(NotImplementedError):
            await gen.generate_messages([[]], [GenerateParams()])

    @pytest.mark.asyncio
    async def test_generate_texts_not_implemented(self):
        """Test generate_texts raises NotImplementedError."""

        class MockGenerator(Generator):
            model: str = "test"
            params: GenerateParams = GenerateParams()

        gen = MockGenerator(model="test", params=GenerateParams())

        with pytest.raises(NotImplementedError):
            await gen.generate_texts(["test"], [GenerateParams()])
