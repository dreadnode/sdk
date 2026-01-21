"""Tests for preserve_tool_pairs functionality in summarize_when_long hook."""

import rigging as rg
from dreadnode.agent.hooks.summarize import _find_tool_aware_boundary


class ToolCall:
    """Minimal tool call representation for testing."""
    def __init__(self, call_id: str):
        self.id = call_id


class ToolMessage(rg.Message):
    """Tool response message for testing."""
    def __init__(self, call_id: str, content: str):
        super().__init__("tool", content)
        self.tool_call_id = call_id


def test_preserves_tool_pairs():
    """Tool call and response stay together when split."""
    messages = [
        rg.Message("user", "Hello"),
        rg.Message("assistant", "Let me check", tool_calls=[ToolCall("call_1")]),
        ToolMessage("call_1", "Result"),
        rg.Message("assistant", "Done"),
        rg.Message("user", "Thanks"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)

    # Should keep tool pair together by moving boundary earlier
    assert boundary <= 1, "Boundary should preserve tool call/response pair"


def test_no_tools():
    """Works correctly without any tool messages."""
    messages = [
        rg.Message("user", "Hello"),
        rg.Message("assistant", "Hi"),
        rg.Message("user", "How are you"),
        rg.Message("assistant", "Good"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)
    assert boundary == 2, "Should split at natural boundary"


def test_multiple_tool_pairs():
    """Handles multiple tool call/response pairs correctly."""
    messages = [
        rg.Message("user", "Do A and B"),
        rg.Message("assistant", "Running A", tool_calls=[ToolCall("a")]),
        ToolMessage("a", "A done"),
        rg.Message("assistant", "Running B", tool_calls=[ToolCall("b")]),
        ToolMessage("b", "B done"),
        rg.Message("user", "Thanks"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)

    # Should not split between any tool pairs
    kept = messages[boundary:]
    assert len(kept) >= 2, "Should keep minimum messages"


def test_no_valid_boundary():
    """Returns 0 when entire conversation is tool chain."""
    messages = [
        rg.Message("assistant", "Start", tool_calls=[ToolCall("1")]),
        ToolMessage("1", "Result 1"),
        rg.Message("assistant", "Continue", tool_calls=[ToolCall("2")]),
        ToolMessage("2", "Result 2"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)
    assert boundary == 0, "Should keep everything when no valid split exists"


if __name__ == "__main__":
    test_preserves_tool_pairs()
    test_no_tools()
    test_multiple_tool_pairs()
    test_no_valid_boundary()
    print("All tests passed")
