"""Tests for preserve_tool_pairs functionality in summarize_when_long hook."""

import rigging as rg
from dreadnode.agent.hooks.summarize import _find_tool_aware_boundary


def test_preserves_tool_pairs():
    """Tool call and response stay together when split."""
    messages = [
        rg.Message("user", "Hello"),
        rg.Message(
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "check", "arguments": "{}"}}],
        ),
        rg.Message("tool", "Result", tool_call_id="call_1"),
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
        rg.Message(
            "assistant",
            "Running A",
            tool_calls=[{"id": "a", "type": "function", "function": {"name": "run_a", "arguments": "{}"}}],
        ),
        rg.Message("tool", "A done", tool_call_id="a"),
        rg.Message(
            "assistant",
            "Running B",
            tool_calls=[{"id": "b", "type": "function", "function": {"name": "run_b", "arguments": "{}"}}],
        ),
        rg.Message("tool", "B done", tool_call_id="b"),
        rg.Message("user", "Thanks"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)

    # Should not split between any tool pairs
    kept = messages[boundary:]
    assert len(kept) >= 2, "Should keep minimum messages"


def test_no_valid_boundary():
    """Returns 0 when entire conversation is tool chain."""
    messages = [
        rg.Message(
            "assistant",
            "Start",
            tool_calls=[{"id": "1", "type": "function", "function": {"name": "start", "arguments": "{}"}}],
        ),
        rg.Message("tool", "Result 1", tool_call_id="1"),
        rg.Message(
            "assistant",
            "Continue",
            tool_calls=[{"id": "2", "type": "function", "function": {"name": "continue", "arguments": "{}"}}],
        ),
        rg.Message("tool", "Result 2", tool_call_id="2"),
    ]

    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=2)
    assert boundary == 0, "Should keep everything when no valid split exists"


if __name__ == "__main__":
    test_preserves_tool_pairs()
    test_no_tools()
    test_multiple_tool_pairs()
    test_no_valid_boundary()
    print("All tests passed")
