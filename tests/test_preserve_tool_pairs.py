"""Tests for preserve_tool_pairs functionality in summarize_when_long hook."""

import rigging as rg

from dreadnode.agent.hooks.summarize import _find_tool_aware_boundary


def test_preserves_tool_pairs():
    """Tool call and response stay together when boundary would split them."""
    messages = [
        rg.Message("user", "Hello"),
        rg.Message(
            "assistant",
            "Let me check",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check", "arguments": "{}"},
                }
            ],
        ),
        rg.Message("tool", "Result", tool_call_id="call_1"),
        rg.Message("assistant", "Done"),
        rg.Message("user", "Thanks"),
    ]

    # With min=3, naive boundary would be at index 2, keeping [2,3,4]
    # But that would orphan the tool response at index 2 (call at index 1)
    # So boundary should move back to index 1 to keep the pair together
    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=3)
    assert boundary == 1, f"Expected boundary 1 to keep tool pair together, got {boundary}"

    # Verify the kept messages include the complete tool pair
    kept = messages[boundary:]
    assert len(kept) == 4
    assert kept[0].role == "assistant"
    assert kept[1].role == "tool"


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
            tool_calls=[
                {"id": "a", "type": "function", "function": {"name": "run_a", "arguments": "{}"}}
            ],
        ),
        rg.Message("tool", "A done", tool_call_id="a"),
        rg.Message(
            "assistant",
            "Running B",
            tool_calls=[
                {"id": "b", "type": "function", "function": {"name": "run_b", "arguments": "{}"}}
            ],
        ),
        rg.Message("tool", "B done", tool_call_id="b"),
        rg.Message("user", "Thanks"),
    ]

    # With min=3, naive boundary at index 3 would keep [3,4,5]
    # But index 4 (tool response "b") references call at index 3
    # So boundary should move back to index 3 to keep the second pair
    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=3)
    assert boundary == 3, f"Expected boundary 3 to preserve second tool pair, got {boundary}"


def test_no_valid_boundary():
    """Returns 0 when min_messages would force splitting all tool pairs."""
    messages = [
        rg.Message(
            "assistant",
            "Start",
            tool_calls=[
                {"id": "1", "type": "function", "function": {"name": "start", "arguments": "{}"}}
            ],
        ),
        rg.Message("tool", "Result 1", tool_call_id="1"),
        rg.Message(
            "assistant",
            "Continue",
            tool_calls=[
                {"id": "2", "type": "function", "function": {"name": "continue", "arguments": "{}"}}
            ],
        ),
        rg.Message("tool", "Result 2", tool_call_id="2"),
    ]

    # With min=3, we'd need to keep last 3 messages
    # Any boundary would orphan at least one tool response
    # So should return 0 to keep everything
    boundary = _find_tool_aware_boundary(messages, min_messages_to_keep=3)
    assert boundary == 0, f"Should keep everything when no valid split exists, got {boundary}"


if __name__ == "__main__":
    test_preserves_tool_pairs()
    test_no_tools()
    test_multiple_tool_pairs()
    test_no_valid_boundary()
