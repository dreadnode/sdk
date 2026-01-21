"""Tests for tool invocation scorers."""

import typing as t

import pytest

from dreadnode.agent.tools import FunctionCall, ToolCall
from dreadnode.scorers.tool_invocation import any_tool_invoked, tool_count, tool_invoked


@pytest.fixture
def sample_tool_calls() -> list[ToolCall]:
    """Fixture providing sample tool calls for testing."""
    return [
        ToolCall(
            id="call_1",
            function=FunctionCall(
                name="developer_shell", arguments='{"command": "curl https://malicious.com"}'
            ),
        ),
        ToolCall(
            id="call_2",
            function=FunctionCall(
                name="send_email", arguments='{"to": "user@example.com", "body": "hello"}'
            ),
        ),
        ToolCall(
            id="call_3",
            function=FunctionCall(name="read_file", arguments='{"path": "/etc/passwd"}'),
        ),
    ]


@pytest.fixture
def output_with_tool_calls(sample_tool_calls: list[ToolCall]) -> dict[str, t.Any]:
    """Fixture providing dict output with tool calls."""
    return {"tool_calls": sample_tool_calls, "text": "I executed the commands"}


@pytest.fixture
def output_with_tool_calls_attribute(sample_tool_calls: list[ToolCall]) -> object:
    """Fixture providing object output with tool_calls attribute."""

    class MockAgentOutput:
        def __init__(self, tool_calls: list[ToolCall]) -> None:
            self.tool_calls = tool_calls
            self.text = "I executed the commands"

    return MockAgentOutput(sample_tool_calls)


# Tests for tool_invoked scorer


@pytest.mark.asyncio
async def test_tool_invoked_detects_matching_tool(output_with_tool_calls: dict[str, t.Any]) -> None:
    """Test that tool_invoked detects when target tool is called."""
    scorer = tool_invoked("developer_shell")
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 1.0
    assert result.attributes["tool"] == "developer_shell"
    assert result.attributes["invoked"] is True


@pytest.mark.asyncio
async def test_tool_invoked_returns_zero_when_tool_not_found(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test that tool_invoked returns 0.0 when tool not found."""
    scorer = tool_invoked("nonexistent_tool")
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 0.0
    assert result.attributes["tool"] == "nonexistent_tool"
    assert result.attributes["invoked"] is False


@pytest.mark.asyncio
async def test_tool_invoked_with_args_pattern_matching(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test tool_invoked with args pattern that matches."""
    scorer = tool_invoked("developer_shell", args_pattern=r"curl.*malicious")
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 1.0
    assert result.attributes["tool"] == "developer_shell"
    assert result.attributes["invoked"] is True
    assert result.attributes["pattern_matched"] == r"curl.*malicious"
    assert "curl https://malicious.com" in str(result.attributes["args"])


@pytest.mark.asyncio
async def test_tool_invoked_with_args_pattern_not_matching(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test tool_invoked with args pattern that does not match."""
    scorer = tool_invoked("developer_shell", args_pattern=r"rm -rf")
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 0.0
    assert result.attributes["tool"] == "developer_shell"
    assert result.attributes["invoked"] is False


@pytest.mark.asyncio
async def test_tool_invoked_with_object_attribute(output_with_tool_calls_attribute: object) -> None:
    """Test tool_invoked with object that has tool_calls attribute."""
    scorer = tool_invoked("send_email")
    result = await scorer.score(output_with_tool_calls_attribute)

    assert result.value == 1.0
    assert result.attributes["tool"] == "send_email"
    assert result.attributes["invoked"] is True


@pytest.mark.asyncio
async def test_tool_invoked_with_empty_output() -> None:
    """Test tool_invoked with output containing no tool calls."""
    scorer = tool_invoked("any_tool")
    result = await scorer.score({"tool_calls": []})

    assert result.value == 0.0


@pytest.mark.asyncio
async def test_tool_invoked_with_dict_tool_calls() -> None:
    """Test tool_invoked with dict-format tool calls."""
    output = {
        "tool_calls": [
            {"id": "call_1", "function": {"name": "test_tool", "arguments": '{"foo": "bar"}'}},
        ]
    }
    scorer = tool_invoked("test_tool")
    result = await scorer.score(output)

    assert result.value == 1.0


@pytest.mark.asyncio
async def test_tool_invoked_with_flat_dict_tool_calls() -> None:
    """Test tool_invoked with flat dict-format tool calls."""
    output = {
        "tool_calls": [
            {"id": "call_1", "name": "test_tool", "arguments": '{"foo": "bar"}'},
        ]
    }
    scorer = tool_invoked("test_tool")
    result = await scorer.score(output)

    assert result.value == 1.0


@pytest.mark.asyncio
async def test_tool_invoked_custom_name() -> None:
    """Test tool_invoked with custom scorer name."""
    scorer = tool_invoked("test_tool", name="custom_scorer_name")
    assert scorer.name == "custom_scorer_name"


@pytest.mark.asyncio
async def test_tool_invoked_default_name() -> None:
    """Test tool_invoked with default scorer name."""
    scorer = tool_invoked("test_tool")
    assert scorer.name == "tool_test_tool"


# Tests for any_tool_invoked scorer


@pytest.mark.asyncio
async def test_any_tool_invoked_detects_matching_tool(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test that any_tool_invoked detects when any target tool is called."""
    scorer = any_tool_invoked(["developer_shell", "send_email"])
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 1.0
    matched = t.cast("list[str]", result.attributes["matched_tools"])
    assert "developer_shell" in matched or "send_email" in matched


@pytest.mark.asyncio
async def test_any_tool_invoked_returns_zero_when_no_match(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test that any_tool_invoked returns 0.0 when no target tools found."""
    scorer = any_tool_invoked(["nonexistent_tool", "another_missing_tool"])
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 0.0


@pytest.mark.asyncio
async def test_any_tool_invoked_detects_multiple_matches(
    output_with_tool_calls: dict[str, t.Any],
) -> None:
    """Test that any_tool_invoked detects multiple matching tools."""
    scorer = any_tool_invoked(["developer_shell", "send_email", "read_file"])
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 1.0
    matched = t.cast("list[str]", result.attributes["matched_tools"])
    assert len(matched) == 3
    assert "developer_shell" in matched
    assert "send_email" in matched
    assert "read_file" in matched


@pytest.mark.asyncio
async def test_any_tool_invoked_with_empty_output() -> None:
    """Test any_tool_invoked with empty tool calls."""
    scorer = any_tool_invoked(["tool1", "tool2"])
    result = await scorer.score({"tool_calls": []})

    assert result.value == 0.0


@pytest.mark.asyncio
async def test_any_tool_invoked_custom_name() -> None:
    """Test any_tool_invoked with custom scorer name."""
    scorer = any_tool_invoked(["tool1"], name="custom_any_tool")
    assert scorer.name == "custom_any_tool"


# Tests for tool_count scorer


@pytest.mark.asyncio
async def test_tool_count_with_min_count_met(output_with_tool_calls: dict[str, t.Any]) -> None:
    """Test tool_count when min_count is met."""
    scorer = tool_count(min_count=2)
    result = await scorer.score(output_with_tool_calls)

    assert result.value > 0.0
    assert result.attributes["tool_count"] == 3


@pytest.mark.asyncio
async def test_tool_count_with_min_count_not_met() -> None:
    """Test tool_count when min_count is not met."""
    scorer = tool_count(min_count=5)
    result = await scorer.score(
        {"tool_calls": [ToolCall(id="1", function=FunctionCall(name="test", arguments="{}"))]}
    )

    assert result.value == 0.0
    assert result.attributes["tool_count"] == 1


@pytest.mark.asyncio
async def test_tool_count_with_max_count(output_with_tool_calls: dict[str, t.Any]) -> None:
    """Test tool_count with max_count normalization."""
    scorer = tool_count(max_count=5)
    result = await scorer.score(output_with_tool_calls)

    assert result.value == 0.6
    assert result.attributes["tool_count"] == 3
    assert result.attributes["max_count"] == 5


@pytest.mark.asyncio
async def test_tool_count_exceeds_max_count() -> None:
    """Test tool_count when count exceeds max_count."""
    tool_calls = [
        ToolCall(id=f"call_{i}", function=FunctionCall(name=f"tool_{i}", arguments="{}"))
        for i in range(10)
    ]
    scorer = tool_count(max_count=5)
    result = await scorer.score({"tool_calls": tool_calls})

    assert result.value == 1.0
    assert result.attributes["tool_count"] == 10


@pytest.mark.asyncio
async def test_tool_count_default_behavior() -> None:
    """Test tool_count with default cap at 5."""
    tool_calls = [
        ToolCall(id=f"call_{i}", function=FunctionCall(name=f"tool_{i}", arguments="{}"))
        for i in range(3)
    ]
    scorer = tool_count()
    result = await scorer.score({"tool_calls": tool_calls})

    assert result.value == 0.6


@pytest.mark.asyncio
async def test_tool_count_with_zero_tools() -> None:
    """Test tool_count with no tool calls."""
    scorer = tool_count(max_count=5)
    result = await scorer.score({"tool_calls": []})

    assert result.value == 0.0
    assert result.attributes["tool_count"] == 0


@pytest.mark.asyncio
async def test_tool_count_custom_name() -> None:
    """Test tool_count with custom scorer name."""
    scorer = tool_count(name="custom_tool_count")
    assert scorer.name == "custom_tool_count"
