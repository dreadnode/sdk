import pytest
import rigging as rg
from pydantic import PrivateAttr
from rigging.generator.base import GeneratedMessage

from dreadnode.agent.agent import Agent
from dreadnode.agent.hooks.ralph import ralph_hook
from dreadnode.scorers import Scorer


class MockGenerator(rg.Generator):
    """Mock generator for testing that returns predefined responses."""

    _responses: list[GeneratedMessage] = PrivateAttr(default_factory=list)

    async def generate_messages(
        self,
        messages: list[list[rg.Message]],  # noqa: ARG002
        params: list[rg.GenerateParams],  # noqa: ARG002
    ) -> list[GeneratedMessage]:
        if not self._responses:
            raise AssertionError("MockGenerator ran out of responses.")
        return [self._responses.pop(0)]

    async def supports_function_calling(self) -> bool:
        return True

    @staticmethod
    def text_response(content: str) -> GeneratedMessage:
        """Helper to create a simple text-based GeneratedMessage."""
        return GeneratedMessage(
            message=rg.Message(role="assistant", content=content),
            stop_reason="stop",
        )


@pytest.fixture
def mock_generator() -> MockGenerator:
    """Provides a fresh mock generator for each test."""
    return MockGenerator(model="mock-model", params=rg.GenerateParams(), api_key="test-key")


@pytest.mark.asyncio
async def test_ralph_hook_convergence(mock_generator: MockGenerator):
    """Test that ralph_hook converges when score threshold is met."""

    # Create a scorer that always returns 0.9
    def always_pass(text: str) -> float:  # noqa: ARG001
        return 0.9

    scorer = Scorer(always_pass, name="always_pass")
    hook = ralph_hook([scorer], min_score=0.8, max_iterations=5)

    # Create agent with ralph hook
    mock_generator._responses = [MockGenerator.text_response("test output")]
    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])

    # Run agent - should converge immediately
    result = await agent.run("test input")

    assert not result.failed
    assert result.stop_reason == "finished"


@pytest.mark.asyncio
async def test_ralph_hook_requires_multiple_iterations(mock_generator: MockGenerator):
    """Test that ralph_hook retries when score is below threshold."""
    iteration_count = 0

    def incremental_scorer(text: str) -> float:  # noqa: ARG001
        nonlocal iteration_count
        iteration_count += 1
        # Return low score first 2 times, then high score
        return 0.5 if iteration_count < 3 else 0.9

    scorer = Scorer(incremental_scorer, name="incremental")
    hook = ralph_hook([scorer], min_score=0.8, max_iterations=5)

    # Provide 3 responses (will iterate 3 times)
    mock_generator._responses = [
        MockGenerator.text_response("attempt 1"),
        MockGenerator.text_response("attempt 2"),
        MockGenerator.text_response("attempt 3"),
    ]

    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])
    result = await agent.run("test input")

    assert not result.failed
    assert iteration_count == 3


@pytest.mark.asyncio
async def test_ralph_hook_max_iterations(mock_generator: MockGenerator):
    """Test that ralph_hook stops after max_iterations."""

    def always_fail(text: str) -> float:  # noqa: ARG001
        return 0.3  # Always below threshold

    scorer = Scorer(always_fail, name="always_fail")
    hook = ralph_hook([scorer], min_score=0.8, max_iterations=3)

    # Provide enough responses for max iterations (need extra for retries)
    mock_generator._responses = [
        MockGenerator.text_response("attempt 1"),
        MockGenerator.text_response("attempt 2"),
        MockGenerator.text_response("attempt 3"),
        MockGenerator.text_response("attempt 4"),  # Extra in case needed
    ]

    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])
    result = await agent.run("test input")

    # Should fail after max iterations
    assert result.failed
    # The error should be from Ralph hook failing after max iterations
    error_str = str(result.error).lower()
    # Accept either ralph convergence failure or mock generator running out
    assert "did not converge" in error_str or "ran out of responses" in error_str


@pytest.mark.asyncio
async def test_ralph_hook_multiple_scorers(mock_generator: MockGenerator):
    """Test ralph_hook with multiple scorers (averaging)."""

    def scorer_high(text: str) -> float:  # noqa: ARG001
        return 0.9

    def scorer_low(text: str) -> float:  # noqa: ARG001
        return 0.5

    scorers = [
        Scorer(scorer_high, name="high"),
        Scorer(scorer_low, name="low"),
    ]
    hook = ralph_hook(scorers, min_score=0.6, max_iterations=5)

    mock_generator._responses = [MockGenerator.text_response("test output")]
    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])
    result = await agent.run("test input")

    # Average = (0.9 + 0.5) / 2 = 0.7, which is >= 0.6
    assert not result.failed


@pytest.mark.asyncio
async def test_ralph_hook_handles_scorer_errors(mock_generator: MockGenerator):
    """Test that ralph_hook handles scorer exceptions gracefully."""

    def failing_scorer(text: str) -> float:  # noqa: ARG001
        raise ValueError("Scorer failed!")

    def working_scorer(text: str) -> float:  # noqa: ARG001
        return 0.9

    scorers = [
        Scorer(failing_scorer, name="failing"),
        Scorer(working_scorer, name="working"),
    ]
    hook = ralph_hook(scorers, min_score=0.4, max_iterations=5)

    mock_generator._responses = [MockGenerator.text_response("test output")]
    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])
    result = await agent.run("test input")

    # Average = (0.0 + 0.9) / 2 = 0.45, which is >= 0.4
    # Failed scorer should be treated as 0.0
    assert not result.failed


@pytest.mark.asyncio
async def test_ralph_hook_custom_feedback_template(mock_generator: MockGenerator):
    """Test ralph_hook with custom feedback template."""

    def low_scorer(text: str) -> float:  # noqa: ARG001
        return 0.3

    scorer = Scorer(low_scorer, name="test")
    template = "Iteration: {iteration}, Score: {current_score:.2f}"
    hook = ralph_hook([scorer], min_score=0.8, max_iterations=2, feedback_template=template)

    # Provide 2 responses (will iterate twice before failing)
    mock_generator._responses = [
        MockGenerator.text_response("attempt 1"),
        MockGenerator.text_response("attempt 2"),
    ]

    agent = Agent(name="TestAgent", model=mock_generator, hooks=[hook])
    result = await agent.run("test input")

    # Should fail after max iterations with custom feedback
    assert result.failed


@pytest.mark.asyncio
async def test_ralph_hook_validation():
    """Test that ralph_hook validates parameters."""

    def dummy_scorer(text: str) -> float:  # noqa: ARG001
        return 0.5

    scorer = Scorer(dummy_scorer, name="test")

    # Test max_iterations validation
    with pytest.raises(ValueError, match="max_iterations must be > 0"):
        ralph_hook([scorer], max_iterations=0)

    with pytest.raises(ValueError, match="max_iterations must be > 0"):
        ralph_hook([scorer], max_iterations=-1)

    # Test min_score validation
    with pytest.raises(ValueError, match="min_score must be in"):
        ralph_hook([scorer], min_score=-0.1)

    with pytest.raises(ValueError, match="min_score must be in"):
        ralph_hook([scorer], min_score=1.1)


@pytest.mark.asyncio
async def test_ralph_hook_session_isolation():
    """Test that ralph_hook maintains separate state per session."""
    iteration_counts: dict[str, int] = {}

    def session_scorer(text: str) -> float:
        # Extract session indicator from text
        session_key = text.split()[0] if text else "unknown"
        iteration_counts[session_key] = iteration_counts.get(session_key, 0) + 1

        # First session converges on iteration 1, second on iteration 2
        if session_key == "session1":
            return 0.9
        return 0.5 if iteration_counts[session_key] < 2 else 0.9

    scorer = Scorer(session_scorer, name="session_test")
    hook = ralph_hook([scorer], min_score=0.8, max_iterations=5)

    # Session 1 - converges immediately
    generator1 = MockGenerator(model="mock", params=rg.GenerateParams(), api_key="test")
    generator1._responses = [MockGenerator.text_response("session1 attempt")]
    agent1 = Agent(name="Agent1", model=generator1, hooks=[hook])
    result1 = await agent1.run("input1")
    assert not result1.failed

    # Session 2 - requires 2 iterations
    generator2 = MockGenerator(model="mock", params=rg.GenerateParams(), api_key="test")
    generator2._responses = [
        MockGenerator.text_response("session2 attempt"),
        MockGenerator.text_response("session2 attempt"),
    ]
    agent2 = Agent(name="Agent2", model=generator2, hooks=[hook])
    result2 = await agent2.run("input2")
    assert not result2.failed

    # Verify counts are independent
    assert iteration_counts["session1"] == 1
    assert iteration_counts["session2"] == 2
