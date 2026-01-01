"""Tests for Trajectory scores functionality."""

import pytest

from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.agents.events import GenerationStep
from dreadnode.core.generators.message import Message
from dreadnode.core.generators.generator import Usage
from dreadnode.core.metric import MetricSeries


class TestTrajectoryScores:
    """Tests for Trajectory.scores field and log_score method."""

    def test_scores_initialized_empty(self):
        """Test that scores is initialized as empty dict."""
        traj = Trajectory()

        assert traj.scores == {}
        assert isinstance(traj.scores, dict)

    def test_log_score_creates_series(self):
        """Test that log_score creates a MetricSeries."""
        traj = Trajectory()

        traj.log_score("quality", 0.8, step=1)

        assert "quality" in traj.scores
        assert isinstance(traj.scores["quality"], MetricSeries)
        assert traj.scores["quality"].count() == 1

    def test_log_score_appends_to_series(self):
        """Test that log_score appends to existing series."""
        traj = Trajectory()

        traj.log_score("quality", 0.8, step=1)
        traj.log_score("quality", 0.9, step=2)
        traj.log_score("quality", 0.85, step=3)

        assert traj.scores["quality"].count() == 3
        assert traj.scores["quality"].values == [0.8, 0.9, 0.85]

    def test_log_score_multiple_scorers(self):
        """Test logging scores from multiple scorers."""
        traj = Trajectory()

        traj.log_score("quality", 0.8, step=1)
        traj.log_score("safety", 0.95, step=1)
        traj.log_score("quality", 0.85, step=2)
        traj.log_score("safety", 0.9, step=2)

        assert "quality" in traj.scores
        assert "safety" in traj.scores
        assert traj.scores["quality"].count() == 2
        assert traj.scores["safety"].count() == 2

    def test_log_score_default_step(self):
        """Test that step defaults to current step count."""
        traj = Trajectory()

        # Add some steps first
        traj.log_step(GenerationStep(
            step=1,
            messages=[Message(role="assistant", content="Hello")],
        ))
        traj.log_step(GenerationStep(
            step=2,
            messages=[Message(role="assistant", content="World")],
        ))

        # Log score without explicit step
        traj.log_score("quality", 0.8)

        # Should use len(steps) = 2 as the step
        assert traj.scores["quality"].steps[0] == 2


class TestMetricSeriesAggregations:
    """Tests for MetricSeries aggregation methods via Trajectory."""

    @pytest.fixture
    def trajectory_with_scores(self):
        """Create a trajectory with multiple scores."""
        traj = Trajectory()
        traj.log_score("quality", 0.7, step=1)
        traj.log_score("quality", 0.8, step=2)
        traj.log_score("quality", 0.9, step=3)
        traj.log_score("quality", 0.6, step=4)
        return traj

    def test_mean(self, trajectory_with_scores):
        """Test mean aggregation."""
        mean = trajectory_with_scores.scores["quality"].mean()

        assert mean == pytest.approx(0.75, rel=0.01)

    def test_min(self, trajectory_with_scores):
        """Test min aggregation."""
        min_val = trajectory_with_scores.scores["quality"].min()

        assert min_val == 0.6

    def test_max(self, trajectory_with_scores):
        """Test max aggregation."""
        max_val = trajectory_with_scores.scores["quality"].max()

        assert max_val == 0.9

    def test_sum(self, trajectory_with_scores):
        """Test sum aggregation."""
        total = trajectory_with_scores.scores["quality"].sum()

        assert total == pytest.approx(3.0, rel=0.01)

    def test_count(self, trajectory_with_scores):
        """Test count aggregation."""
        count = trajectory_with_scores.scores["quality"].count()

        assert count == 4

    def test_last(self, trajectory_with_scores):
        """Test last value."""
        last = trajectory_with_scores.scores["quality"].last()

        assert last == 0.6

    def test_first(self, trajectory_with_scores):
        """Test first value."""
        first = trajectory_with_scores.scores["quality"].first()

        assert first == 0.7


class TestTrajectoryWithStepsAndScores:
    """Tests for Trajectory with both steps and scores."""

    def test_steps_and_scores_together(self):
        """Test that steps and scores work together."""
        traj = Trajectory()

        # Log a step
        step1 = GenerationStep(
            step=1,
            messages=[Message(role="assistant", content="Hello")],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        traj.log_step(step1)

        # Log a score for that step
        traj.log_score("quality", 0.8, step=1)

        # Log another step
        step2 = GenerationStep(
            step=2,
            messages=[Message(role="assistant", content="World")],
            usage=Usage(input_tokens=15, output_tokens=8, total_tokens=23),
        )
        traj.log_step(step2)

        # Log a score for that step
        traj.log_score("quality", 0.9, step=2)

        assert len(traj.steps) == 2
        assert traj.scores["quality"].count() == 2
        assert traj.usage.total_tokens == 38

    def test_empty_scores_aggregation(self):
        """Test aggregating empty scores."""
        traj = Trajectory()

        # No scores logged
        assert traj.scores == {}


class TestTrajectoryScoresSerialization:
    """Tests for Trajectory scores in serialization context."""

    def test_scores_in_model_dump(self):
        """Test that scores are included in model dump."""
        traj = Trajectory()
        traj.log_score("quality", 0.8, step=1)

        data = traj.model_dump()

        assert "scores" in data
        assert "quality" in data["scores"]

    def test_empty_scores_in_model_dump(self):
        """Test that empty scores serialize correctly."""
        traj = Trajectory()

        data = traj.model_dump()

        assert "scores" in data
        assert data["scores"] == {}
