"""Tests for the optimization module (Study/Trial)."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from dreadnode.core.optimization.trial import Trial, TrialStatus
from dreadnode.core.optimization.study import Study, current_trial, Direction, fit_objectives
from dreadnode.core.optimization.result import StudyResult
from dreadnode.core.optimization.events import (
    StudyStart,
    StudyEnd,
    TrialStart,
    TrialComplete,
    TrialAdded,
    TrialPruned,
    NewBestTrialFound,
)
from dreadnode.core.scorer import Scorer
from dreadnode.core.task import task
from dreadnode.core.search import Search


# ==============================================================================
# Trial Tests
# ==============================================================================


class TestTrialCreation:
    """Tests for Trial creation."""

    @pytest.mark.asyncio
    async def test_basic_creation(self):
        """Test creating a basic trial."""
        trial = Trial(candidate={"lr": 0.01})

        assert trial.candidate == {"lr": 0.01}
        assert trial.status == "pending"
        assert trial.score == -float("inf")

    @pytest.mark.asyncio
    async def test_creation_with_status(self):
        """Test creating trial with status."""
        trial = Trial(candidate={"lr": 0.01}, status="running")

        assert trial.status == "running"

    @pytest.mark.asyncio
    async def test_creation_with_scores(self):
        """Test creating trial with scores."""
        trial = Trial(
            candidate={"lr": 0.01},
            score=0.85,
            scores={"accuracy": 0.9, "loss": 0.1},
        )

        assert trial.score == 0.85
        assert trial.scores == {"accuracy": 0.9, "loss": 0.1}

    @pytest.mark.asyncio
    async def test_creation_with_step(self):
        """Test creating trial with step."""
        trial = Trial(candidate={"lr": 0.01}, step=5)

        assert trial.step == 5

    @pytest.mark.asyncio
    async def test_id_is_ulid(self):
        """Test that id is ULID."""
        from ulid import ULID

        trial = Trial(candidate={"lr": 0.01})

        assert isinstance(trial.id, ULID)

    @pytest.mark.asyncio
    async def test_unique_ids(self):
        """Test that trials have unique IDs."""
        trial1 = Trial(candidate={"lr": 0.01})
        trial2 = Trial(candidate={"lr": 0.01})

        assert trial1.id != trial2.id


class TestTrialProperties:
    """Tests for Trial properties."""

    @pytest.mark.asyncio
    async def test_repr(self):
        """Test __repr__ method."""
        trial = Trial(
            candidate={"lr": 0.01},
            status="finished",
            step=3,
            score=0.85,
            scores={"accuracy": 0.9},
        )

        result = repr(trial)

        assert "Trial(" in result
        assert "status='finished'" in result
        assert "step=3" in result

    @pytest.mark.asyncio
    async def test_str(self):
        """Test __str__ method."""
        trial = Trial(candidate={"lr": 0.01})

        result = str(trial)

        assert "Trial(" in result

    @pytest.mark.asyncio
    async def test_created_at_property(self):
        """Test created_at computed property."""
        trial = Trial(candidate={"lr": 0.01})

        created_at = trial.created_at

        assert isinstance(created_at, datetime)

    @pytest.mark.asyncio
    async def test_cost_property_no_result(self):
        """Test cost property with no evaluation result."""
        trial = Trial(candidate={"lr": 0.01})

        assert trial.cost == 0

    @pytest.mark.asyncio
    async def test_output_property_no_result(self):
        """Test output property with no evaluation result."""
        trial = Trial(candidate={"lr": 0.01})

        assert trial.output is None


class TestTrialMethods:
    """Tests for Trial methods."""

    @pytest.mark.asyncio
    async def test_as_probe(self):
        """Test as_probe method."""
        trial = Trial(candidate={"lr": 0.01})

        result = trial.as_probe()

        assert result is trial
        assert trial.is_probe is True

    @pytest.mark.asyncio
    async def test_as_probe_with_dataset(self):
        """Test as_probe with dataset."""
        trial = Trial(candidate={"lr": 0.01})
        dataset = [{"x": 1}, {"x": 2}]

        result = trial.as_probe(dataset=dataset)

        assert trial.is_probe is True
        assert trial.dataset == dataset

    @pytest.mark.asyncio
    async def test_as_trial(self):
        """Test as_trial method."""
        trial = Trial(candidate={"lr": 0.01}, is_probe=True)

        result = trial.as_trial()

        assert result is trial
        assert trial.is_probe is False
        assert trial.dataset is None

    @pytest.mark.asyncio
    async def test_get_directional_score_no_name(self):
        """Test get_directional_score without name returns overall score."""
        trial = Trial(candidate={"lr": 0.01}, score=0.85)

        result = trial.get_directional_score()

        assert result == 0.85

    @pytest.mark.asyncio
    async def test_get_directional_score_with_name(self):
        """Test get_directional_score with name."""
        trial = Trial(
            candidate={"lr": 0.01},
            scores={"accuracy": 0.9},
        )

        result = trial.get_directional_score("accuracy")

        assert result == 0.9

    @pytest.mark.asyncio
    async def test_get_directional_score_missing_returns_default(self):
        """Test get_directional_score returns default for missing score."""
        trial = Trial(candidate={"lr": 0.01})

        result = trial.get_directional_score("missing", default=0.0)

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_done_not_done(self):
        """Test done() returns False initially."""
        trial = Trial(candidate={"lr": 0.01})

        # Check done() - should be False before completion
        assert trial.done() is False


class TestTrialStatus:
    """Tests for trial status values."""

    @pytest.mark.asyncio
    async def test_valid_statuses(self):
        """Test all valid status values."""
        for status in ["pending", "running", "finished", "failed", "pruned"]:
            trial = Trial(candidate={"lr": 0.01}, status=status)
            assert trial.status == status


# ==============================================================================
# Study Result Tests
# ==============================================================================


class TestStudyResult:
    """Tests for StudyResult class."""

    def test_empty_result(self):
        """Test empty result."""
        result = StudyResult(trials=[], stop_reason="unknown")

        assert len(result.trials) == 0
        assert result.stop_reason == "unknown"
        assert result.best_trial is None

    @pytest.mark.asyncio
    async def test_best_trial(self):
        """Test best_trial property."""
        trials = [
            Trial(candidate={"lr": 0.01}, score=0.5, status="finished"),
            Trial(candidate={"lr": 0.1}, score=0.9, status="finished"),
            Trial(candidate={"lr": 0.001}, score=0.7, status="finished"),
        ]

        result = StudyResult(trials=trials, stop_reason="max_trials_reached")

        assert result.best_trial is not None
        assert result.best_trial.score == 0.9

    @pytest.mark.asyncio
    async def test_best_trial_excludes_non_finished(self):
        """Test best_trial excludes non-finished trials."""
        trials = [
            Trial(candidate={"lr": 0.01}, score=0.9, status="failed"),
            Trial(candidate={"lr": 0.1}, score=0.5, status="finished"),
        ]

        result = StudyResult(trials=trials, stop_reason="max_trials_reached")

        assert result.best_trial is not None
        assert result.best_trial.score == 0.5


# ==============================================================================
# Context Variable Tests
# ==============================================================================


class TestCurrentTrial:
    """Tests for current_trial context variable."""

    def test_default_is_none(self):
        """Test default value is None."""
        assert current_trial.get() is None

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting the context variable."""
        trial = Trial(candidate={"lr": 0.01})
        token = current_trial.set(trial)

        try:
            assert current_trial.get() is trial
        finally:
            current_trial.reset(token)

        assert current_trial.get() is None


# ==============================================================================
# fit_objectives Tests
# ==============================================================================


class TestFitObjectives:
    """Tests for fit_objectives function."""

    def test_fit_string_objectives(self):
        """Test fitting string objectives."""
        result = fit_objectives(["accuracy", "loss"])

        assert len(result) == 2
        assert result[0] == "accuracy"
        assert result[1] == "loss"

    def test_fit_scorer_objectives(self):
        """Test fitting scorer objectives."""

        @Scorer
        async def accuracy(x: float) -> float:
            return x

        result = fit_objectives([accuracy])

        assert len(result) == 1
        assert isinstance(result[0], Scorer)

    def test_fit_mixed_objectives(self):
        """Test fitting mixed string and scorer objectives."""

        @Scorer
        async def accuracy(x: float) -> float:
            return x

        result = fit_objectives(["loss", accuracy])

        assert len(result) == 2
        assert result[0] == "loss"
        assert isinstance(result[1], Scorer)

    def test_fit_mapping_objectives(self):
        """Test fitting mapping objectives."""

        @Scorer
        async def accuracy(x: float) -> float:
            return x

        result = fit_objectives({"accuracy": accuracy})

        assert len(result) == 1


# ==============================================================================
# Study Creation Tests
# ==============================================================================


class TestStudyCreation:
    """Tests for Study creation."""

    def test_basic_creation(self):
        """Test creating a basic study."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        # Mock search strategy
        search = MagicMock(spec=Search)

        study = Study(
            name="test_study",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}, {"x": 2}],
        )

        assert study.name == "test_study"
        assert study.max_trials == 100  # Default
        assert "maximize" in study.directions

    def test_creation_with_multiple_objectives(self):
        """Test creating study with multiple objectives."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy", "loss"],
            directions=["maximize", "minimize"],
            dataset=[{"x": 1}],
        )

        assert len(study.directions) == 2

    def test_creation_mismatched_directions_raises(self):
        """Test that mismatched directions raises ValueError."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        with pytest.raises(ValueError, match="Number of directions"):
            Study(
                name="test",
                search_strategy=search,
                task_factory=task_factory,
                objectives=["accuracy", "loss"],
                directions=["maximize", "minimize", "maximize"],  # 3 directions for 2 objectives
                dataset=[{"x": 1}],
            )

    def test_creation_with_constraints(self):
        """Test creating study with constraints."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        @Scorer
        async def valid_lr(candidate: dict) -> float:
            return 1.0 if candidate.get("lr", 0) > 0 else 0.0

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            constraints=[valid_lr],
            dataset=[{"x": 1}],
        )

        assert len(Scorer.fit_many(study.constraints)) == 1


class TestStudyProperties:
    """Tests for Study properties."""

    def test_objective_names(self):
        """Test objective_names property."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy", "f1"],
            dataset=[{"x": 1}],
        )

        assert study.objective_names == ["accuracy", "f1"]

    def test_auto_name_from_objectives(self):
        """Test auto-generated name from objectives."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="",  # Empty name
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        assert "study" in study.name
        assert "accuracy" in study.name


class TestStudyTraceContext:
    """Tests for Study trace context."""

    @pytest.mark.xfail(reason="Pre-existing recursion bug in warn_at_user_stacklevel")
    def test_get_trace_context(self):
        """Test _get_trace_context method."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)
        search.__class__.__name__ = "MockSearch"

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        ctx = study._get_trace_context()

        assert ctx.span_type == "study"
        assert "search_strategy" in ctx.inputs
        assert "objectives" in ctx.inputs
        assert "max_trials" in ctx.params


# ==============================================================================
# Study Events Tests
# ==============================================================================


class TestStudyEvents:
    """Tests for study events."""

    def test_study_start_event(self):
        """Test StudyStart event."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        event = StudyStart(
            study=study,
            trials=[],
            probes=[],
            max_trials=100,
        )

        assert event.study is study
        assert event.max_trials == 100

    @pytest.mark.asyncio
    async def test_trial_added_event(self):
        """Test TrialAdded event."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        trial = Trial(candidate={"lr": 0.01})

        event = TrialAdded(
            study=study,
            trials=[trial],
            probes=[],
            trial=trial,
        )

        assert event.trial is trial

    @pytest.mark.asyncio
    async def test_trial_complete_event(self):
        """Test TrialComplete event."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        trial = Trial(candidate={"lr": 0.01}, status="finished", score=0.9)

        event = TrialComplete(
            study=study,
            trials=[trial],
            probes=[],
            trial=trial,
        )

        assert event.trial.status == "finished"

    @pytest.mark.asyncio
    async def test_new_best_trial_event(self):
        """Test NewBestTrialFound event."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        def task_factory(candidate):
            return my_task.with_()

        search = MagicMock(spec=Search)

        study = Study(
            name="test",
            search_strategy=search,
            task_factory=task_factory,
            objectives=["accuracy"],
            dataset=[{"x": 1}],
        )

        trial = Trial(candidate={"lr": 0.01}, status="finished", score=0.95)

        event = NewBestTrialFound(
            study=study,
            trials=[trial],
            probes=[],
            trial=trial,
        )

        assert event.trial.score == 0.95


# ==============================================================================
# Trial Awaitable Tests
# ==============================================================================


class TestTrialAwaitable:
    """Tests for Trial await functionality."""

    @pytest.mark.asyncio
    async def test_await_trial(self):
        """Test awaiting a trial."""
        trial = Trial(candidate={"lr": 0.01})

        # Set the future result to simulate completion
        trial._future.set_result(trial)

        result = await trial

        assert result is trial

    @pytest.mark.asyncio
    async def test_wait_for_multiple_trials(self):
        """Test waiting for multiple trials."""
        trials = [
            Trial(candidate={"lr": 0.01}),
            Trial(candidate={"lr": 0.1}),
            Trial(candidate={"lr": 0.001}),
        ]

        # Set all futures
        for t in trials:
            t._future.set_result(t)

        results = await Trial.wait_for(*trials)

        assert len(results) == 3
        assert all(isinstance(r, Trial) for r in results)
