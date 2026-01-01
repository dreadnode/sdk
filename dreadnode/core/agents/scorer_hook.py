"""ScorerHook: Bridge between Scorers and Agent hooks for per-step scoring."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from dreadnode.core.agents.events import AgentEvent, AgentEventT, AgentStep
from dreadnode.core.agents.reactions import Fail, Finish, Reaction, RetryWithFeedback
from dreadnode.core.metric import Metric
from dreadnode.core.scorer import Scorer, ScorerLike


@dataclass
class ScorerHookResult:
    """Result from a ScorerHook execution."""

    scorer_name: str
    """Name of the scorer that produced this result."""
    metric: Metric | None = None
    """The computed metric, if any."""
    reaction: Reaction | None = None
    """The reaction to trigger, if any."""
    step: int | None = None
    """The step number when this score was recorded."""

if t.TYPE_CHECKING:
    from dreadnode.core.agents.events import GenerationStep, ToolStep


# Type for extracting the object to score from an event
EventAdapter = t.Callable[[AgentEventT], t.Any]


@dataclass
class ScoreCondition(t.Generic[AgentEventT]):
    """
    A condition based on a scorer's result.

    ScoreConditions can be used to:
    1. Filter when a ScorerHook should run
    2. Determine reactions based on score thresholds
    3. Compose with other conditions using & and |

    Examples:
        ```
        # Run if quality score > 0.5
        quality_scorer = dn.scorer(check_quality)
        condition = ScoreCondition(quality_scorer, gt=0.5)

        # Compose conditions
        safe_and_good = ScoreCondition(safety, gt=0.8) & ScoreCondition(quality, gt=0.5)
        ```
    """

    scorer: Scorer[t.Any]
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None
    eq: float | None = None
    ne: float | None = None
    adapter: EventAdapter[AgentEventT] | None = None

    async def evaluate(self, event: AgentEventT) -> tuple[bool, Metric]:
        """
        Evaluate the condition against an event.

        Args:
            event: The agent event to evaluate.

        Returns:
            A tuple of (passed, metric) where passed indicates if the condition was met.
        """
        # Extract the object to score from the event
        obj = self.adapter(event) if self.adapter else event
        metric = await self.scorer.score(obj)
        score = metric.value

        passed = False
        if self.gt is not None and score > self.gt:
            passed = True
        if self.gte is not None and score >= self.gte:
            passed = True
        if self.lt is not None and score < self.lt:
            passed = True
        if self.lte is not None and score <= self.lte:
            passed = True
        if self.eq is not None and score == self.eq:
            passed = True
        if self.ne is not None and score != self.ne:
            passed = True

        # If no threshold specified, always pass (just run the scorer)
        if all(
            x is None for x in [self.gt, self.gte, self.lt, self.lte, self.eq, self.ne]
        ):
            passed = True

        return passed, metric

    def __and__(self, other: ScoreCondition[AgentEventT]) -> CompositeCondition[AgentEventT]:
        """Combine conditions with AND."""
        return CompositeCondition(conditions=[self, other], operator="and")

    def __or__(self, other: ScoreCondition[AgentEventT]) -> CompositeCondition[AgentEventT]:
        """Combine conditions with OR."""
        return CompositeCondition(conditions=[self, other], operator="or")


@dataclass
class CompositeCondition(t.Generic[AgentEventT]):
    """
    A composite condition combining multiple ScoreConditions.

    Supports AND and OR operators for combining conditions.
    """

    conditions: list[ScoreCondition[AgentEventT] | CompositeCondition[AgentEventT]]
    operator: t.Literal["and", "or"] = "and"

    async def evaluate(self, event: AgentEventT) -> tuple[bool, list[Metric]]:
        """
        Evaluate all conditions and combine results.

        Args:
            event: The agent event to evaluate.

        Returns:
            A tuple of (passed, metrics) where passed indicates if the composite condition was met.
        """
        metrics: list[Metric] = []
        results: list[bool] = []

        for condition in self.conditions:
            if isinstance(condition, CompositeCondition):
                passed, sub_metrics = await condition.evaluate(event)
                metrics.extend(sub_metrics)
            else:
                passed, metric = await condition.evaluate(event)
                metrics.append(metric)
            results.append(passed)

        if self.operator == "and":
            return all(results), metrics
        else:
            return any(results), metrics

    def __and__(
        self, other: ScoreCondition[AgentEventT] | CompositeCondition[AgentEventT]
    ) -> CompositeCondition[AgentEventT]:
        """Combine with another condition using AND."""
        if self.operator == "and":
            return CompositeCondition(conditions=[*self.conditions, other], operator="and")
        return CompositeCondition(conditions=[self, other], operator="and")

    def __or__(
        self, other: ScoreCondition[AgentEventT] | CompositeCondition[AgentEventT]
    ) -> CompositeCondition[AgentEventT]:
        """Combine with another condition using OR."""
        if self.operator == "or":
            return CompositeCondition(conditions=[*self.conditions, other], operator="or")
        return CompositeCondition(conditions=[self, other], operator="or")


# Type alias for conditions
ConditionLike = ScoreCondition[AgentEventT] | CompositeCondition[AgentEventT] | None


class ScorerHook(t.Generic[AgentEventT]):
    """
    A hook that runs a Scorer against agent events.

    ScorerHook bridges the gap between Scorers (which evaluate objects) and
    Agent hooks (which react to events). This enables per-step scoring during
    agent execution, even outside of an Evaluation context.

    Features:
    - Trigger on specific event types (GenerationStep, ToolStep, etc.)
    - Extract specific fields from events to score
    - Optionally react based on score thresholds
    - Create scorer spans for tracing
    - Accumulate scores in the Trajectory

    Examples:
        ```
        # Score every generation step
        @dn.scorer
        async def quality(text: str) -> float:
            return await check_quality(text)

        # Create a hook that runs on GenerationStep
        quality_hook = ScorerHook(
            scorer=quality,
            event_type=GenerationStep,
            adapter=lambda e: e.messages[0].content if e.messages else "",
        )

        # Use in an agent
        agent = Agent(
            ...,
            scorers=[quality_hook],
        )

        # Or use the convenience method on Scorer
        quality_hook = quality.on(GenerationStep, adapter=lambda e: e.messages[0].content)
        ```
    """

    def __init__(
        self,
        scorer: ScorerLike[t.Any],
        event_type: type[AgentEventT],
        *,
        adapter: EventAdapter[AgentEventT] | None = None,
        condition: ConditionLike[AgentEventT] = None,
        on_low: Reaction | t.Callable[[Metric], Reaction | None] | None = None,
        on_high: Reaction | t.Callable[[Metric], Reaction | None] | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        log_metrics: bool = True,
        create_span: bool = True,
    ):
        """
        Create a ScorerHook.

        Args:
            scorer: The scorer to run on matching events.
            event_type: The event type to trigger on.
            adapter: Optional function to extract the object to score from the event.
            condition: Optional condition that must be met for the hook to run.
            on_low: Reaction to trigger when score is below low_threshold.
            on_high: Reaction to trigger when score is above high_threshold.
            low_threshold: Threshold for triggering on_low reaction.
            high_threshold: Threshold for triggering on_high reaction.
            log_metrics: Whether to log metrics to the current span.
            create_span: Whether to create a scorer span for tracing.
        """
        self.scorer = Scorer.fit(scorer)
        self.event_type = event_type
        self.adapter = adapter
        self.condition = condition
        self.on_low = on_low
        self.on_high = on_high
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.log_metrics = log_metrics
        self.create_span = create_span
        self.__name__ = f"scorer_hook:{self.scorer.name}"

    async def __call__(self, event: AgentEvent) -> ScorerHookResult | None:
        """
        Execute the scorer hook on an event.

        Args:
            event: The agent event to potentially score.

        Returns:
            A ScorerHookResult containing the metric and any triggered reaction,
            or None if the event type doesn't match.
        """
        # Type check
        if not isinstance(event, self.event_type):
            return None

        # Evaluate condition if present
        if self.condition is not None:
            if isinstance(self.condition, CompositeCondition):
                passed, _ = await self.condition.evaluate(event)
            else:
                passed, _ = await self.condition.evaluate(event)
            if not passed:
                return None

        # Extract object to score
        obj = self.adapter(event) if self.adapter else event

        # Get step number if available
        step = event.step if isinstance(event, AgentStep) else None

        # Score the object
        metric: Metric
        if self.create_span:
            from dreadnode.core.tracing.spans import scorer_span

            with scorer_span(
                self.scorer.name,
                step=step,
            ) as span:
                metric = await self.scorer.score(obj)
                span.set_attribute("dreadnode.scorer.score", metric.value)
                if metric.rationale:
                    span.set_attribute("dreadnode.scorer.rationale", metric.rationale[:1000])
                span.set_attribute("dreadnode.scorer.passed", bool(metric.value))
        else:
            metric = await self.scorer.score(obj)

        # Log metrics if enabled
        if self.log_metrics:
            from dreadnode import log_metric

            log_metric(
                f"scorer/{self.scorer.name}",
                metric.value,
                step=step or 0,
                mode="direct",
            )

        # Check thresholds and trigger reactions
        reaction: Reaction | None = None
        if self.low_threshold is not None and metric.value < self.low_threshold:
            if callable(self.on_low):
                reaction = self.on_low(metric)
            else:
                reaction = self.on_low

        if self.high_threshold is not None and metric.value > self.high_threshold:
            if callable(self.on_high):
                reaction = self.on_high(metric)
            else:
                reaction = self.on_high

        return ScorerHookResult(
            scorer_name=self.scorer.name,
            metric=metric,
            reaction=reaction,
            step=step,
        )

    def when(
        self, condition: ConditionLike[AgentEventT]
    ) -> ScorerHook[AgentEventT]:
        """
        Add a condition for when this hook should run.

        Args:
            condition: The condition that must be met.

        Returns:
            A new ScorerHook with the condition applied.
        """
        return ScorerHook(
            scorer=self.scorer,
            event_type=self.event_type,
            adapter=self.adapter,
            condition=condition,
            on_low=self.on_low,
            on_high=self.on_high,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            log_metrics=self.log_metrics,
            create_span=self.create_span,
        )

    def react_on_low(
        self,
        threshold: float,
        reaction: Reaction | t.Callable[[Metric], Reaction | None],
    ) -> ScorerHook[AgentEventT]:
        """
        Configure a reaction when score falls below a threshold.

        Args:
            threshold: The score threshold.
            reaction: The reaction to trigger, or a callable that returns one.

        Returns:
            A new ScorerHook with the reaction configured.
        """
        return ScorerHook(
            scorer=self.scorer,
            event_type=self.event_type,
            adapter=self.adapter,
            condition=self.condition,
            on_low=reaction,
            on_high=self.on_high,
            low_threshold=threshold,
            high_threshold=self.high_threshold,
            log_metrics=self.log_metrics,
            create_span=self.create_span,
        )

    def react_on_high(
        self,
        threshold: float,
        reaction: Reaction | t.Callable[[Metric], Reaction | None],
    ) -> ScorerHook[AgentEventT]:
        """
        Configure a reaction when score exceeds a threshold.

        Args:
            threshold: The score threshold.
            reaction: The reaction to trigger, or a callable that returns one.

        Returns:
            A new ScorerHook with the reaction configured.
        """
        return ScorerHook(
            scorer=self.scorer,
            event_type=self.event_type,
            adapter=self.adapter,
            condition=self.condition,
            on_low=self.on_low,
            on_high=reaction,
            low_threshold=self.low_threshold,
            high_threshold=threshold,
            log_metrics=self.log_metrics,
            create_span=self.create_span,
        )

    def fail_if_below(
        self, threshold: float, error: str | None = None
    ) -> ScorerHook[AgentEventT]:
        """
        Convenience method to fail the agent if score is below threshold.

        Args:
            threshold: The minimum acceptable score.
            error: Optional error message.

        Returns:
            A new ScorerHook configured to fail on low scores.
        """
        return self.react_on_low(
            threshold,
            Fail(error=error or f"Score {self.scorer.name} below threshold {threshold}"),
        )

    def finish_if_above(
        self, threshold: float, reason: str | None = None
    ) -> ScorerHook[AgentEventT]:
        """
        Convenience method to finish the agent if score exceeds threshold.

        Args:
            threshold: The score threshold to trigger finish.
            reason: Optional reason for finishing.

        Returns:
            A new ScorerHook configured to finish on high scores.
        """
        return self.react_on_high(
            threshold,
            Finish(reason=reason or f"Score {self.scorer.name} exceeded threshold {threshold}"),
        )

    def retry_if_below(
        self, threshold: float, feedback: str | None = None
    ) -> ScorerHook[AgentEventT]:
        """
        Convenience method to retry with feedback if score is below threshold.

        Args:
            threshold: The minimum acceptable score.
            feedback: Optional feedback message for the retry.

        Returns:
            A new ScorerHook configured to retry on low scores.
        """
        return self.react_on_low(
            threshold,
            lambda m: RetryWithFeedback(
                feedback=feedback
                or f"Score {self.scorer.name} was {m.value:.2f}, below threshold {threshold}. Please improve."
            ),
        )


# Type alias for scorer hooks
ScorerHookLike = ScorerHook[AgentEventT] | ScorerLike[t.Any]


def scorer_on(
    scorer: ScorerLike[t.Any],
    event_type: type[AgentEventT],
    *,
    adapter: EventAdapter[AgentEventT] | None = None,
    **kwargs: t.Any,
) -> ScorerHook[AgentEventT]:
    """
    Create a ScorerHook from a scorer.

    This is a convenience function that wraps Scorer.on() for functional style usage.

    Args:
        scorer: The scorer to wrap.
        event_type: The event type to trigger on.
        adapter: Optional function to extract the object to score from the event.
        **kwargs: Additional arguments passed to ScorerHook.

    Returns:
        A configured ScorerHook.

    Examples:
        ```
        from dreadnode.core.agents.scorer_hook import scorer_on
        from dreadnode.core.agents.events import GenerationStep

        hook = scorer_on(
            quality_scorer,
            GenerationStep,
            adapter=lambda e: e.messages[0].content if e.messages else "",
        )
        ```
    """
    return ScorerHook(
        scorer=scorer,
        event_type=event_type,
        adapter=adapter,
        **kwargs,
    )
