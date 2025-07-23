import typing as t

from dreadnode.metric import Metric, Scorer

ScorerT = t.TypeVar("ScorerT", bound="Scorer[t.Any]")


def invert(scorer: ScorerT, *, max_value: float = 1.0, name: str | None = None) -> ScorerT:
    """
    Creates a new scorer that inverts the result of the wrapped scorer.

    The new score is calculated as `max_value - original_score`.
    Attributes from the original metric are preserved.

    Args:
        scorer: The Scorer instance to wrap.
        max_value: The maximum value of the original score, used for inversion.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: t.Any) -> Metric:
        original_metric = await scorer(data)
        inverted_value = max(0, max_value - original_metric.value)
        return Metric(value=inverted_value, attributes=original_metric.attributes)

    name = name or f"{scorer.name}_inverted"
    return Scorer.from_callable(evaluate, name=name)  # type: ignore [return-value]


def scale(
    scorer: ScorerT,
    new_min: float,
    new_max: float,
    *,
    original_min: float = 0.0,
    original_max: float = 1.0,
    name: str | None = None,
) -> ScorerT:
    """
    Creates a new scorer that scales the result of the wrapped scorer to a new range.

    Args:
        scorer: The Scorer instance to wrap.
        new_min: The minimum value of the new range.
        new_max: The maximum value of the new range.
        original_min: The assumed minimum of the original score (default 0.0).
        original_max: The assumed maximum of the original score (default 1.0).
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """
    if original_min >= original_max or new_min >= new_max:
        raise ValueError("Min values must be less than max values.")

    original_range = original_max - original_min
    new_range = new_max - new_min

    async def evaluate(data: t.Any) -> Metric:
        original_metric = await scorer(data)

        if original_range == 0:  # Avoid division by zero
            scaled_value = new_min
        else:
            # Normalize original score to 0-1
            normalized = (original_metric.value - original_min) / original_range
            # Scale to new range
            scaled_value = new_min + (normalized * new_range)

        # Clamp the value to the new range to handle potential floating point errors
        final_value = max(new_min, min(new_max, scaled_value))

        return Metric(value=final_value, attributes=original_metric.attributes)

    name = name or f"{scorer.name}_scaled"
    return Scorer.from_callable(evaluate, name=name)  # type: ignore [return-value]


def threshold(
    scorer: ScorerT,
    *,
    gt: float | None = None,
    gte: float | None = None,
    lt: float | None = None,
    lte: float | None = None,
    pass_value: float = 1.0,
    fail_value: float = 0.0,
    name: str | None = None,
) -> ScorerT:
    """
    Creates a binary scorer that returns one of two values based on a threshold.

    If any threshold condition is met, it returns `pass_value`, otherwise `fail_value`.

    Args:
        scorer: The Scorer instance to wrap.
        gt: Passes if score is greater than this value.
        gte: Passes if score is greater than or equal to this value.
        lt: Passes if score is less than this value.
        lte: Passes if score is less than or equal to this value.
        pass_value: The score to return on a successful threshold check.
        fail_value: The score to return on a failed threshold check.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: t.Any) -> Metric:
        original_metric = await scorer(data)
        v = original_metric.value

        passed = False
        if gt is not None and v > gt:
            passed = True
        if gte is not None and v >= gte:
            passed = True
        if lt is not None and v < lt:
            passed = True
        if lte is not None and v <= lte:
            passed = True

        return Metric(
            value=pass_value if passed else fail_value, attributes=original_metric.attributes
        )

    name = name or f"{scorer.name}_threshold"
    return Scorer.from_callable(evaluate, name=name)  # type: ignore [return-value]
