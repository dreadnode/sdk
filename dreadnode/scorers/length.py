import typing as t

from dreadnode.metric import Metric, Scorer
from dreadnode.task import TaskInput
from dreadnode.util import clean_str


def length_ratio(
    reference: str | TaskInput,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 5.0,
    name: str | None = None,
) -> "Scorer[t.Any]":
    """
    Score the length of the data against a reference text.

    The score is 1.0 if the ratio (candidate/reference) is within the
    [min_ratio, max_ratio] bounds and degrades towards 0.0 outside them.

    Args:
        reference: The reference text (static string) or a `TaskInput` to resolve dynamically.
        min_ratio: The minimum acceptable length ratio. Must be > 0.
        max_ratio: The maximum acceptable length ratio.
        name: Name of the scorer.
    """
    if min_ratio <= 0:
        raise ValueError("min_ratio must be greater than 0.")

    def evaluate(data: t.Any) -> Metric:
        candidate_text = str(data)
        reference_text = str(reference.resolve()) if isinstance(reference, TaskInput) else reference

        if not reference_text:
            raise ValueError("Reference text must not be empty.")

        ratio = len(candidate_text) / len(reference_text)

        if ratio < min_ratio:
            score = ratio / min_ratio
        elif ratio > max_ratio:
            score = max_ratio / ratio
        else:
            score = 1.0

        return Metric(value=score, attributes={"ratio": round(ratio, 4)})

    if name is None:
        ref_name = reference.name if isinstance(reference, TaskInput) else reference
        name = f"length_ratio_vs_{clean_str(ref_name, max_length=20)}"

    return Scorer.from_callable(evaluate, name=name, catch=True)


def length_in_range(
    min: int = 0,
    max: float = float("inf"),
    name: str = "length_in_range",
) -> "Scorer[t.Any]":
    """
    Scores the length of the data against a specified range.

    The score is 1.0 if the length is within [min, max]. Outside the bounds,
    the score degrades towards 0.0. A score of 0.0 is returned for empty text.

    Args:
        min: The minimum acceptable character length.
        max: The maximum acceptable character length.
        name: Name of the scorer.
    """
    if min < 0 or max < min:
        raise ValueError("Invalid length bounds. Must have 0 <= min <= max.")

    def evaluate(data: t.Any) -> Metric:
        text = str(data)
        text_len = len(text)

        if text_len == 0 and min > 0:
            return Metric(value=0.0, attributes={"length": 0})

        score = 0.0
        if min <= text_len <= max:
            score = 1.0
        elif text_len < min:
            # Degrade score linearly from min down to 0 length
            score = text_len / min
        else:
            # Inverse relationship for text_len > max
            score = max / text_len if text_len > 0 else 0.0

        return Metric(value=score, attributes={"length": text_len, "min": min, "max": max})

    return Scorer.from_callable(evaluate, name=name)


def length_target(
    target_length: int,
    *,
    name: str = "length_target",
) -> "Scorer[t.Any]":
    """
    Scores the length of the data against a target length.

    The score is 1.0 if the length matches the target, and degrades towards 0.0
    as the length deviates from the target. A score of 0.0 is returned for empty text.

    Args:
        target_length: The target character length to score against.
        name: Name of the scorer.
    """
    if target_length < 0:
        raise ValueError("Target length must be non-negative.")

    def evaluate(data: t.Any) -> Metric:
        text = str(data)
        text_len = len(text)

        if text_len == 0:
            return Metric(value=0.0, attributes={"length": 0, "target": target_length})

        score = 1.0 - abs(text_len - target_length) / target_length if target_length > 0 else 0.0
        return Metric(value=score, attributes={"length": text_len, "target": target_length})

    return Scorer.from_callable(evaluate, name=name)
