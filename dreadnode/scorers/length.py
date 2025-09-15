import typing as t

from dreadnode.metric import Metric
from dreadnode.scorers import Scorer


def length_ratio(
    reference: str,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 5.0,
    name: str = "length_ratio",
) -> "Scorer[t.Any]":
    """
    Score the length of the data against a reference text.

    The score is 1.0 if the ratio (candidate/reference) is within the
    [min_ratio, max_ratio] bounds and degrades towards 0.0 outside them.

    Args:
        reference: The reference text (static string).
        min_ratio: The minimum acceptable length ratio. Must be > 0.
        max_ratio: The maximum acceptable length ratio.
        name: Name of the scorer.
    """
    if min_ratio <= 0:
        raise ValueError("min_ratio must be greater than 0.")

    def evaluate(
        data: t.Any,
        *,
        reference: str = reference,
        min_ratio: float = min_ratio,
        max_ratio: float = max_ratio,
    ) -> Metric:
        candidate_text = str(data)

        if not reference:
            raise ValueError("Reference text must not be empty.")

        ratio = len(candidate_text) / len(reference)

        if ratio < min_ratio:
            score = ratio / min_ratio
        elif ratio > max_ratio:
            score = max_ratio / ratio
        else:
            score = 1.0

        return Metric(value=score, attributes={"ratio": round(ratio, 4)})

    return Scorer(evaluate, name=name)


def length_in_range(
    min_length: int = 0,
    max_length: float = float("inf"),
    *,
    name: str = "length_in_range",
) -> "Scorer[t.Any]":
    """
    Scores the length of the data against a specified range.

    The score is 1.0 if the length is within [min, max]. Outside the bounds,
    the score degrades towards 0.0. A score of 0.0 is returned for empty text.

    Args:
        min_length: The minimum acceptable character length.
        max_length: The maximum acceptable character length.
        name: Name of the scorer.
    """

    def evaluate(
        data: t.Any, *, min_length: int = min_length, max_length: float = max_length
    ) -> Metric:
        if min_length < 0 or max_length < min_length:
            raise ValueError("Invalid length bounds. Must have 0 <= min <= max.")

        text = str(data)
        text_len = len(text)

        score = 0.0
        if min_length <= text_len <= max_length:
            score = 1.0
        elif text_len < min_length:
            # Linear ramp-up from 0 to min. Avoids division by zero if min is 0.
            score = text_len / min_length if min_length > 0 else 0.0
        else:  # text_len > max
            # Linear degradation. Score hits 0 when length is 2*max.
            # This is more predictable than an inverse curve.
            # We define the "penalty zone" as the range from max to 2*max.
            penalty_range = max_length
            overage = text_len - max_length
            score = 1.0 - (overage / penalty_range) if penalty_range > 0 else 0.0

        return Metric(
            value=max(0.0, score),
            attributes={"length": text_len, "min": min_length, "max": max_length},
        )

    return Scorer(evaluate, name=name)


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

    def evaluate(data: t.Any, *, target_length: int = target_length) -> Metric:
        if target_length < 0:
            raise ValueError("Target length must be non-negative.")

        text = str(data)
        text_len = len(text)

        # Handle the perfect match case first, especially for target=0
        if text_len == target_length:
            score = 1.0
        elif target_length == 0:
            # If target is 0, any non-zero length is a total miss.
            score = 0.0
        else:
            # Linear degradation based on distance from target.
            diff = abs(text_len - target_length)
            score = 1.0 - (diff / target_length)

        final_score = max(0.0, score)

        return Metric(value=final_score, attributes={"length": text_len, "target": target_length})

    return Scorer(evaluate, name=name)
