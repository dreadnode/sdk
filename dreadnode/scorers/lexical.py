import re
import typing as t

from dreadnode.metric import Metric
from dreadnode.scorers import Scorer


def type_token_ratio(
    target_ratio: float | None = None,
    *,
    name: str = "type_token_ratio",
) -> "Scorer[t.Any]":
    """
    Scores the lexical diversity of the text using Type-Token Ratio (TTR).

    TTR is the ratio of unique words (types) to total words (tokens).
    A higher TTR indicates greater lexical diversity.

    - If `target_ratio` is None, the score is the raw TTR (0.0 to 1.0).
    - If `target_ratio` is set, the score is 1.0 if the TTR matches the target,
      degrading towards 0.0 as it deviates.

    Args:
        target_ratio: An optional ideal TTR to score against.
        name: Name of the scorer.
    """

    def evaluate(data: t.Any, *, target_ratio: float | None = target_ratio) -> Metric:
        if target_ratio is not None and not (0.0 <= target_ratio <= 1.0):
            raise ValueError("target_ratio must be between 0.0 and 1.0.")

        text = str(data)
        if not text.strip():
            return Metric(
                value=0.0,
                attributes={"ttr": 0, "unique_tokens": 0, "total_tokens": 0},
            )

        tokens = re.findall(r"\w+", text.lower())
        total_tokens = len(tokens)
        if total_tokens == 0:
            return Metric(
                value=0.0,
                attributes={"ttr": 0, "unique_tokens": 0, "total_tokens": 0},
            )

        unique_tokens = len(set(tokens))
        ttr = unique_tokens / total_tokens

        score = ttr
        if target_ratio is not None:
            # Score is 1 minus the normalized distance from the target
            diff = abs(ttr - target_ratio)
            score = max(0.0, 1.0 - (diff / target_ratio)) if target_ratio > 0 else 1.0 - diff

        return Metric(
            value=score,
            attributes={
                "ttr": round(ttr, 4),
                "unique_tokens": unique_tokens,
                "total_tokens": total_tokens,
            },
        )

    return Scorer(evaluate, name=name)
