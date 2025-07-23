import typing as t

from dreadnode.metric import Metric, Scorer
from dreadnode.util import warn_at_user_stacklevel

_TEXTSTAT_AVAILABLE = False
_TEXTSTAT_ERROR_MSG = (
    "textstat dependency is not installed. Please install it with: pip install textstat"
)

try:
    import textstat  # type: ignore[import-not-found,unused-ignore,import-untyped]

    _TEXTSTAT_AVAILABLE = True
except ImportError:
    pass


def readability(
    target_grade: float = 8.0,
    name: str = "readability",
) -> "Scorer[t.Any]":
    """
    Score the readability of the text against a target grade level.

    The score is 1.0 if the calculated grade level matches the target_grade,
    and it degrades towards 0.0 as the distance from the target increases.

    Args:
        target_grade: The ideal reading grade level (e.g., 8.0 for 8th grade).
        metric: The readability metric to use. Currently only 'flesch_kincaid' is supported.
        name: Name of the scorer.
    """
    if not _TEXTSTAT_AVAILABLE:
        warn_at_user_stacklevel(_TEXTSTAT_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _TEXTSTAT_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        text = str(data)
        if not text.strip():
            return Metric(value=0.0, attributes={"error": "Input text is empty."})

        # The Flesch-Kincaid grade level calculation
        grade_level = textstat.flesch_kincaid_grade(text)

        # Score is inversely related to the absolute difference from the target.
        # We normalize by a factor (e.g., 10) to control how quickly the score drops off.
        # A difference of 10 grades or more results in a score of 0.
        diff = abs(grade_level - target_grade)
        score = max(0.0, 1.0 - (diff / 10.0))

        return Metric(
            value=score, attributes={"calculated_grade": grade_level, "target_grade": target_grade}
        )

    return Scorer.from_callable(evaluate, name=name)
