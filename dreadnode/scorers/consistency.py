import re
import typing as t

from dreadnode.metric import Metric, Scorer
from dreadnode.task import TaskInput
from dreadnode.util import clean_str

if t.TYPE_CHECKING:
    from dreadnode.types import JsonDict


def character_consistency(
    reference: str | TaskInput,
    *,
    max_ratio_diff: float = 2.0,
    name: str | None = None,
) -> "Scorer[t.Any]":
    """
    Scores character type consistency between the data and a reference text.

    It compares the ratio of letters, numbers, and symbols in both texts.
    A score of 1.0 indicates identical distributions.

    Args:
        reference: The reference text (e.g., the prompt) or a TaskInput.
        max_ratio_diff: The denominator for normalizing ratio differences.
        name: Name of the scorer.
    """

    def _analyze_text(text: str) -> dict[str, int]:
        return {
            "letters": len(re.findall(r"[a-zA-Z]", text)),
            "numbers": len(re.findall(r"\d", text)),
            "symbols": len(re.findall(r"[^\w\s]", text)),
        }

    def evaluate(data: t.Any) -> Metric:
        candidate_text = str(data)
        reference_text = str(reference.resolve()) if isinstance(reference, TaskInput) else reference

        candidate_chars = _analyze_text(candidate_text)
        reference_chars = _analyze_text(reference_text)

        candidate_total = sum(candidate_chars.values())
        reference_total = sum(reference_chars.values())

        if reference_total == 0 or candidate_total == 0:
            return Metric(value=0.0, attributes={"error": "Reference or candidate text is empty."})

        scores: dict[str, float] = {}
        metadata: JsonDict = {}
        for char_type in ["letters", "numbers", "symbols"]:
            ref_ratio = reference_chars[char_type] / reference_total
            cand_ratio = candidate_chars[char_type] / candidate_total
            diff = abs(ref_ratio - cand_ratio)
            score = max(0.0, 1.0 - (diff / max_ratio_diff))
            scores[char_type] = score
            metadata[f"{char_type}_ratio_diff"] = round(diff, 4)

        return Metric.from_many([(name, score, 1.0) for name, score in scores.items()])

    if name is None:
        ref_name = reference.name if isinstance(reference, TaskInput) else "static_text"
        name = f"char_consistency_{clean_str(ref_name)}"

    return Scorer.from_callable(evaluate, name=name)
