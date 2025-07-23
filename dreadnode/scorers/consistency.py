import re
import typing as t

from dreadnode.lookup import Lookup, resolve_lookup
from dreadnode.metric import Metric, Scorer

if t.TYPE_CHECKING:
    from dreadnode.types import JsonDict


def _analyze_text(text: str) -> dict[str, int]:
    return {
        "letters": len(re.findall(r"[a-zA-Z]", text)),
        "numbers": len(re.findall(r"\d", text)),
        "symbols": len(re.findall(r"[^\w\s]", text)),
    }


def character_consistency(
    reference: str | Lookup,
    *,
    max_ratio_diff: float = 2.0,
    name: str = "char_consistency",
) -> "Scorer[t.Any]":
    """
    Scores character type consistency between the data and a reference text.

    It compares the ratio of letters, numbers, and symbols in both texts.
    A score of 1.0 indicates identical distributions.

    Args:
        reference: The reference text (e.g., the prompt) or a Lookup.
        max_ratio_diff: The denominator for normalizing ratio differences.
        name: Name of the scorer.
    """

    def evaluate(data: t.Any) -> Metric:
        nonlocal reference

        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        candidate_chars = _analyze_text(candidate_text)
        reference_chars = _analyze_text(reference)

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

    return Scorer.from_callable(evaluate, name=name)
