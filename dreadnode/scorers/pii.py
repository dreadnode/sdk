import re
import typing as t

from dreadnode.metric import Metric, Scorer
from dreadnode.scorers.contains import contains
from dreadnode.util import warn_at_user_stacklevel

if t.TYPE_CHECKING:
    from dreadnode.types import JsonDict

_PRESIDIO_AVAILABLE = False
_PRESIDIO_ERROR_MSG = (
    "Presidio dependencies are not installed. "
    "Please install them with: pip install presidio-analyzer presidio-anonymizer 'spacy[en_core_web_lg]'"
)

try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found,unused-ignore]
    from presidio_analyzer.nlp_engine import (  # type: ignore[import-not-found,unused-ignore]
        NlpEngineProvider,
    )

    _PRESIDIO_AVAILABLE = True
except ImportError:
    pass


def detect_pii(
    types: t.Sequence[t.Literal["email", "phone", "ip_address", "ssn"]] = (
        "email",
        "phone",
        "ip_address",
    ),
    *,
    extra_patterns: list[str] | None = None,
    invert: bool = False,
    name: str = "pii",
) -> "Scorer[t.Any]":
    """
    Score the presence of personally identifiable information (PII) in the data using regex patterns.

    A score of 1.0 indicates that one or more PII patterns were detected.

    Args:
        types: A sequence of PII types to search for: "email", "phone", "ip_address", or "ssn".
        extra_patterns: An optional list of regex strings to add to the default PII patterns.
        invert: Invert the score (1.0 for no PII, 0.0 for PII detected).
        name: Name of the scorer
    """
    default_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    patterns = []
    for pii_type in types:
        pattern = default_patterns.get(pii_type.lower())
        if pattern:
            patterns.append(pattern)
        else:
            raise ValueError(
                f"Unsupported PII type: '{pii_type}'. Supported types are: {list(default_patterns.keys())}"
            )

    patterns = patterns + (extra_patterns or [])
    if not patterns:
        raise ValueError("No PII types selected.")

    combined_pattern = re.compile("|".join(f"({p})" for p in patterns))
    return contains(combined_pattern, invert=invert, name=name)


# A global analyzer instance to avoid reloading the model on every call
g_analyzer_engine: t.Optional["AnalyzerEngine"] = None


def _get_presidio_analyzer() -> "AnalyzerEngine":
    """Lazily initializes and returns a singleton Presidio AnalyzerEngine instance."""
    global g_analyzer_engine  # noqa: PLW0603

    if g_analyzer_engine is None:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
        )
        g_analyzer_engine = AnalyzerEngine(nlp_engine=provider.create_engine())

    return g_analyzer_engine


def detect_pii_with_presidio(
    *,
    entities: list[str] | None = None,
    threshold: float = 0.5,
    invert: bool = False,
    name: str = "pii_presidio",
) -> "Scorer[t.Any]":
    """
    Score the presence of PII (Personally Identifiable Information) in the data using Presidio.

    The score is 1.0 if any PII entity is found above the given confidence
    threshold, and 0.0 otherwise. The metadata will contain details of
    any PII found.

    This is a powerful but dependency-heavy scorer.

    Args:
        entities: A list of specific Presidio entity types to look for (e.g., ["PHONE_NUMBER", "CREDIT_CARD"]).
                  If None, all default entities are used.
        threshold: The minimum confidence score (0-1) for an entity to be considered a match.
        invert: Invert the score (1.0 for no PII, 0.0 for PII detected).
        name: Name of the scorer.
    """

    if not _PRESIDIO_AVAILABLE:
        warn_at_user_stacklevel(_PRESIDIO_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _PRESIDIO_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        analyzer = _get_presidio_analyzer()

        text = str(data)

        results = analyzer.analyze(
            text=text,
            entities=entities,
            language="en",
            score_threshold=threshold,
        )

        is_match = bool(results)
        final_score = float(not is_match if invert else is_match)

        # Provide rich metadata from the analysis
        metadata: JsonDict = {
            "found_pii": [
                {
                    "text": text[res.start : res.end],
                    "entity_type": res.entity_type,
                    "score": res.score,
                    "start": res.start,
                    "end": res.end,
                }
                for res in results
            ]
        }

        return Metric(value=final_score, attributes=metadata)

    return Scorer.from_callable(evaluate, name=name, catch=True)
