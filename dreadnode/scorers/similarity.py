import typing as t
from difflib import SequenceMatcher

from dreadnode.metric import Metric, Scorer
from dreadnode.task import TaskInput
from dreadnode.util import clean_str, warn_at_user_stacklevel

_NLTK_AVAILABLE = False
_NLTK_ERROR_MSG = "nltk dependency is not installed. Please run: pip install nltk && python -m nltk.downloader punkt"

try:
    import nltk  # type: ignore[import-not-found,unused-ignore]
    from nltk.tokenize import word_tokenize  # type: ignore[import-not-found,unused-ignore]
    from nltk.translate.bleu_score import (  # type: ignore[import-not-found,unused-ignore]
        sentence_bleu,
    )

    # Check for the 'punkt' tokenizer data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        _NLTK_ERROR_MSG = (
            "NLTK 'punkt' tokenizer not found. Please run: python -m nltk.downloader punkt"
        )
        raise ImportError(_NLTK_ERROR_MSG) from e

    _NLTK_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
_SKLEARN_ERROR_MSG = (
    "scikit-learn dependency is not installed. Please install it with: pip install scikit-learn"
)

try:
    from sklearn.feature_extraction.text import (  # type: ignore[import-not-found,unused-ignore]
        TfidfVectorizer,
    )
    from sklearn.metrics.pairwise import (  # type: ignore[import-not-found,unused-ignore]
        cosine_similarity,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


def similarity(
    reference: str | TaskInput,
    *,
    method: t.Literal["ratio", "quick_ratio", "real_quick_ratio"] = "ratio",
    case_sensitive: bool = False,
    name: str | None = None,
) -> "Scorer[t.Any]":
    """
    Score the similarity of the data to a reference text using sequence matching.

    The score is a float between 0.0 (completely different) and 1.0 (identical),
    based on `difflib.SequenceMatcher`.

    Args:
        reference: The reference text (static string) or a `TaskInput` to resolve dynamically.
        method: The similarity comparison method to use.
        case_sensitive: Perform a case-sensitive comparison.
        name: Name of the scorer.
    """

    def evaluate(data: t.Any) -> Metric:
        candidate_text = str(data)
        reference_text = (
            reference.resolve(cast_as=str) if isinstance(reference, TaskInput) else reference
        )

        if not case_sensitive:
            candidate_text = candidate_text.lower()
            reference_text = reference_text.lower()

        matcher = SequenceMatcher(a=reference_text, b=candidate_text)

        if method == "quick_ratio":
            score = matcher.quick_ratio()
        elif method == "real_quick_ratio":
            score = matcher.real_quick_ratio()
        else:  # "ratio"
            score = matcher.ratio()

        return Metric(value=score, attributes={"method": method})

    if name is None:
        ref_name = reference.name if isinstance(reference, TaskInput) else reference
        name = f"similarity_to_{clean_str(ref_name, max_length=20)}"

    return Scorer.from_callable(evaluate, name=name, catch=True)


def semantic_similarity(
    reference: str | TaskInput,
    *,
    name: str | None = None,
) -> "Scorer[t.Any]":
    """
    Scores semantic similarity using TF-IDF and cosine similarity.

    Requires scikit-learn.

    Args:
        reference: The reference text (e.g., expected output) or a TaskInput.
        name: Name of the scorer.
    """
    if not _SKLEARN_AVAILABLE:
        warn_at_user_stacklevel(_SKLEARN_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _SKLEARN_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    vectorizer = TfidfVectorizer(stop_words="english")

    def evaluate(data: t.Any) -> Metric:
        candidate_text = str(data)
        reference_text = (
            reference.resolve(cast_as=str) if isinstance(reference, TaskInput) else reference
        )
        tfidf_matrix = vectorizer.fit_transform([candidate_text, reference_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return Metric(value=float(sim))

    if name is None:
        ref_name = reference.name if isinstance(reference, TaskInput) else "static_text"
        name = f"semantic_sim_to_{clean_str(ref_name)}"

    return Scorer.from_callable(evaluate, name=name, catch=True)


def bleu(
    reference: str | TaskInput,
    *,
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    name: str | None = None,
) -> "Scorer[t.Any]":
    """
    Scores the data using the BLEU score against a reference text.

    A score of 1.0 indicates a perfect match. Requires NLTK.

    Args:
        reference: The reference text (e.g., the prompt) or a TaskInput.
        weights: Weights for unigram, bigram, etc. Must sum to 1.
        name: Name of the scorer.
    """
    if not _NLTK_AVAILABLE:
        warn_at_user_stacklevel(_NLTK_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _NLTK_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        candidate_text = str(data)
        reference_text = (
            reference.resolve(cast_as=str) if isinstance(reference, TaskInput) else reference
        )

        if not reference_text or not candidate_text:
            return Metric(value=0.0, attributes={"error": "Reference or candidate text is empty."})

        ref_tokens = word_tokenize(reference_text)
        cand_tokens = word_tokenize(candidate_text)

        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        return Metric(value=score)

    if name is None:
        ref_name = reference.name if isinstance(reference, TaskInput) else "static_text"
        name = f"bleu_{clean_str(ref_name)}"

    return Scorer.from_callable(evaluate, name=name)
