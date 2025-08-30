import typing as t
from difflib import SequenceMatcher

from dreadnode.lookup import Lookup, resolve_lookup
from dreadnode.metric import Metric, Scorer
from dreadnode.scorers.util import cosine_similarity
from dreadnode.util import warn_at_user_stacklevel

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
        cosine_similarity as sklearn_cosine_similarity,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

_SENTENCE_TRANSFORMERS_AVAILABLE = False
_SENTENCE_TRANSFORMERS_ERROR_MSG = "sentence-transformers dependency is not installed. Please install it with: pip install sentence-transformers"

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore[import-not-found]

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def similarity(
    reference: str | Lookup,
    *,
    method: t.Literal["ratio", "quick_ratio", "real_quick_ratio"] = "ratio",
    case_sensitive: bool = False,
    name: str = "similarity",
) -> "Scorer[t.Any]":
    """
    Score the similarity of the data to a reference text using sequence matching.

    The score is a float between 0.0 (completely different) and 1.0 (identical),
    based on `difflib.SequenceMatcher`.

    Args:
        reference: The reference text (static string).
        method: The similarity comparison method to use.
        case_sensitive: Perform a case-sensitive comparison.
        name: Name of the scorer.
    """

    def evaluate(data: t.Any) -> Metric:
        nonlocal reference

        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        if not case_sensitive:
            candidate_text = candidate_text.lower()
            reference = reference.lower()

        matcher = SequenceMatcher(a=reference, b=candidate_text)

        if method == "quick_ratio":
            score = matcher.quick_ratio()
        elif method == "real_quick_ratio":
            score = matcher.real_quick_ratio()
        else:  # "ratio"
            score = matcher.ratio()

        return Metric(value=score, attributes={"method": method})

    return Scorer.from_callable(evaluate, name=name, catch=True)


def similarity_with_tf_idf(reference: str | Lookup, *, name: str = "similarity") -> "Scorer[t.Any]":
    """
    Scores semantic similarity using TF-IDF and cosine similarity.

    Requires scikit-learn.

    Args:
        reference: The reference text (e.g., expected output).
        name: Name of the scorer.
    """
    if not _SKLEARN_AVAILABLE:
        warn_at_user_stacklevel(_SKLEARN_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _SKLEARN_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    vectorizer = TfidfVectorizer(stop_words="english")

    def evaluate(data: t.Any) -> Metric:
        nonlocal reference

        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        tfidf_matrix = vectorizer.fit_transform([candidate_text, reference])
        sim = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return Metric(value=float(sim))

    return Scorer.from_callable(evaluate, name=name, catch=True)


# A global model cache to avoid reloading on every call
g_sentence_transformers_models: dict[str, "SentenceTransformer"] = {}


def similarity_with_sentence_transformers(
    reference: str | Lookup,
    *,
    model_name: str | Lookup = "all-MiniLM-L6-v2",
    name: str = "similarity",
) -> "Scorer[t.Any]":
    """
    Scores semantic similarity using a sentence-transformer embedding model.

    This is a more robust alternative to TF-IDF or sequence matching, as it
    understands the meaning of words and sentences. The score is the
    cosine similarity between the reference and candidate text embeddings.

    Requires sentence-transformers.

    Args:
        reference: The reference text (e.g., expected output).
        model_name: The name of the sentence-transformer model to use.
        name: Name of the scorer.
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        warn_at_user_stacklevel(_SENTENCE_TRANSFORMERS_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _SENTENCE_TRANSFORMERS_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        nonlocal reference, model_name

        # Lazily load and cache the model
        model_name = str(resolve_lookup(model_name))
        if model_name not in g_sentence_transformers_models:
            g_sentence_transformers_models[model_name] = SentenceTransformer(model_name)
        model = g_sentence_transformers_models[model_name]

        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        embeddings = model.encode([candidate_text, reference])
        sim_tensor = util.cos_sim(embeddings[0], embeddings[1])
        return Metric(
            value=float(sim_tensor[0][0]),
            attributes={
                "model": model_name,
            },
        )

    return Scorer.from_callable(evaluate, name=name, catch=True)


def similarity_with_litellm(
    reference: str | Lookup,
    model: str | Lookup,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    name: str = "similarity",
) -> "Scorer[t.Any]":
    """
    Scores semantic similarity using any embedding model supported by `litellm`.

    This provides a unified interface to calculate embedding-based similarity using
    models from OpenAI, Cohere, Azure, Bedrock, and many others. The score is the
    cosine similarity between the reference and candidate text embeddings.

    See the `litellm` documentation for supported models.

    Args:
        reference: The reference text (e.g., expected output).
        model: The model string recognised by litellm (e.g., "text-embedding-ada-002",
               "cohere/embed-english-v3.0").
        api_key: The API key for the embedding provider. If None, litellm will try
                 to use the corresponding environment variable (e.g., OPENAI_API_KEY).
        api_base: The API base URL, for use with custom endpoints like Azure OpenAI
                  or self-hosted models.
        name: Name of the scorer.
    """
    import litellm

    async def evaluate(data: t.Any) -> Metric:
        nonlocal reference, model

        model = str(resolve_lookup(model))
        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        if not candidate_text.strip() or not reference.strip():
            return Metric(value=0.0, attributes={"error": "Candidate or reference text is empty."})

        response = await litellm.aembedding(
            model=model,
            input=[candidate_text, reference],
            api_key=api_key,
            api_base=api_base,
        )

        candidate_embedding = response.data[0].embedding
        reference_embedding = response.data[1].embedding

        similarity = cosine_similarity(candidate_embedding, reference_embedding)

        return Metric(
            value=similarity,
            attributes={
                "model": model,
            },
        )

    return Scorer.from_callable(evaluate, name=name, catch=True)


def bleu(
    reference: str | Lookup,
    *,
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    name: str = "bleu",
) -> "Scorer[t.Any]":
    """
    Scores the data using the BLEU score against a reference text.

    A score of 1.0 indicates a perfect match. Requires NLTK.

    Args:
        reference: The reference text (e.g., the prompt).
        weights: Weights for unigram, bigram, etc. Must sum to 1.
        name: Name of the scorer.
    """
    if not _NLTK_AVAILABLE:
        warn_at_user_stacklevel(_NLTK_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _NLTK_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        nonlocal reference

        candidate_text = str(data)
        reference = str(resolve_lookup(reference))

        if not reference or not candidate_text:
            return Metric(value=0.0, attributes={"error": "Reference or candidate text is empty."})

        ref_tokens = word_tokenize(reference)
        cand_tokens = word_tokenize(candidate_text)

        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        return Metric(value=score)

    return Scorer.from_callable(evaluate, name=name)
