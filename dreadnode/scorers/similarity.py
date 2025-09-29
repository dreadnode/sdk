import functools
import typing as t
from difflib import SequenceMatcher

from dreadnode.meta import Config
from dreadnode.metric import Metric
from dreadnode.scorers.base import Scorer
from dreadnode.scorers.util import cosine_similarity
from dreadnode.util import catch_import_error

if t.TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]


def similarity(
    reference: t.Any,
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

    def evaluate(
        data: t.Any,
        *,
        reference: t.Any = reference,
        method: t.Literal["ratio", "quick_ratio", "real_quick_ratio"] = method,
        case_sensitive: bool = case_sensitive,
    ) -> Metric:
        candidate_text = str(data)

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

    return Scorer(evaluate, name=name)


def similarity_with_rapidfuzz(
    reference: str,
    *,
    method: t.Literal[
        "ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "WRatio",
        "QRatio",
    ] = "ratio",
    normalize: bool = True,
    preprocessor: bool = True,
    score_cutoff: float | None = None,
    name: str = "similarity",
) -> "Scorer[t.Any]":
    """
    Score the similarity of the data to a reference text using RapidFuzz.

    RapidFuzz is significantly faster than difflib and provides more scoring methods.
    The score is a float between 0.0 (completely different) and 100.0 (identical),
    which is normalized to 0.0-1.0 for consistency with other scorers.

    Requires `rapidfuzz`, see https://github.com/rapidfuzz/RapidFuzz

    Args:
        reference: The reference text (static string).
        method: The RapidFuzz similarity method to use.
        normalize: Normalize the score to [0.0, 1.0].
        preprocessor: Use default preprocessing (lowercase, remove non-alphanumeric).
        score_cutoff: Optional score cutoff below which to return 0.0.
        name: Name of the scorer.
    """
    with catch_import_error("dreadnode[scoring]"):
        from rapidfuzz import fuzz, utils  # type: ignore[import-not-found]

    def evaluate(
        data: t.Any,
        *,
        reference: str = reference,
        method: t.Literal[
            "ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "WRatio",
            "QRatio",
        ] = method,
        normalize: bool = normalize,
        preprocessor: bool = preprocessor,
        score_cutoff: float | None = score_cutoff,
    ) -> Metric:
        candidate_text = str(data)
        processor = utils.default_process if preprocessor else None

        # Select the appropriate RapidFuzz method
        if method == "ratio":
            score = fuzz.ratio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        elif method == "partial_ratio":
            score = fuzz.partial_ratio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        elif method == "token_sort_ratio":
            score = fuzz.token_sort_ratio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        elif method == "token_set_ratio":
            score = fuzz.token_set_ratio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        elif method == "WRatio":
            score = fuzz.WRatio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        elif method == "QRatio":
            score = fuzz.QRatio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )
        else:
            score = fuzz.ratio(
                reference,
                candidate_text,
                processor=processor,
                score_cutoff=score_cutoff,
            )

        if normalize:
            score = score / 100.0 if score is not None else 0.0

        return Metric(
            value=score,
            attributes={
                "method": method,
                "preprocessor": preprocessor,
                "score_cutoff": score_cutoff,
                "raw_score": score,
            },
        )

    return Scorer(evaluate, name=name)


def string_distance(
    reference: str,
    *,
    method: t.Literal[
        "levenshtein", "hamming", "jaro", "jaro_winkler", "damerau_levenshtein"
    ] = "levenshtein",
    normalize: bool = True,
    name: str = "distance",
) -> "Scorer[t.Any]":
    """
    Score the distance between data and reference text using RapidFuzz distance metrics.

    Lower distance values indicate higher similarity. When normalize=True, distances
    are converted to similarity scores (1 - normalized_distance).

    Requires `rapidfuzz`, see See https://github.com/rapidfuzz/RapidFuzz

    Args:
        reference: The reference text (static string).
        method: The distance metric to use.
        normalize: Normalize distances and convert to similarity scores.
        name: Name of the scorer.
    """
    with catch_import_error("dreadnode[scoring]"):
        from rapidfuzz import distance

    def evaluate(  # noqa: PLR0912
        data: t.Any,
        *,
        reference: str = reference,
        method: t.Literal[
            "levenshtein", "hamming", "jaro", "jaro_winkler", "damerau_levenshtein"
        ] = method,
        normalize: bool = normalize,
    ) -> Metric:
        candidate_text = str(data)

        # Select the appropriate distance method
        if method == "levenshtein":
            if normalize:
                score = distance.Levenshtein.normalized_similarity(reference, candidate_text)
            else:
                dist = distance.Levenshtein.distance(reference, candidate_text)
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
        elif method == "hamming":
            if normalize:
                score = distance.Hamming.normalized_similarity(reference, candidate_text)
            else:
                dist = distance.Hamming.distance(reference, candidate_text)
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
        elif method == "jaro":
            score = distance.Jaro.similarity(reference, candidate_text)
        elif method == "jaro_winkler":
            score = distance.JaroWinkler.similarity(reference, candidate_text)
        elif method == "damerau_levenshtein":
            if normalize:
                score = distance.DamerauLevenshtein.normalized_similarity(reference, candidate_text)
            else:
                dist = distance.DamerauLevenshtein.distance(reference, candidate_text)
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
        elif normalize:
            score = distance.Levenshtein.normalized_similarity(reference, candidate_text)
        else:
            dist = distance.Levenshtein.distance(reference, candidate_text)
            score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0

        return Metric(value=float(score), attributes={"method": method, "normalize": normalize})

    return Scorer(evaluate, name=name)


@functools.lru_cache(maxsize=1)
def tf_idf_vectorizer() -> "TfidfVectorizer":
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(stop_words="english")


def similarity_with_tf_idf(reference: str, *, name: str = "similarity") -> "Scorer[t.Any]":
    """
    Scores semantic similarity using TF-IDF and cosine similarity.

    Requires `scikit-learn`, see https://scikit-learn.org

    Args:
        reference: The reference text (e.g., expected output).
        name: Name of the scorer.
    """
    with catch_import_error("dreadnode[scoring]"):
        from sklearn.metrics.pairwise import (  # type: ignore[import-not-found]
            cosine_similarity as sklearn_cosine_similarity,
        )

    vectorizer = tf_idf_vectorizer()

    def evaluate(data: t.Any, *, reference: str = reference) -> Metric:
        candidate_text = str(data)
        tfidf_matrix = vectorizer.fit_transform([candidate_text, reference])
        sim = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return Metric(value=float(sim))

    return Scorer(evaluate, name=name)


# A global model cache to avoid reloading on every call
g_sentence_transformers_models: dict[str, "SentenceTransformer"] = {}


def similarity_with_sentence_transformers(
    reference: str,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    name: str = "similarity",
) -> "Scorer[t.Any]":
    """
    Scores semantic similarity using a sentence-transformer embedding model.

    This is a more robust alternative to TF-IDF or sequence matching, as it
    understands the meaning of words and sentences. The score is the
    cosine similarity between the reference and candidate text embeddings.

    Requires `sentence-transformers`, see https://huggingface.co/sentence-transformers.

    Args:
        reference: The reference text (e.g., expected output).
        model_name: The name of the sentence-transformer model to use.
        name: Name of the scorer.
    """
    with catch_import_error("dreadnode[scoring]"):
        from sentence_transformers import (
            SentenceTransformer,
            util,
        )

    def evaluate(
        data: t.Any, *, reference: str = reference, model_name: str = Config(model_name)
    ) -> Metric:
        # Lazily load and cache the model
        if model_name not in g_sentence_transformers_models:
            g_sentence_transformers_models[model_name] = SentenceTransformer(model_name)
        model = g_sentence_transformers_models[model_name]

        candidate_text = str(data)

        embeddings = model.encode([candidate_text, reference])
        sim_tensor = util.cos_sim(embeddings[0], embeddings[1])
        return Metric(
            value=float(sim_tensor[0][0]),
            attributes={
                "model": model_name,
            },
        )

    return Scorer(evaluate, name=name)


def similarity_with_litellm(
    reference: str,
    model: str,
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

    Requires `litellm`, see https://docs.litellm.ai/docs/

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

    async def evaluate(
        data: t.Any,
        *,
        reference: str = reference,
        model: str = Config(model),
        api_key: str | None = Config(api_key),
        api_base: str | None = Config(api_base),
    ) -> Metric:
        candidate_text = str(data)
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

    return Scorer(evaluate, name=name)


def bleu(
    reference: str,
    *,
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    name: str = "bleu",
) -> "Scorer[t.Any]":
    """
    Scores the data using the BLEU score against a reference text.

    A score of 1.0 indicates a perfect match.

    Requires `nltk`, see https://www.nltk.org.

    Args:
        reference: The reference text (e.g., the prompt).
        weights: Weights for unigram, bigram, etc. Must sum to 1.
        name: Name of the scorer.
    """
    with catch_import_error("dreadnode[scoring]"):
        import nltk  # type: ignore[import-not-found]
        from nltk.tokenize import word_tokenize  # type: ignore[import-not-found]
        from nltk.translate.bleu_score import sentence_bleu  # type: ignore[import-not-found]
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        nltk_import_error_msg = (
            "NLTK 'punkt' tokenizer not found. Please run: python -m nltk.downloader punkt"
        )
        raise LookupError(nltk_import_error_msg) from e

    def evaluate(
        data: t.Any, *, reference: str = reference, weights: tuple[float, ...] = weights
    ) -> Metric:
        candidate_text = str(data)

        if not reference or not candidate_text:
            return Metric(value=0.0, attributes={"error": "Reference or candidate text is empty."})

        ref_tokens = word_tokenize(reference)
        cand_tokens = word_tokenize(candidate_text)

        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        return Metric(value=score)

    return Scorer(evaluate, name=name)
