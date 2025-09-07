import typing as t

from dreadnode.lazy.core import LazyAttr, LazyImport

if t.TYPE_CHECKING:
    import litellm as litellm  # type: ignore[import-not-found]
    import nltk as nltk  # type: ignore[import-not-found]
    from nltk.tokenize import (  # type: ignore[import-not-found]
        word_tokenize as word_tokenize,
    )
    from nltk.translate.bleu_score import (  # type: ignore[import-not-found]
        sentence_bleu as sentence_bleu,
    )
    from rapidfuzz import distance as distance  # type: ignore[import-not-found]
    from rapidfuzz import fuzz as fuzz  # type: ignore[import-not-found]
    from rapidfuzz import utils as utils  # type: ignore[import-not-found]
    from sentence_transformers import (  # type: ignore[import-not-found]
        SentenceTransformer as SentenceTransformer,
    )
    from sentence_transformers import (  # type: ignore[import-not-found]
        util as util,
    )
    from sklearn.feature_extraction.text import (  # type: ignore[import-not-found]
        TfidfVectorizer as TfidfVectorizer,
    )
    from sklearn.metrics.pairwise import (  # type: ignore[import-not-found]
        cosine_similarity as cosine_similarity,
    )
else:
    fuzz = LazyAttr("rapidfuzz", "fuzz", "text")
    utils = LazyAttr("rapidfuzz", "utils", "text")
    distance = LazyAttr("rapidfuzz", "distance", "text")
    litellm = LazyImport("litellm", "llm")
    util = LazyAttr("sentence_transformers", "util", "text", package_name="sentence-transformers")
    TfidfVectorizer = LazyAttr(
        "sklearn.feature_extraction.text", "TfidfVectorizer", "text", package_name="scikit-learn"
    )
    SentenceTransformer = LazyAttr(
        "sentence_transformers", "SentenceTransformer", "text", package_name="sentence-transformers"
    )
    cosine_similarity = LazyAttr(
        "sklearn.metrics.pairwise", "cosine_similarity", "text", package_name="scikit-learn"
    )
    nltk = LazyImport("nltk", "text")
    word_tokenize = LazyAttr("nltk.tokenize", "word_tokenize", "text", package_name="nltk")
    sentence_bleu = LazyAttr(
        "nltk.translate.bleu_score", "sentence_bleu", "text", package_name="nltk"
    )
