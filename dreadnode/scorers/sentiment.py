import os
import typing as t

import httpx

from dreadnode.metric import Metric, Scorer
from dreadnode.util import warn_at_user_stacklevel

_TEXTBLOB_AVAILABLE = False
_TEXTBLOB_ERROR_MSG = "textblob dependency is not installed. Please run: pip install textblob && python -m textblob.download_corpora"

try:
    from textblob import TextBlob  # type: ignore[import-not-found,unused-ignore,import-untyped]

    _TEXTBLOB_AVAILABLE = True
except ImportError:
    pass


def sentiment(
    target: t.Literal["positive", "negative", "neutral"] = "neutral",
    name: str = "score_sentiment",
) -> "Scorer[t.Any]":
    """
    Score the sentiment of the text against a target sentiment.

    The score indicates how well the text's sentiment matches the target.
    - For "positive", score is 0-1 (0=negative, 1=very positive).
    - For "negative", score is 0-1 (0=positive, 1=very negative).
    - For "neutral", score is 0-1 (1=perfectly neutral, 0=very polarized).

    Args:
        target: The desired sentiment to score against.
        name: Name of the scorer.
    """
    if not _TEXTBLOB_AVAILABLE:
        warn_at_user_stacklevel(_TEXTBLOB_ERROR_MSG, UserWarning)

        def disabled_evaluate(_: t.Any) -> Metric:
            return Metric(value=0.0, attributes={"error": _TEXTBLOB_ERROR_MSG})

        return Scorer.from_callable(disabled_evaluate, name=name)

    def evaluate(data: t.Any) -> Metric:
        text = str(data)
        if not text.strip():
            return Metric(value=0.0, attributes={"error": "Input text is empty."})

        blob = TextBlob(text)
        # Polarity is a float from -1.0 (very negative) to 1.0 (very positive)
        polarity = blob.sentiment.polarity

        score = 0.0
        if target == "positive":
            # Scale [0, 1] to [0, 1]
            score = max(0.0, polarity)
        elif target == "negative":
            # Scale [-1, 0] to [0, 1]
            score = max(0.0, -polarity)
        else:  # "neutral"
            # Score is 1 minus the absolute distance from 0
            score = 1.0 - abs(polarity)

        return Metric(value=score, attributes={"polarity": polarity, "target": target})

    return Scorer.from_callable(evaluate, name=name)


PerspectiveAttribute = t.Literal[
    "TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"
]


def sentiment_with_perspective(
    *,
    api_key: str | None = None,
    attribute: PerspectiveAttribute = "TOXICITY",
    name: str | None = None,
) -> Scorer[t.Any]:
    """
    Score the sentiment of the text using the Perspective API.

    Returns a float score between 0.0 and 1.0 indicating the level of the attribute in the text.

    Args:
        api_key: Your Perspective API key, or set in the PERSPECTIVE_API_KEY environment variable.
        attribute: The attribute to analyze (e.g., TOXICITY, SEVERE_TOXICITY).
        name: Name of the scorer.
    """

    api_key = api_key or os.getenv("PERSPECTIVE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key must be provided or set in the PERSPECTIVE_API_KEY environment variable."
        )

    async def evaluate(data: t.Any) -> float:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze",
                params={"key": api_key},
                json={
                    "comment": {"text": str(data)},
                    "languages": ["en"],
                    "requestedAttributes": {attribute: {}},
                    "doNotStore": True,
                },
                timeout=10,
            )
        response.raise_for_status()
        result = await response.json()
        return float(result["attributeScores"][attribute]["summaryScore"]["value"])

    if name is None:
        name = f"perspective_{attribute.lower()}"

    return Scorer.from_callable(evaluate, name=name, catch=True)
