import typing as t

import openai

from dreadnode.metric import Metric, Scorer


def detect_harm_with_openai(
    *,
    api_key: str | None = None,
    model: t.Literal["text-moderation-stable", "text-moderation-latest"] = "text-moderation-stable",
    client: openai.AsyncOpenAI | None = None,
    name: str = "openai_harm",
) -> "Scorer[t.Any]":
    """
    Scores content for harmfulness using the OpenAI Moderation API.

    The primary score (value) of the metric is the *highest* score received
    across all moderation categories. A score of 0.0 indicates no detected harm,
    while a score approaching 1.0 indicates high confidence of harm.

    The metric's attributes contain a detailed breakdown of all category scores
    and whether the content was flagged by OpenAI.

    Requires the `openai` python package.

    Args:
        api_key: Your OpenAI API key. If not provided, it will be sourced from the
                 `OPENAI_API_KEY` environment variable.
        model: The moderation model to use.
        name: Name of the scorer.
    """

    async def evaluate(data: t.Any) -> Metric:
        text = str(data)
        _client = client or openai.AsyncOpenAI(api_key=api_key)

        if not text.strip():
            return Metric(value=0.0, attributes={"error": "Input text is empty."})

        response = await _client.moderations.create(input=text, model=model)
        result = response.results[0]

        # The main value is the max score across all categories
        category_scores = result.category_scores.model_dump()
        max_score = max(category_scores.values())

        attributes = {
            "flagged": result.flagged,
            "model": model,
            **category_scores,
        }
        return Metric(value=max_score, attributes=attributes)

    return Scorer.from_callable(evaluate, name=name, catch=True)
