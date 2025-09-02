import json
import typing as t
import xml.etree.ElementTree as ET  # nosec

from dreadnode.metric import Metric
from dreadnode.scorers import Scorer


def is_json(*, name: str = "is_json") -> "Scorer[t.Any]":
    """
    Scores whether the data is a valid JSON string.

    The score is 1.0 if the string can be successfully parsed as JSON,
    and 0.0 otherwise. The error message is included in the attributes.

    Args:
        name: Name of the scorer.
    """

    def evaluate(data: t.Any) -> Metric:
        text = str(data).strip()

        if text.startswith("```json\n"):
            text = text[10:]
        text = text.removeprefix("```")
        text = text.removesuffix("\n```")

        try:
            json.loads(text)
            return Metric(value=1.0)
        except json.JSONDecodeError as e:
            return Metric(value=0.0, attributes={"error": str(e)})

    return Scorer(evaluate, name=name)


def is_xml(*, name: str = "is_xml") -> "Scorer[t.Any]":
    """
    Scores whether the data is a valid XML string.

    The score is 1.0 if the string can be successfully parsed as XML,
    and 0.0 otherwise. The error message is included in the attributes.

    Args:
        name: Name of the scorer.
    """

    def evaluate(data: t.Any) -> Metric:
        text = str(data).strip()

        if text.startswith("```xml\n"):
            text = text[10:]
        text = text.removeprefix("```")
        text = text.removesuffix("\n```")

        try:
            ET.fromstring(text)  # noqa: S314 # nosec
            return Metric(value=1.0)
        except ET.ParseError as e:
            return Metric(value=0.0, attributes={"error": str(e)})

    return Scorer(evaluate, name=name)
