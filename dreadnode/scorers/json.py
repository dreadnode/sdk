import typing as t

from jsonpath_ng import parse

from dreadnode.metric import Metric
from dreadnode.scorers.base import Scorer


def json_path(path: str, default_value: float = 0.0) -> Scorer[t.Any]:
    """
    Extracts a numeric value from a JSON-like object (dict/list) using a JSONPath query.
    """

    def evaluate(data: t.Any, *, path: str = path, default_value: float = default_value) -> Metric:
        jsonpath_expr = parse(path)
        matches = jsonpath_expr.find(data)
        if not matches:
            return Metric(value=default_value, attributes={"path_found": False})

        # Return the value of the first match found
        try:
            first_value = matches[0].value
            score = float(first_value)
            return Metric(value=score, attributes={"path_found": True})
        except (ValueError, TypeError):
            # If the value isn't numeric, we can't score it. Return default.
            return Metric(
                value=default_value, attributes={"path_found": True, "error": "Value not numeric"}
            )

    return Scorer(evaluate, name="json_path")
