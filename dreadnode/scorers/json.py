import typing as t

import jsonpath

from dreadnode.metric import Metric
from dreadnode.scorers.base import Scorer


def json_path(
    expression: str, *, default: float | None = None, name: str = "json_path"
) -> Scorer[t.Any]:
    """
    Extracts a numeric value from a JSON-like object (dict/list) using a JSONPath query.

    See: https://jg-rp.github.io/python-jsonpath/syntax/

    Args:
        expression: The JSONPath expression.
        default: The default value to return if the expression is not found or not numeric.
                  If None, the scorer will raise an error when the expression is not found.
    """

    def evaluate(data: t.Any, *, path: str = expression, default: float | None = default) -> Metric:
        matches = jsonpath.findall(path, data)
        if not matches:
            if default is None:
                raise ValueError(f"JSON path '{path}' not found in data and no default provided.")
            return Metric(value=default, attributes={"path_found": False})

        # Return the value of the first match found
        try:
            first_value = matches[0]
            score = float(first_value)  # type: ignore[arg-type]
            return Metric(value=score, attributes={"path_found": True})
        except (ValueError, TypeError) as e:
            if default is None:
                raise ValueError(
                    f"Value at JSON path '{path}' is not numeric and no default provided."
                ) from e
            return Metric(
                value=default, attributes={"path_found": True, "error": "Value not numeric"}
            )

    return Scorer(evaluate, name=name)
