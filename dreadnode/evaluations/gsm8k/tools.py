"""GSM8K tools for math problem solving."""

import dreadnode as dn


@dn.tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        The result of the calculation as a string.
    """
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters."

        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e!s}"


@dn.tool
def submit_answer(answer: float) -> str:
    """
    Submit the final numeric answer to the problem.

    When used with an agent, combine with a stop_condition like:
        `stop_conditions=[tool_use("submit_answer")]`

    Args:
        answer: The final numeric answer to submit.

    Returns:
        Confirmation of the submitted answer.
    """
    return f"Answer submitted: {answer}"
