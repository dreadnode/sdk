import typing as t

if t.TYPE_CHECKING:
    from dreadnode.metric import Metric


class AssertionFailedError(Exception):
    """Raised when a task's output fails one or more assertions."""

    def __init__(self, message: str, failures: "dict[str, list[Metric]]"):
        """
        Args:
            message: The overall exception message.
            failures: A dictionary mapping the name of each failed assertion
                      to the list of Metrics that caused the failure
                      (or an empty list if the metric was not produced).
        """
        super().__init__(message)
        self.failures = failures
