import typing as t


@t.runtime_checkable
class Constraint(t.Protocol):
    """Protocol defining the constraint interface."""

    def __call__(self, transformed_text: str, reference_text: str) -> bool:
        """Check if the constraint is satisfied."""
