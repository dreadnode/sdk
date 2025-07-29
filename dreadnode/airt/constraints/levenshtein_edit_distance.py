from rapidfuzz import fuzz

from dreadnode.airt.constraints.base import Constraint


def levenshtein_edit_distance(
    max_edit_distance: int = 30,
    compare_against_original: bool = True,
) -> Constraint:
    """Create a Levenshtein edit distance constraint.

    Args:
        max_edit_distance (int): Maximum edit distance allowed.
        compare_against_original (bool): If `True`, compare new text against the original text.
            Otherwise, compare it against the previous text.

    Returns:
        Constraint: A function that checks the edit distance constraint.
    """
    if not isinstance(max_edit_distance, int):
        raise TypeError("max_edit_distance must be an int")

    def constraint(transformed_text: str, reference_text: str) -> bool:
        """Check if edit distance is within the allowed limit."""
        edit_distance = fuzz.distance(reference_text.text, transformed_text.text)
        return edit_distance <= max_edit_distance

    # Add attributes to the function for introspection
    constraint.max_edit_distance = max_edit_distance
    constraint.compare_against_original = compare_against_original
    constraint.__name__ = "LevenshteinEditDistance"

    return constraint
