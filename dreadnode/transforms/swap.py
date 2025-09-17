import random
import re
import typing as t

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def swap(
    *,
    unit: t.Literal["char", "word"] = "char",
    mode: t.Literal["adjacent", "random"] = "adjacent",
    ratio: float = 0.1,
    seed: int | None = None,
    name: str = "general_swap",
) -> Transform[str, str]:
    """
    Swaps text units (characters or words) in a string.

    Args:
        unit: The unit of text to operate on ('char' or 'word').
        mode: 'adjacent' swaps with neighbors, 'random' swaps with any other unit.
        ratio: The proportion of units to select for swapping (0.0 to 1.0).
        seed: Seed for the random number generator.
        name: The name of the transform.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0.")

    rand = random.Random(seed)  # noqa: S311 # nosec

    def transform(
        text: str,
        *,
        ratio: float = Config(
            ratio,
            ge=0.0,
            le=1.0,
            help="The proportion of words/chars to select for swapping (0.0 to 1.0).",
        ),
    ) -> str:
        items = list(text) if unit == "char" else re.findall(r"\w+|\S+", text)
        if len(items) < 2:
            return text

        num_to_swap = int(len(items) * ratio)
        indices_to_swap = rand.sample(range(len(items)), k=num_to_swap)

        for i in indices_to_swap:
            if mode == "adjacent":
                # Swap with the next item, wrapping around at the end
                neighbor_idx = (i + 1) % len(items)
                items[i], items[neighbor_idx] = items[neighbor_idx], items[i]
            elif mode == "random":
                # Swap with any other random item
                swap_with_idx = rand.choice([j for j in range(len(items)) if i != j])
                items[i], items[swap_with_idx] = items[swap_with_idx], items[i]

        separator = "" if unit == "char" else " "
        result = separator.join(items)
        if unit == "word":
            return re.sub(r'\s([?.!,"\'`])', r"\1", result).strip()
        return result

    return Transform(transform, name=name)


def adjacent_char_swap(
    *,
    ratio: float = 0.1,
    seed: int | None = None,
    name: str = "adjacent_char_swap",
) -> Transform[str, str]:
    """
    Perturbs text by swapping a ratio of adjacent characters.

    Args:
        ratio: The proportion of characters to swap (0.0 to 1.0).
        seed: Seed for the random number generator.
        name: The name of the transform.
    """
    return swap(unit="char", mode="adjacent", ratio=ratio, seed=seed, name=name)


def random_word_reorder(
    *,
    ratio: float = 0.1,
    seed: int | None = None,
    name: str = "random_word_reorder",
) -> Transform[str, str]:
    """
    Randomly reorders a ratio of words within the text.

    Args:
        ratio: The proportion of words to reorder (0.0 to 1.0).
        seed: Seed for the random number generator.
        name: The name of the transform.
    """
    return swap(unit="word", mode="random", ratio=ratio, seed=seed, name=name)
