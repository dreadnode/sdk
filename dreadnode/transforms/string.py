import random
import re
import typing as t

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def reverse(*, name: str = "reverse") -> Transform[str, str]:
    """Reverses the order of characters in a string."""

    def transform(text: str) -> str:
        return text[::-1]

    return Transform(transform, name=name)


def search_replace(
    pattern: str | re.Pattern[str],
    replacement: str | list[str],
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    seed: int | None = None,
    deterministic: bool = False,
    name: str = "search_replace",
) -> Transform[str, str]:
    """
    Replaces text matching a literal string or a regex pattern.

    Args:
        pattern: String or compiled regex pattern to search for.
        replacement: The string or list of strings to use for replacement.
        regex: If True, the string `pattern` is treated as a regex.
               This is ignored if `pattern` is already a compiled re.Pattern.
        case_sensitive: If False, matching is case-insensitive.
        seed: Seed for the random number generator for reproducibility.
        deterministic: If True, always picks the first replacement option from a list.
        name: The name of the transform.
    """
    rand = random.Random(seed)  # noqa: S311  # nosec
    replace_list = [replacement] if isinstance(replacement, str) else replacement

    def transform(text: str) -> str:
        if deterministic or len(replace_list) == 1:
            chosen_replacement = replace_list[0]
        else:
            chosen_replacement = rand.choice(replace_list)

        is_regex_mode = regex or isinstance(pattern, re.Pattern)

        if is_regex_mode:
            re_flags = 0 if case_sensitive else re.IGNORECASE
            return re.sub(pattern, chosen_replacement, text, flags=re_flags)

        if case_sensitive:
            return text.replace(t.cast("str", pattern), chosen_replacement)

        return re.sub(
            re.escape(t.cast("str", pattern)),
            chosen_replacement,
            text,
            flags=re.IGNORECASE,
        )

    return Transform(transform, name=name)


def join(
    delimiter: str,
    *,
    unit: t.Literal["char", "word"] = "char",
    name: str = "join",
) -> Transform[str, str]:
    """
    Joins the units (characters or words) of a string with a delimiter.

    Args:
        delimiter: The string to insert between each unit.
        unit: The unit of text to operate on ('char' or 'word').
        name: The name of the transform.
    """

    def transform(
        text: str,
        *,
        delimiter: str = Config(delimiter, help="The string to insert between each unit"),
    ) -> str:
        items = list(text) if unit == "char" else text.split()
        return delimiter.join(items)

    return Transform(transform, name=name)


def char_join(delimiter: str = "-", *, name: str = "char_join") -> Transform[str, str]:
    """
    Joins each character of a string with a delimiter.

    Args:
        delimiter: The string to insert between each character.
    """
    return join(delimiter, unit="char", name=name)


def word_join(delimiter: str = "-", *, name: str = "word_join") -> Transform[str, str]:
    """
    Joins each word of a string with a delimiter.

    Args:
        delimiter: The string to insert between each word.
    """
    return join(delimiter, unit="word", name=name)


def affix(
    text_to_add: str,
    *,
    position: t.Literal["prefix", "suffix"] = "prefix",
    delimiter: str = " ",
    name: str = "affix",
) -> Transform[str, str]:
    """
    Adds text as a prefix or suffix to the input string.

    Args:
        text_to_add: The string to be added.
        position: 'prefix' to add to the beginning, 'suffix' to add to the end.
        delimiter: The string used to join the original and new text. Use "" for none.
        name: The name of the transform.
    """
    if not text_to_add:
        raise ValueError("Text to add cannot be empty.")

    def transform(
        text: str,
        *,
        delimiter: str = Config(
            delimiter, help="The string used to join the original and new text"
        ),
        position: t.Literal["prefix", "suffix"] = Config(
            position, help="The position to add the text"
        ),
    ) -> str:
        if position == "prefix":
            return text_to_add + delimiter + text
        return text + delimiter + text_to_add

    return Transform(transform, name=name)


def prefix(text: str, *, name: str = "prefix") -> Transform[str, str]:
    """Prepends a specified prefix to the input text with a space."""
    return affix(text, position="prefix", delimiter=" ", name=name)


def suffix(text: str, *, name: str = "suffix") -> Transform[str, str]:
    """Appends a specified suffix to the input text with a space."""
    return affix(text, position="suffix", delimiter=" ", name=name)
