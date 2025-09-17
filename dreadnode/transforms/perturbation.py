import random
import string
import typing as t
import unicodedata

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform
from dreadnode.util import catch_import_error


def random_capitalization(
    *,
    ratio: float = 0.2,
    seed: int | None = None,
    name: str = "random_capitalization",
) -> Transform[str, str]:
    """
    Randomly capitalizes a ratio of lowercase letters in text.

    Args:
        ratio: The ratio of lowercase letters to capitalize (0.0 to 1.0).
        seed: Random seed for reproducibility.
        name: Name of the transform.
    """

    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Capitalization ratio must be between 0.0 and 1.0.")

    rand = random.Random(seed)  # noqa: S311  # nosec

    def transform(
        text: str,
        *,
        ratio: float = Config(
            ratio, ge=0.0, le=1.0, help="The ratio of lowercase letters to capitalize"
        ),
    ) -> str:
        chars = list(text)
        indices = [i for i, char in enumerate(chars) if "a" <= char <= "z"]
        num_to_capitalize = int(len(indices) * ratio)
        indices_to_capitalize = rand.sample(indices, k=num_to_capitalize)
        for i in indices_to_capitalize:
            chars[i] = chars[i].upper()
        return "".join(chars)

    return Transform(transform, name=name)


def insert_punctuation(
    *,
    ratio: float = 0.2,
    punctuations: list[str] | None = None,
    seed: int | None = None,
    name: str = "insert_punctuation",
) -> Transform[str, str]:
    """
    Inserts punctuation randomly between words in text.

    Args:
        ratio: The ratio of word pairs to insert punctuation between (0.0 to 1.0).
        punctuations: A list of custom punctuation characters to use (default: all ASCII punctuation).
        seed: Random seed for reproducibility.
        name: Name of the transform.
    """

    if not 0.0 < ratio <= 1.0:
        raise ValueError("Insertion ratio must be between 0.0 and 1.0.")

    rand = random.Random(seed)  # noqa: S311  # nosec
    punctuations = punctuations or list(string.punctuation)

    def transform(
        text: str,
        *,
        ratio: float = Config(
            ratio,
            ge=0.0,
            le=1.0,
            help="The ratio of word pairs to insert punctuation between",
        ),
    ) -> str:
        words = text.split()
        if not words:
            return text
        num_to_insert = max(1, round(len(words) * ratio))
        indices = rand.sample(range(len(words)), k=min(len(words), num_to_insert))

        for i in sorted(indices, reverse=True):
            punc = rand.choice(punctuations)
            if rand.choice([True, False]):
                words[i] = punc + words[i]
            else:
                words[i] = words[i] + punc
        return " ".join(words)

    return Transform(transform, name=name)


def diacritic(
    target_chars: str = "aeiou",
    accent: t.Literal["acute", "grave", "tilde", "umlaut"] = "acute",
    *,
    name: str = "diacritic",
) -> Transform[str, str]:
    """
    Applies diacritics (accent marks) to specified characters in text.

    Args:
        target_chars: The characters to apply diacritics to.
        accent: The type of accent to apply.
        name: Name of the transform.
    """
    diacritics = {
        "acute": "\u0301",
        "grave": "\u0300",
        "tilde": "\u0303",
        "umlaut": "\u0308",
    }

    def transform(
        text: str,
        *,
        target_chars: str = Config(target_chars, help="The characters to apply diacritics to"),
        accent: str = Config(accent, help="The type of accent to apply"),
    ) -> str:
        accent_mark = diacritics[accent]
        target_set = set(target_chars.lower())
        return "".join(
            # Normalize with NFC to correctly combine characters and accents
            unicodedata.normalize("NFC", char + accent_mark) if char.lower() in target_set else char
            for char in text
        )

    return Transform(transform, name=name or f"diacritic_{accent}")


def underline(*, name: str = "underline") -> Transform[str, str]:
    """Adds an underline effect to each character using Unicode combining characters."""

    def transform(text: str) -> str:
        return "".join(char + "\u0332" for char in text)

    return Transform(transform, name=name)


def character_space(*, name: str = "character_space") -> Transform[str, str]:
    """Spaces out all characters and removes common punctuation."""

    def transform(text: str) -> str:
        punctuation_to_remove = str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        text_no_punc = text.translate(punctuation_to_remove)
        return " ".join(text_no_punc)

    return Transform(transform, name=name)


def zero_width(*, name: str = "zero_width") -> Transform[str, str]:
    """Injects zero-width spaces between every character in the text."""

    def transform(text: str) -> str:
        return "\u200b".join(text)

    return Transform(transform, name=name)


def zalgo(
    intensity: int = 10,
    *,
    ratio: float = 1.0,
    seed: int | None = None,
    name: str | None = None,
) -> Transform[str, str]:
    """
    Converts text into 'zalgo' text by adding random combining characters.

    Args:
        intensity: The intensity of the zalgo effect (0-100).
        ratio: The ratio of characters to apply the effect to (0.0-1.0).
        seed: Random seed for reproducibility.
        name: Name of the transform.
    """
    if not 0 <= intensity <= 100:
        raise ValueError("Intensity must be between 0 and 100.")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Application ratio must be between 0.0 and 1.0.")

    # Unicode combining diacritical marks range
    zalgo_marks = [chr(code) for code in range(0x0300, 0x036F + 1)]
    rand = random.Random(seed)  # noqa: S311  # nosec

    def transform(
        text: str,
        *,
        intensity: int = Config(intensity, ge=0, le=100, help="The intensity of the zalgo effect"),
        ratio: float = Config(
            ratio, ge=0.0, le=1.0, help="The ratio of characters to apply the effect to"
        ),
    ) -> str:
        if intensity == 0 or ratio == 0.0:
            return text

        chars = list(text)
        # Identify indices of alphanumeric characters eligible for zalgo
        eligible_indices = [i for i, char in enumerate(chars) if char.isalnum()]
        num_to_apply = int(len(eligible_indices) * ratio)
        indices_to_apply = rand.sample(eligible_indices, k=num_to_apply)

        for i in indices_to_apply:
            num_marks = rand.randint(1, intensity)
            zalgo_chars = "".join(rand.choices(zalgo_marks, k=num_marks))
            chars[i] += zalgo_chars

        return "".join(chars)

    return Transform(transform, name=name or f"zalgo_{intensity}")


def unicode_confusable(
    *,
    ratio: float = 1.0,
    deterministic: bool = False,
    seed: int | None = None,
    name: str = "unicode_confusable",
) -> Transform[str, str]:
    """
    Replaces characters with visually similar Unicode characters (homoglyphs).

    Args:
        ratio: The ratio of characters to apply the effect to (0.0-1.0).
        deterministic: Whether to use a deterministic random seed.
        seed: Random seed for reproducibility.
        name: Name of the transform.
    """

    with catch_import_error("dreadnode[scoring]"):
        from confusables import confusable_characters  # type: ignore[import-not-found]

    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Application ratio must be between 0.0 and 1.0.")

    rand = random.Random(seed)  # noqa: S311  # nosec

    def transform(
        text: str,
        *,
        ratio: float = Config(
            ratio, ge=0.0, le=1.0, help="The ratio of characters to apply the effect to"
        ),
        deterministic: bool = Config(
            deterministic, help="Whether to always take the first replacement option"
        ),
    ) -> str:
        chars = list(text)
        eligible_indices = [i for i, char in enumerate(chars) if confusable_characters(char)]
        num_to_apply = int(len(eligible_indices) * ratio)
        indices_to_apply = rand.sample(eligible_indices, k=num_to_apply)

        for i in indices_to_apply:
            options = confusable_characters(chars[i])
            if options:
                # The original character is the first in the list
                replacement_options = options[1:]
                if replacement_options:
                    if deterministic:
                        chars[i] = replacement_options[0]
                    else:
                        chars[i] = rand.choice(replacement_options)
        return "".join(chars)

    return Transform(transform, name=name)


def unicode_replacement(
    *, encode_spaces: bool = False, name: str = "unicode_replacement"
) -> Transform[str, str]:
    """
    Converts text to its Unicode escape sequence representation (e.g., 'A' -> '\\u0041').

    Args:
        encode_spaces: Whether to encode spaces as Unicode escape sequences.
        name: Name of the transform.
    """

    def transform(text: str) -> str:
        result = "".join(f"\\u{ord(ch):04x}" for ch in text)
        if not encode_spaces:
            result = result.replace("\\u0020", " ")
        return result

    return Transform(transform, name=name)


def unicode_substitution(
    *, start_value: int = 0xE0000, name: str = "unicode_substitution"
) -> Transform[str, str]:
    """
    Substitutes characters with Unicode characters from a specified private use area.

    Args:
        start_value: The starting Unicode code point for the substitution.
        name: Name of the transform.
    """

    def transform(text: str) -> str:
        return "".join(chr(start_value + ord(ch)) for ch in text)

    return Transform(transform, name=name)
