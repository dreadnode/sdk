import codecs
import string

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def atbash_cipher(*, name: str = "atbash") -> Transform[str, str]:
    """Encodes text using the Atbash cipher."""

    def reverse(alphabet: str) -> str:
        return alphabet[::-1]

    def transform(text: str) -> str:
        alphabet = (string.ascii_lowercase, string.ascii_uppercase, string.digits)
        reversed_alphabet = tuple(map(reverse, alphabet))
        translation_table = str.maketrans("".join(alphabet), "".join(reversed_alphabet))
        return text.translate(translation_table)

    return Transform(transform, name=name)


def caesar_cipher(offset: int, *, name: str = "caesar") -> Transform[str, str]:
    """Encodes text using the Caesar cipher."""

    if not -25 <= offset <= 25:
        raise ValueError("Caesar offset must be between -25 and 25.")

    def transform(
        text: str, *, offset: int = Config(offset, ge=-25, le=25, help="The cipher offset")
    ) -> str:
        def shift(alphabet: str) -> str:
            return alphabet[offset:] + alphabet[:offset]

        alphabet = (string.ascii_lowercase, string.ascii_uppercase, string.digits)
        shifted_alphabet = tuple(map(shift, alphabet))
        translation_table = str.maketrans("".join(alphabet), "".join(shifted_alphabet))
        return text.translate(translation_table)

    return Transform(transform, name=name)


def rot13_cipher(*, name: str = "rot13") -> Transform[str, str]:
    """Encodes text using the ROT13 cipher."""

    def transform(text: str) -> str:
        return codecs.encode(text, "rot13")

    return Transform(transform, name=name)


def rot47_cipher(*, name: str = "rot47") -> Transform[str, str]:
    """Encodes text using the ROT47 cipher."""

    def transform(text: str) -> str:
        transformed = []
        for char in text:
            char_ord = ord(char)
            if 33 <= char_ord <= 126:
                shifted_ord = char_ord + 47
                if shifted_ord > 126:
                    shifted_ord -= 94
                transformed.append(chr(shifted_ord))
            else:
                transformed.append(char)
        return "".join(transformed)

    return Transform(transform, name=name)
