import base64
import html
import urllib.parse

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def ascii85_encode(*, name: str = "ascii85") -> Transform[str, str]:
    """Encodes text to ASCII85."""

    def transform(text: str) -> str:
        return base64.a85encode(text.encode("utf-8")).decode("ascii")

    return Transform(transform, name=name)


def base32_encode(*, name: str = "base32") -> Transform[str, str]:
    """Encodes text to Base32."""

    def transform(text: str) -> str:
        return base64.b32encode(text.encode("utf-8")).decode("ascii")

    return Transform(transform, name=name)


def base64_encode(*, name: str = "base64") -> Transform[str, str]:
    """Encodes text to Base64."""

    def transform(text: str) -> str:
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    return Transform(transform, name=name)


def binary_encode(bits_per_char: int = 16, *, name: str = "binary") -> Transform[str, str]:
    """Converts text into its binary representation."""

    def transform(
        text: str,
        *,
        bits_per_char: int = Config(bits_per_char, help="The number of bits per character"),
    ) -> str:
        max_code_point = max((ord(char) for char in text), default=0)
        min_bits_required = max_code_point.bit_length()
        if bits_per_char < min_bits_required:
            raise ValueError(
                f"bits_per_char={bits_per_char} is too small. Minimum required: {min_bits_required}."
            )
        return " ".join(format(ord(char), f"0{bits_per_char}b") for char in text)

    return Transform(transform, name=name)


def hex_encode(*, name: str = "hex") -> Transform[str, str]:
    """Encodes text to its hexadecimal representation."""

    def transform(text: str) -> str:
        return text.encode("utf-8").hex().upper()

    return Transform(transform, name=name)


def html_escape(*, name: str = "html_escape") -> Transform[str, str]:
    """Converts special characters to their HTML entities."""

    def transform(text: str) -> str:
        return html.escape(text, quote=True)

    return Transform(transform, name=name)


def url_encode(*, name: str = "url_encode") -> Transform[str, str]:
    """URL-encodes text."""

    def transform(text: str) -> str:
        return urllib.parse.quote(text)

    return Transform(transform, name=name)
