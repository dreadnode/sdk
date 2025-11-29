import base64
import html
import json
import random
import typing as t
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


def unicode_escape(
    *,
    encode_spaces: bool = False,
    format_style: t.Literal["\\u", "\\U", "\\x"] = "\\u",
    name: str = "unicode_escape",
) -> Transform[str, str]:
    """
    Converts text to Unicode escape sequences.

    Useful for testing Unicode handling and bypassing text-based filters.

    Args:
        encode_spaces: If True, also encode spaces as escape sequences.
        format_style: The escape sequence format to use.
        name: Name of the transform.
    """

    def transform(
        text: str,
        *,
        encode_spaces: bool = Config(encode_spaces, help="Whether to encode spaces"),
        format_style: t.Literal["\\u", "\\U", "\\x"] = Config(
            format_style, help="Escape sequence format"
        ),
    ) -> str:
        result = []
        for char in text:
            if char == " " and not encode_spaces:
                result.append(" ")
            else:
                code_point = ord(char)
                if format_style == "\\u":
                    result.append(f"\\u{code_point:04x}")
                elif format_style == "\\U":
                    result.append(f"\\U{code_point:08x}")
                else:  # \\x
                    result.append(f"\\x{code_point:02x}")

        return "".join(result)

    return Transform(transform, name=name)


def json_encode(
    *,
    ensure_ascii: bool = True,
    name: str = "json_encode",
) -> Transform[str, str]:
    """
    Encodes text as a JSON string.

    Tests JSON parsing and escaping behavior. Useful for testing
    injection vulnerabilities in JSON-based APIs.

    Args:
        ensure_ascii: If True, escape non-ASCII characters.
        name: Name of the transform.
    """

    def transform(
        text: str,
        *,
        ensure_ascii: bool = Config(ensure_ascii, help="Whether to escape non-ASCII characters"),
    ) -> str:
        return json.dumps(text, ensure_ascii=ensure_ascii)

    return Transform(transform, name=name)


def punycode_encode(*, name: str = "punycode") -> Transform[str, str]:
    """
    Encodes text using Punycode (used for internationalized domain names).

    Tests handling of IDN homograph attacks and punycode processing.
    """

    def transform(text: str) -> str:
        return text.encode("punycode").decode("ascii")

    return Transform(transform, name=name)


def quoted_printable_encode(*, name: str = "quoted_printable") -> Transform[str, str]:
    """
    Encodes text using Quoted-Printable encoding.

    Tests email encoding handling and = character processing.
    """
    import quopri

    def transform(text: str) -> str:
        return quopri.encodestring(text.encode("utf-8")).decode("ascii")

    return Transform(transform, name=name)


def base58_encode(*, name: str = "base58") -> Transform[str, str]:
    """
    Encodes text using Base58 (commonly used in cryptocurrencies).

    Tests handling of alternative encoding schemes.
    """
    # Base58 alphabet (Bitcoin variant)
    alphabet = (
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"  # pragma: allowlist secret
    )

    def transform(text: str) -> str:
        # Convert text to integer
        num = int.from_bytes(text.encode("utf-8"), "big")

        if num == 0:
            return alphabet[0]

        result = []
        while num > 0:
            num, remainder = divmod(num, 58)
            result.append(alphabet[remainder])

        # Add leading zeros
        for char in text:
            if char == "\x00":
                result.append(alphabet[0])
            else:
                break

        return "".join(reversed(result))

    return Transform(transform, name=name)


def percent_encoding(
    *,
    safe: str = "",
    double_encode: bool = False,
    name: str = "percent_encoding",
) -> Transform[str, str]:
    """
    Applies percent encoding (like URL encoding but customizable).

    Tests handling of percent-encoded payloads and double encoding attacks.

    Args:
        safe: Characters that should not be encoded.
        double_encode: If True, encode the result again.
        name: Name of the transform.
    """

    def transform(
        text: str,
        *,
        safe: str = Config(safe, help="Characters that should not be encoded"),
        double_encode: bool = Config(double_encode, help="Whether to double-encode"),
    ) -> str:
        encoded = urllib.parse.quote(text, safe=safe)
        if double_encode:
            encoded = urllib.parse.quote(encoded, safe="")
        return encoded

    return Transform(transform, name=name)


def html_entity_encode(
    *,
    encoding_type: t.Literal["named", "decimal", "hex", "mixed"] = "named",
    name: str = "html_entity_encode",
) -> Transform[str, str]:
    """
    Encodes text as HTML entities.

    Tests HTML entity handling and XSS filter bypasses.

    Args:
        encoding_type: Type of HTML entity encoding to use.
        name: Name of the transform.
    """

    def transform(
        text: str,
        *,
        encoding_type: t.Literal["named", "decimal", "hex", "mixed"] = Config(
            encoding_type, help="HTML entity encoding type"
        ),
    ) -> str:
        result = []
        for char in text:
            if encoding_type == "named":
                result.append(html.escape(char, quote=True))
            elif encoding_type == "decimal":
                result.append(f"&#{ord(char)};")
            elif encoding_type == "hex":
                result.append(f"&#x{ord(char):x};")
            else:  # mixed
                choice = random.choice(["named", "decimal", "hex"])  # noqa: S311 # nosec B311
                if choice == "named":
                    result.append(html.escape(char, quote=True))
                elif choice == "decimal":
                    result.append(f"&#{ord(char)};")
                else:
                    result.append(f"&#x{ord(char):x};")

        return "".join(result)

    return Transform(transform, name=name)


def octal_encode(*, name: str = "octal") -> Transform[str, str]:
    """
    Encodes text as octal escape sequences.

    Tests octal sequence handling in parsers and interpreters.
    """

    def transform(text: str) -> str:
        return "".join(f"\\{ord(char):03o}" for char in text)

    return Transform(transform, name=name)


def utf7_encode(*, name: str = "utf7") -> Transform[str, str]:
    """
    Encodes text using UTF-7 encoding.

    Tests UTF-7 handling, which has been used in XSS attacks.
    Note: UTF-7 is deprecated but still useful for testing.
    """

    def transform(text: str) -> str:
        # UTF-7 is not in standard library, so we'll use a basic implementation
        # This is a simplified version for ASCII-compatible text
        encoded = text.encode("utf-8")
        # Basic UTF-7 encoding simulation
        result = []
        for byte in encoded:
            if 32 <= byte <= 126 and byte not in (43, 92):  # printable ASCII except + and \
                result.append(chr(byte))
            else:
                # Use modified Base64
                result.append(f"+{base64.b64encode(bytes([byte])).decode('ascii').rstrip('=')}-")
        return "".join(result)

    return Transform(transform, name=name)


def base91_encode(*, name: str = "base91") -> Transform[str, str]:
    """
    Encodes text using Base91 (more efficient than Base64).

    Tests handling of non-standard encoding schemes.
    """
    # Base91 alphabet
    base91_alphabet = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        '0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~"'
    )

    def transform(text: str) -> str:
        data = text.encode("utf-8")
        result = []
        ebq = 0
        en = 0

        for byte in data:
            ebq |= byte << en
            en += 8
            if en > 13:
                ev = ebq & 8191
                if ev > 88:
                    ebq >>= 13
                    en -= 13
                else:
                    ev = ebq & 16383
                    ebq >>= 14
                    en -= 14
                result.append(base91_alphabet[ev % 91])
                result.append(base91_alphabet[ev // 91])

        if en > 0:
            result.append(base91_alphabet[ebq % 91])
            if en > 7 or ebq > 90:
                result.append(base91_alphabet[ebq // 91])

        return "".join(result)

    return Transform(transform, name=name)


def mixed_case_hex(*, name: str = "mixed_case_hex") -> Transform[str, str]:
    """
    Encodes text as hex with mixed case.

    Tests case-sensitivity in hex parsing, useful for filter bypass.
    """
    import random

    def transform(text: str) -> str:
        result = []
        for char in text:
            hex_val = f"{ord(char):02x}"
            # Randomly capitalize each hex digit
            mixed = "".join(c.upper() if random.random() < 0.5 else c.lower() for c in hex_val)  # noqa: S311 # nosec B311
            result.append(mixed)
        return "".join(result)

    return Transform(transform, name=name)


def backslash_escape(
    *,
    chars_to_escape: str = "\"'\\",
    name: str = "backslash_escape",
) -> Transform[str, str]:
    """
    Adds backslash escaping to specified characters.

    Tests string escaping and parsing in various contexts.

    Args:
        chars_to_escape: Characters to escape with backslashes.
        name: Name of the transform.
    """

    def transform(
        text: str,
        *,
        chars_to_escape: str = Config(chars_to_escape, help="Characters to escape"),
    ) -> str:
        result = []
        for char in text:
            if char in chars_to_escape:
                result.append(f"\\{char}")
            else:
                result.append(char)
        return "".join(result)

    return Transform(transform, name=name)


def zero_width_encode(
    *,
    encoding_type: t.Literal["binary", "ternary"] = "binary",
    name: str = "zero_width_encode",
) -> Transform[str, str]:
    """
    Encodes text using zero-width Unicode characters.

    Creates invisible text that may bypass visual inspection.
    Useful for steganography and filter bypass testing.

    Args:
        encoding_type: The encoding scheme to use.
        name: Name of the transform.
    """
    # Zero-width characters
    zwc_zero = "\u200b"  # Zero-width space
    zwc_one = "\u200c"  # Zero-width non-joiner
    zwc_two = "\u200d"  # Zero-width joiner

    def transform(
        text: str,
        *,
        encoding_type: t.Literal["binary", "ternary"] = Config(
            encoding_type, help="Encoding scheme"
        ),
    ) -> str:
        result = []
        for char in text:
            code_point = ord(char)

            if encoding_type == "binary":
                # Binary encoding using two zero-width chars
                binary = format(code_point, "016b")
                encoded = binary.replace("0", zwc_zero).replace("1", zwc_one)
                result.append(encoded)
            else:  # ternary
                # Ternary encoding using three zero-width chars
                ternary = []
                num = code_point
                while num > 0:
                    ternary.append(str(num % 3))
                    num //= 3
                ternary_str = "".join(reversed(ternary)) or "0"
                encoded = (
                    ternary_str.replace("0", zwc_zero).replace("1", zwc_one).replace("2", zwc_two)
                )
                result.append(encoded)

        return "".join(result)

    return Transform(transform, name=name)


def leetspeak_encoding(
    *,
    intensity: float = 0.5,
    include_numbers: bool = True,  # noqa: ARG001
    seed: int | None = None,
    name: str = "leetspeak_encoding",
) -> Transform[str, str]:
    """
    Encodes text using leetspeak substitutions.

    Tests character substitution handling and alternative representations.

    Args:
        intensity: How many characters to substitute (0.0 to 1.0).
        include_numbers: If True, also substitute numbers.
        seed: Random seed for reproducibility.
        name: Name of the transform.
    """
    import random

    leet_map = {
        "a": ["4", "@", "/-\\"],
        "b": ["8", "6", "|3"],
        "e": ["3"],
        "g": ["9", "6"],
        "i": ["1", "!", "|"],
        "l": ["1", "|", "7"],
        "o": ["0"],
        "s": ["5", "$", "z"],
        "t": ["7", "+"],
        "z": ["2"],
    }

    if not 0.0 <= intensity <= 1.0:
        raise ValueError("Intensity must be between 0.0 and 1.0.")

    rand = random.Random(seed)  # noqa: S311  # nosec

    def transform(
        text: str,
        *,
        intensity: float = Config(intensity, ge=0.0, le=1.0, help="Substitution intensity"),
    ) -> str:
        result = []
        for char in text:
            lower_char = char.lower()
            if lower_char in leet_map and rand.random() < intensity:
                result.append(rand.choice(leet_map[lower_char]))
            else:
                result.append(char)
        return "".join(result)

    return Transform(transform, name=name)


def morse_encode(
    *,
    include_spacing: bool = True,
    name: str = "morse_encode",
) -> Transform[str, str]:
    """
    Encodes text as Morse code.

    Tests alternative encoding handling and pattern recognition.

    Args:
        include_spacing: If True, use spaces and slashes for separation.
        name: Name of the transform.
    """
    morse_map = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "0": "-----",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        " ": "/",
    }

    def transform(
        text: str,
        *,
        include_spacing: bool = Config(include_spacing, help="Whether to include spacing"),
    ) -> str:
        text_clean = " ".join([line.strip() for line in text.splitlines()])
        morse_chars = [morse_map.get(char.upper(), "........") for char in text_clean]

        if include_spacing:
            return " ".join(morse_chars)
        return "".join(morse_chars)

    return Transform(transform, name=name)
