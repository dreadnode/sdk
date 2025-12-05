import codecs
import random
import string
import typing as t

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


def vigenere_cipher(
    key: str,
    *,
    name: str = "vigenere",
) -> Transform[str, str]:
    """
    Encodes text using the Vigenère cipher.

    A polyalphabetic substitution cipher using a keyword.
    More secure than Caesar cipher due to multiple shift values.

    Args:
        key: The keyword to use for encoding.
        name: Name of the transform.
    """
    if not key or not key.isalpha():
        raise ValueError("Key must be a non-empty alphabetic string.")

    def transform(
        text: str,
        *,
        key: str = Config(key, help="The cipher key"),
    ) -> str:
        result = []
        key_lower = key.lower()
        key_length = len(key_lower)
        key_index = 0

        for char in text:
            if char.isalpha():
                # Get shift amount from key
                shift = ord(key_lower[key_index % key_length]) - ord("a")

                if char.islower():
                    shifted = chr((ord(char) - ord("a") + shift) % 26 + ord("a"))
                else:
                    shifted = chr((ord(char) - ord("A") + shift) % 26 + ord("A"))

                result.append(shifted)
                key_index += 1
            else:
                result.append(char)

        return "".join(result)

    return Transform(transform, name=name)


def substitution_cipher(
    key: str | None = None,
    *,
    seed: int | None = None,
    name: str = "substitution",
) -> Transform[str, str]:
    """
    Encodes text using a substitution cipher with custom or random key.

    Maps each letter to another letter according to a substitution key.
    If no key provided, generates a random substitution.

    Args:
        key: 26-letter substitution key (None for random).
        seed: Random seed if generating random key.
        name: Name of the transform.
    """

    def generate_random_key(seed: int | None) -> str:
        rand = random.Random(seed)  # noqa: S311  # nosec
        letters = list(string.ascii_lowercase)
        rand.shuffle(letters)
        return "".join(letters)

    if key is not None:
        if len(key) != 26 or not key.isalpha():
            raise ValueError("Key must be exactly 26 alphabetic characters.")
        key = key.lower()
    else:
        key = generate_random_key(seed)

    def transform(text: str) -> str:
        translation_table = str.maketrans(
            string.ascii_lowercase + string.ascii_uppercase, key + key.upper()
        )
        return text.translate(translation_table)

    return Transform(transform, name=name)


def xor_cipher(
    key: str,
    *,
    output_format: t.Literal["hex", "base64", "raw"] = "hex",
    name: str = "xor_cipher",
) -> Transform[str, str]:
    """
    Encodes text using XOR cipher with a repeating key.

    Tests XOR-based encoding, commonly used in malware obfuscation.

    Args:
        key: The XOR key (will be repeated to match text length).
        output_format: How to format the output.
        name: Name of the transform.
    """
    import base64

    if not key:
        raise ValueError("Key cannot be empty.")

    def transform(
        text: str,
        *,
        key: str = Config(key, help="The XOR key"),
        output_format: t.Literal["hex", "base64", "raw"] = Config(
            output_format, help="Output format"
        ),
    ) -> str:
        text_bytes = text.encode("utf-8")
        key_bytes = key.encode("utf-8")

        xored = bytes(
            [text_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(text_bytes))]
        )

        if output_format == "hex":
            return xored.hex()
        if output_format == "base64":
            return base64.b64encode(xored).decode("ascii")
        # raw
        return xored.decode("latin-1")

    return Transform(transform, name=name)


def rail_fence_cipher(
    rails: int = 3,
    *,
    name: str = "rail_fence",
) -> Transform[str, str]:
    """
    Encodes text using the Rail Fence cipher (zigzag pattern).

    A transposition cipher that writes text in a zigzag pattern.
    Tests pattern-based obfuscation.

    Args:
        rails: Number of rails (rows) to use.
        name: Name of the transform.
    """
    if rails < 2:
        raise ValueError("Number of rails must be at least 2.")

    def transform(
        text: str,
        *,
        rails: int = Config(rails, ge=2, help="Number of rails"),
    ) -> str:
        if rails >= len(text):
            return text

        # Create rail fence pattern
        fence: list[list[str]] = [[] for _ in range(rails)]
        rail = 0
        direction = 1

        for char in text:
            fence[rail].append(char)
            rail += direction

            # Change direction at top and bottom
            if rail == 0 or rail == rails - 1:
                direction = -direction

        # Read off the rails
        return "".join("".join(rail) for rail in fence)

    return Transform(transform, name=name)


def columnar_transposition(
    key: str,
    *,
    name: str = "columnar_transposition",
) -> Transform[str, str]:
    """
    Encodes text using columnar transposition cipher.

    Writes text in rows and reads in column order based on key.
    Tests position-based obfuscation.

    Args:
        key: The keyword that determines column order.
        name: Name of the transform.
    """
    if not key:
        raise ValueError("Key cannot be empty.")

    def transform(
        text: str,
        *,
        key: str = Config(key, help="The transposition key"),
    ) -> str:
        # Remove spaces for cleaner output
        text_clean = text.replace(" ", "")

        # Create column order based on alphabetical key order
        key_order = sorted(range(len(key)), key=lambda k: key[k])

        # Calculate number of rows needed
        num_cols = len(key)
        num_rows = (len(text_clean) + num_cols - 1) // num_cols

        # Pad text if necessary
        padded_text = text_clean.ljust(num_rows * num_cols, "X")

        # Create grid
        grid = [padded_text[i : i + num_cols] for i in range(0, len(padded_text), num_cols)]

        # Read columns in key order
        result = []
        for col_idx in key_order:
            for row in grid:
                if col_idx < len(row):
                    result.append(row[col_idx])  # noqa: PERF401

        return "".join(result)

    return Transform(transform, name=name)


def playfair_cipher(
    key: str = "KEYWORD",
    *,
    name: str = "playfair",
) -> Transform[str, str]:
    """
    Encodes text using the Playfair cipher.

    A digraph substitution cipher using a 5x5 key matrix.
    Tests complex substitution patterns.

    Args:
        key: The keyword for generating the cipher matrix.
        name: Name of the transform.
    """

    def create_matrix(key: str) -> list[list[str]]:
        # Create 5x5 matrix from key (I/J treated as same)
        key_clean = "".join(dict.fromkeys(key.upper().replace("J", "I")))
        key_clean = "".join(c for c in key_clean if c.isalpha())

        alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # pragma: allowlist secret
        matrix_str = key_clean + "".join(c for c in alphabet if c not in key_clean)

        return [list(matrix_str[i : i + 5]) for i in range(0, 25, 5)]

    def find_position(matrix: list[list[str]], char: str) -> tuple[int, int]:
        for i, row in enumerate(matrix):
            for j, c in enumerate(row):
                if c == char:
                    return i, j
        return 0, 0

    def transform(
        text: str,
        *,
        key: str = Config(key, help="The cipher key"),
    ) -> str:
        matrix = create_matrix(key)

        # Prepare text: remove non-alpha, uppercase, replace J with I
        text_clean = "".join(c.upper() for c in text if c.isalpha()).replace("J", "I")

        # Split into digraphs
        digraphs = []
        i = 0
        while i < len(text_clean):
            a = text_clean[i]
            b = text_clean[i + 1] if i + 1 < len(text_clean) else "X"

            # If letters are same, insert X
            if a == b:
                digraphs.append((a, "X"))
                i += 1
            else:
                digraphs.append((a, b))
                i += 2

        # Encode digraphs
        result = []
        for a, b in digraphs:
            row_a, col_a = find_position(matrix, a)
            row_b, col_b = find_position(matrix, b)

            if row_a == row_b:
                # Same row: shift right
                result.append(matrix[row_a][(col_a + 1) % 5])
                result.append(matrix[row_b][(col_b + 1) % 5])
            elif col_a == col_b:
                # Same column: shift down
                result.append(matrix[(row_a + 1) % 5][col_a])
                result.append(matrix[(row_b + 1) % 5][col_b])
            else:
                # Rectangle: swap columns
                result.append(matrix[row_a][col_b])
                result.append(matrix[row_b][col_a])

        return "".join(result)

    return Transform(transform, name=name)


def affine_cipher(
    a: int = 5,
    b: int = 8,
    *,
    name: str = "affine",
) -> Transform[str, str]:
    """
    Encodes text using the Affine cipher.

    Combines multiplicative and additive ciphers: E(x) = (ax + b) mod 26
    Tests mathematical transformations.

    Args:
        a: Multiplicative key (must be coprime with 26).
        b: Additive key (0-25).
        name: Name of the transform.
    """
    import math

    if math.gcd(a, 26) != 1:
        raise ValueError("Parameter 'a' must be coprime with 26.")
    if not 0 <= b <= 25:
        raise ValueError("Parameter 'b' must be between 0 and 25.")

    def transform(
        text: str,
        *,
        a: int = Config(a, help="Multiplicative key"),
        b: int = Config(b, ge=0, le=25, help="Additive key"),
    ) -> str:
        result = []
        for char in text:
            if char.isalpha():
                if char.islower():
                    x = ord(char) - ord("a")
                    encrypted = (a * x + b) % 26
                    result.append(chr(encrypted + ord("a")))
                else:
                    x = ord(char) - ord("A")
                    encrypted = (a * x + b) % 26
                    result.append(chr(encrypted + ord("A")))
            else:
                result.append(char)

        return "".join(result)

    return Transform(transform, name=name)


def bacon_cipher(
    *,
    variant: t.Literal["distinct", "standard"] = "standard",
    name: str = "bacon",
) -> Transform[str, str]:
    """
    Encodes text using Bacon's cipher.

    Encodes each letter as a 5-bit binary pattern using A and B.
    Tests binary pattern encoding.

    Args:
        variant: "distinct" uses unique codes for I/J and U/V, "standard" doesn't.
        name: Name of the transform.
    """
    # Standard Bacon cipher (I/J and U/V are same)
    standard_codes = {
        "A": "AAAAA",
        "B": "AAAAB",
        "C": "AAABA",
        "D": "AAABB",
        "E": "AABAA",
        "F": "AABAB",
        "G": "AABBA",
        "H": "AABBB",
        "I": "ABAAA",
        "J": "ABAAA",
        "K": "ABAAB",
        "L": "ABABA",
        "M": "ABABB",
        "N": "ABBAA",
        "O": "ABBAB",
        "P": "ABBBA",
        "Q": "ABBBB",
        "R": "BAAAA",
        "S": "BAAAB",
        "T": "BAABA",
        "U": "BAABB",
        "V": "BAABB",
        "W": "BABAA",
        "X": "BABAB",
        "Y": "BABBA",
        "Z": "BABBB",
    }

    # Distinct codes for all 26 letters
    distinct_codes = {
        "A": "AAAAA",
        "B": "AAAAB",
        "C": "AAABA",
        "D": "AAABB",
        "E": "AABAA",
        "F": "AABAB",
        "G": "AABBA",
        "H": "AABBB",
        "I": "ABAAA",
        "J": "ABAAB",
        "K": "ABABA",
        "L": "ABABB",
        "M": "ABBAA",
        "N": "ABBAB",
        "O": "ABBBA",
        "P": "ABBBB",
        "Q": "BAAAA",
        "R": "BAAAB",
        "S": "BAABA",
        "T": "BAABB",
        "U": "BABAA",
        "V": "BABAB",
        "W": "BABBA",
        "X": "BABBB",
        "Y": "BBAAA",
        "Z": "BBAAB",
    }

    def transform(
        text: str,
        *,
        variant: t.Literal["distinct", "standard"] = Config(variant, help="Cipher variant"),
    ) -> str:
        codes = distinct_codes if variant == "distinct" else standard_codes
        result = []

        for char in text:
            if char.isalpha():
                result.append(codes[char.upper()])
            else:
                result.append(char)

        return " ".join(result)

    return Transform(transform, name=name)


def autokey_cipher(
    key: str,
    *,
    name: str = "autokey",
) -> Transform[str, str]:
    """
    Encodes text using the Autokey cipher.

    Similar to Vigenère but uses the plaintext itself as part of the key.
    More secure than Vigenère due to non-repeating key.

    Args:
        key: Initial key (plaintext is appended to it).
        name: Name of the transform.
    """
    if not key or not key.isalpha():
        raise ValueError("Key must be a non-empty alphabetic string.")

    def transform(
        text: str,
        *,
        key: str = Config(key, help="Initial cipher key"),
    ) -> str:
        result = []
        key_stream = key.lower()
        key_index = 0

        for char in text:
            if char.isalpha():
                # Get shift from key stream
                shift = ord(key_stream[key_index]) - ord("a")

                if char.islower():
                    shifted = chr((ord(char) - ord("a") + shift) % 26 + ord("a"))
                    key_stream += char  # Add plaintext to key
                else:
                    shifted = chr((ord(char) - ord("A") + shift) % 26 + ord("A"))
                    key_stream += char.lower()

                result.append(shifted)
                key_index += 1
            else:
                result.append(char)

        return "".join(result)

    return Transform(transform, name=name)


def beaufort_cipher(
    key: str,
    *,
    name: str = "beaufort",
) -> Transform[str, str]:
    """
    Encodes text using the Beaufort cipher.

    Similar to Vigenère but uses subtraction instead of addition.
    Reciprocal cipher (encoding and decoding are the same operation).

    Args:
        key: The cipher key.
        name: Name of the transform.
    """
    if not key or not key.isalpha():
        raise ValueError("Key must be a non-empty alphabetic string.")

    def transform(
        text: str,
        *,
        key: str = Config(key, help="The cipher key"),
    ) -> str:
        result = []
        key_lower = key.lower()
        key_length = len(key_lower)
        key_index = 0

        for char in text:
            if char.isalpha():
                # Beaufort: E(x) = (key - plaintext) mod 26
                key_char = ord(key_lower[key_index % key_length]) - ord("a")

                if char.islower():
                    plain_char = ord(char) - ord("a")
                    encrypted = (key_char - plain_char) % 26
                    result.append(chr(encrypted + ord("a")))
                else:
                    plain_char = ord(char) - ord("A")
                    encrypted = (key_char - plain_char) % 26
                    result.append(chr(encrypted + ord("A")))

                key_index += 1
            else:
                result.append(char)

        return "".join(result)

    return Transform(transform, name=name)
