from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def ascii_art(font: str = "rand", *, name: str = "ascii_art") -> Transform[str, str]:
    """Converts text into ASCII art using the 'art' library."""

    try:
        from art import text2art  # type: ignore[import-not-found,unused-ignore,import-untyped]
    except ImportError as e:
        raise ImportError("ascii_art requires 'art'. Install with: pip install art") from e

    def transform(text: str, *, font: str = Config(font, help="The font to use")) -> str:
        return str(text2art(text, font=font))

    return Transform(transform, name=name)
