from dreadnode.meta import Config
from dreadnode.transforms.base import Transform
from dreadnode.util import catch_import_error


def ascii_art(font: str = "rand", *, name: str = "ascii_art") -> Transform[str, str]:
    """Converts text into ASCII art using the 'art' library."""

    with catch_import_error("dreadnode[scoring]"):
        from art import text2art  # type: ignore[import-not-found]

    def transform(text: str, *, font: str = Config(font, help="The font to use")) -> str:
        return str(text2art(text, font=font))

    return Transform(transform, name=name)
