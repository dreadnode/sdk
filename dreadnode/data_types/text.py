import typing as t

from dreadnode.data_types.base import DataType


class Text(DataType):
    """
    Text data type for Dreadnode logging.
    """

    def __init__(self, text: str, format: str):
        """
        Initialize a Text object.

        Args:
            text: The text content to log
            format: The format hint of the text
        """
        self._text = text
        self._format = format

    def to_serializable(self) -> tuple[str, dict[str, t.Any]]:
        return self._text, {"format": self._format}


class Markdown(Text):
    """
    Hint type for markdown-formatted text.

    This is a subclass of Text with format set to "markdown".

    Example:
        ```
        log_output("report", Markdown("..."))
        ```
    """

    def __init__(self, text: str):
        super().__init__(text, format="markdown")


class Code(Text):
    """
    Hint type for code-formatted text.

    This is a subclass of Text with format set to "code".

    Example:
        ```
        log_output("code_snippet", Code("print('Hello, World!')", language="python"))
        ```
    """

    def __init__(self, text: str, language: str = ""):
        super().__init__(text, format="code")
        self._language = language

    def to_serializable(self) -> tuple[str, dict[str, t.Any]]:
        return self._text, {"format": self._format, "code-language": self._language}
