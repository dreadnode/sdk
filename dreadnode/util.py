import typing as t


def safe_repr(obj: t.Any) -> str:
    """
    Return some kind of non-empty string representation of an object, catching exceptions.

    Taken from `logfire`.
    """

    try:
        result = repr(obj)
    except Exception:  # noqa: BLE001
        result = ""

    if result:
        return result

    try:
        return f"<{type(obj).__name__} object>"
    except Exception:  # noqa: BLE001
        return "<unknown (repr failed)>"
