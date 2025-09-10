import typing as t
from dataclasses import dataclass

import typing_extensions as te
from pydantic import PlainSerializer, WithJsonSchema

T = t.TypeVar("T")

# Common types

Primitive = int | float | str | bool | None
JsonValue = te.TypeAliasType(
    "JsonValue",
    "Primitive | list[JsonValue] | tuple[JsonValue, ...] | JsonDict",
)
JsonDict = te.TypeAliasType("JsonDict", dict[str, JsonValue])
AnyDict = dict[str, t.Any]


@dataclass
class Arguments:
    """
    Represents the arguments passed to a function or task.
    Contains both positional and keyword arguments.
    """

    args: tuple[t.Any, ...]
    kwargs: dict[str, t.Any]


class Unset:
    def __bool__(self) -> t.Literal[False]:
        return False


UNSET: t.Any = Unset()


class Inherited: ...


INHERITED: t.Any = Inherited()


ErrorField = t.Annotated[
    BaseException,
    PlainSerializer(
        lambda x: str(x),
        return_type=str,
        when_used="json-unless-none",
    ),
    WithJsonSchema({"type": "string", "description": "Error message"}),
]

# from annotated_types


class SupportsGt(t.Protocol):
    def __gt__(self, __other: te.Self) -> bool: ...  # noqa: PYI063


class SupportsGe(t.Protocol):
    def __ge__(self, __other: te.Self) -> bool: ...  # noqa: PYI063


class SupportsLt(t.Protocol):
    def __lt__(self, __other: te.Self) -> bool: ...  # noqa: PYI063


class SupportsLe(t.Protocol):
    def __le__(self, __other: te.Self) -> bool: ...  # noqa: PYI063
