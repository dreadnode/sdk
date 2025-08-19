import typing as t
from dataclasses import dataclass

import typing_extensions as te
from pydantic import PlainSerializer, WithJsonSchema

# Common types

JsonValue = te.TypeAliasType(
    "JsonValue",
    "int | float | str | bool | None | list[JsonValue] | tuple[JsonValue, ...] | JsonDict",
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


UNSET: Unset = Unset()


class Inherited:
    def __repr__(self) -> str:
        return "Inherited"


INHERITED: Inherited = Inherited()


ErrorField = t.Annotated[
    BaseException,
    PlainSerializer(
        lambda x: str(x),
        return_type=str,
        when_used="json-unless-none",
    ),
    WithJsonSchema({"type": "string", "description": "Error message"}),
]
