import typing as t

# Common types

JsonValue = t.Union[
    int,
    float,
    str,
    bool,
    None,
    list["JsonValue"],
    tuple["JsonValue", ...],
    "JsonDict",
]
JsonDict = dict[str, JsonValue]

AnyDict = dict[str, t.Any]


class Unset:
    def __bool__(self) -> t.Literal[False]:
        return False


UNSET: Unset = Unset()
