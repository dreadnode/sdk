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
