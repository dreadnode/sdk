import typing as t
from dataclasses import dataclass

from dreadnode.types import JsonDict


@dataclass
class ObjectRef:
    name: str
    label: str
    hash: str
    attributes: JsonDict


@dataclass
class ObjectUri:
    hash: str
    schema_hash: str
    uri: str
    size: int
    type: t.Literal["uri"] = "uri"


@dataclass
class ObjectVal:
    hash: str
    schema_hash: str
    value: t.Any
    type: t.Literal["val"] = "val"


Object = ObjectUri | ObjectVal
