import typing as t
from dataclasses import dataclass

from pydantic import BaseModel, Field

from dreadnode.types import AnyDict


@dataclass
class ObjectRef:
    name: str
    label: str
    hash: str
    attributes: AnyDict | None


class ObjectUri(BaseModel):
    hash: str
    schema_hash: str
    uri: str
    size: int
    type: t.Literal["uri"] = "uri"

    # During execution, we might want to dynamically pull a value
    # in it's unserialized form, so we store it here.
    runtime_value: t.Any | None = Field(None, init=False, repr=False, exclude=True)

    @property
    def value(self) -> t.Any:
        return self.runtime_value or self.uri


class ObjectVal(BaseModel):
    hash: str
    schema_hash: str
    value_: t.Any = Field(alias="value")
    type: t.Literal["val"] = "val"

    # During execution, we might want to dynamically pull a value
    # in it's unserialized form, so we store it here.
    runtime_value: t.Any | None = Field(None, init=False, repr=False, exclude=True)

    @property
    def value(self) -> t.Any:
        return self.runtime_value or self.value_


Object = ObjectUri | ObjectVal
