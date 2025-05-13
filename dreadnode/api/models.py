import contextlib
import typing as t
from datetime import datetime
from functools import cached_property
from uuid import UUID

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from ulid import ULID

AnyDict = dict[str, t.Any]

# User


class UserAPIKey(BaseModel):
    key: str


class UserResponse(BaseModel):
    id: UUID
    email_address: str
    username: str
    api_key: UserAPIKey


# Strikes

SpanStatus = t.Literal[
    "pending",  # A pending span has been created
    "completed",  # The span has been finished
    "failed",  # The raised an exception
]

ExportFormat = t.Literal["csv", "json", "jsonl", "parquet"]
StatusFilter = t.Literal["all", "completed", "failed"]
TimeAxisType = t.Literal["wall", "relative", "step"]
TimeAggregationType = t.Literal["max", "min", "sum", "count"]
MetricAggregationType = t.Literal[
    "avg",
    "median",
    "min",
    "max",
    "sum",
    "first",
    "last",
    "count",
    "std",
    "var",
]


class SpanException(BaseModel):
    type: str
    message: str
    stacktrace: str


class SpanEvent(BaseModel):
    timestamp: datetime
    name: str
    attributes: AnyDict


class SpanLink(BaseModel):
    trace_id: str
    span_id: str
    attributes: AnyDict


class TraceLog(BaseModel):
    timestamp: datetime
    body: str
    severity: str
    service: str | None
    trace_id: str | None
    span_id: str | None
    attributes: AnyDict
    container: str | None


class TraceSpan(BaseModel):
    timestamp: datetime
    duration: int
    trace_id: str = Field(repr=False)
    span_id: str
    parent_span_id: str | None = Field(repr=False)
    service_name: str | None = Field(repr=False)
    status: SpanStatus
    exception: SpanException | None
    name: str
    attributes: AnyDict = Field(repr=False)
    resource_attributes: AnyDict = Field(repr=False)
    events: list[SpanEvent] = Field(repr=False)
    links: list[SpanLink] = Field(repr=False)


class Metric(BaseModel):
    value: float
    step: int
    timestamp: datetime
    attributes: AnyDict


class ObjectRef(BaseModel):
    name: str
    label: str
    hash: str


class RawObjectUri(BaseModel):
    hash: str
    schema_hash: str
    uri: str
    size: int
    type: t.Literal["uri"]


class RawObjectVal(BaseModel):
    hash: str
    schema_hash: str
    value: t.Any
    type: t.Literal["val"]


RawObject = RawObjectUri | RawObjectVal


class V0Object(BaseModel):
    name: str
    label: str
    value: t.Any


class ObjectVal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    label: str
    hash: str = Field(repr=False)
    schema_: AnyDict
    schema_hash: str = Field(repr=False)
    value: t.Any

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            with contextlib.suppress(ValidationError):
                return TypeAdapter(t.Any).validate_json(value)

        return value


class ObjectUri(BaseModel):
    name: str
    label: str
    hash: str = Field(repr=False)
    schema_: AnyDict
    schema_hash: str = Field(repr=False)
    uri: str
    size: int

    _value: t.Any = PrivateAttr(default=None)

    @cached_property
    def value(self) -> t.Any:
        if self._value is not None:
            return self._value

        try:
            response = requests.get(self.uri, timeout=5)
            response.raise_for_status()
            self._value = response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch object from {self.uri}") from e

        if isinstance(self._value, str):
            with contextlib.suppress(ValidationError):
                self._value = TypeAdapter(t.Any).validate_json(self._value)

        return self._value


Object = ObjectVal | ObjectUri


class ArtifactFile(BaseModel):
    hash: str
    uri: str
    size_bytes: int
    final_real_path: str


class ArtifactDir(BaseModel):
    dir_path: str
    hash: str
    children: list[t.Union["ArtifactDir", ArtifactFile]]


class RunSummary(BaseModel):
    id: ULID
    name: str
    span_id: str = Field(repr=False)
    trace_id: str = Field(repr=False)
    timestamp: datetime
    duration: int
    status: SpanStatus
    exception: SpanException | None
    tags: set[str]
    params: AnyDict = Field(repr=False)
    metrics: dict[str, list[Metric]] = Field(repr=False)


class RawRun(RunSummary):
    inputs: list[ObjectRef] = Field(repr=False)
    outputs: list[ObjectRef] = Field(repr=False)
    objects: dict[str, RawObject] = Field(repr=False)
    object_schemas: AnyDict = Field(repr=False)
    artifacts: list[ArtifactDir] = Field(repr=False)
    schema_: AnyDict = Field(alias="schema", repr=False)


class Run(RunSummary):
    inputs: dict[str, Object] = Field(repr=False)
    outputs: dict[str, Object] = Field(repr=False)
    artifacts: list[ArtifactDir] = Field(repr=False)
    schema_: AnyDict = Field(alias="schema", repr=False)


class _Task(BaseModel):
    name: str
    span_id: str
    trace_id: str = Field(repr=False)
    parent_span_id: str | None = Field(repr=False)
    parent_task_span_id: str | None = Field(repr=False)
    timestamp: datetime
    duration: int
    status: SpanStatus
    exception: SpanException | None
    tags: set[str]
    params: AnyDict = Field(repr=False)
    metrics: dict[str, list[Metric]] = Field(repr=False)
    schema_: AnyDict = Field(alias="schema", repr=False)
    attributes: AnyDict = Field(repr=False)
    resource_attributes: AnyDict = Field(repr=False)
    events: list[SpanEvent] = Field(repr=False)
    links: list[SpanLink] = Field(repr=False)


class RawTask(_Task):
    inputs: list[ObjectRef] | list[V0Object] = Field(repr=False)
    outputs: list[ObjectRef] | list[V0Object] = Field(repr=False)


class Task(_Task):
    inputs: dict[str, Object] = Field(repr=False)
    outputs: dict[str, Object] = Field(repr=False)


class Project(BaseModel):
    id: UUID = Field(repr=False)
    key: str
    name: str
    description: str | None = Field(repr=False)
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run: RawRun | None = Field(repr=False)


# Derived types


class TaskTree(BaseModel):
    task: Task
    children: list["TaskTree"] = []


class TraceTree(BaseModel):
    span: Task | TraceSpan
    children: list["TraceTree"] = []


# User data credentials


class UserDataCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: datetime
    region: str
    bucket: str
    prefix: str
    endpoint: str | None
