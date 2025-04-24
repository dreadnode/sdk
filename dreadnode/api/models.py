import typing as t
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field
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
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str | None
    status: SpanStatus
    exception: SpanException | None
    name: str
    attributes: AnyDict
    resource_attributes: AnyDict
    events: list[SpanEvent]
    links: list[SpanLink]


class Metric(BaseModel):
    value: float
    step: int
    timestamp: datetime
    attributes: AnyDict


class ObjectRef(BaseModel):
    name: str
    label: str
    hash: str


class ObjectUri(BaseModel):
    hash: str
    schema_hash: str
    uri: str
    size: int
    type: t.Literal["uri"]


class ObjectVal(BaseModel):
    hash: str
    schema_hash: str
    value: t.Any
    type: t.Literal["val"]


Object = ObjectUri | ObjectVal


class V0Object(BaseModel):
    name: str
    label: str
    value: t.Any


class Run(BaseModel):
    id: ULID
    name: str
    span_id: str
    trace_id: str
    timestamp: datetime
    duration: int
    status: SpanStatus
    exception: SpanException | None
    tags: set[str]
    params: AnyDict
    metrics: dict[str, list[Metric]]
    inputs: list[ObjectRef]
    outputs: list[ObjectRef]
    objects: dict[str, Object]
    object_schemas: AnyDict
    schema_: AnyDict = Field(alias="schema")


class Task(BaseModel):
    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None
    parent_task_span_id: str | None
    timestamp: datetime
    duration: int
    status: SpanStatus
    exception: SpanException | None
    tags: set[str]
    params: AnyDict
    metrics: dict[str, list[Metric]]
    inputs: list[ObjectRef] | list[V0Object]  # v0 compat
    outputs: list[ObjectRef] | list[V0Object]  # v0 compat
    schema_: AnyDict = Field(alias="schema")
    attributes: AnyDict
    resource_attributes: AnyDict
    events: list[SpanEvent]
    links: list[SpanLink]


class Project(BaseModel):
    id: UUID
    key: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run: Run | None


# Derived types


class TaskTree(BaseModel):
    task: Task
    children: list["TaskTree"] = []


class SpanTree(BaseModel):
    """Tree representation of a trace span with its children"""

    span: Task | TraceSpan
    children: list["SpanTree"] = []


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
