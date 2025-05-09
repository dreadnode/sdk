"""
This module defines the data models used in the Dreadnode API.

The models are implemented using Pydantic's `BaseModel` and include various
types for representing tasks, spans, metrics, objects, and user-related data.
These models are used for serialization, validation, and type enforcement
throughout the API.
"""

import typing as t
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field
from ulid import ULID

AnyDict = dict[str, t.Any]


class UserAPIKey(BaseModel):
    """Represents a user's API key."""

    key: str


class UserResponse(BaseModel):
    """Represents a response containing user information.

    Attributes:
        id (UUID): The unique identifier for the user.
        email_address (str): The user's email address.
        username (str): The user's username.
        api_key (UserAPIKey): The user's API key.
    """

    id: UUID
    email_address: str
    username: str
    api_key: UserAPIKey


SpanStatus = t.Literal[
    "pending",  # A pending span has been created
    "completed",  # The span has been finished
    "failed",  # The span raised an exception
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
    """Represents an exception that occurred during a span.

    Attributes:
        type (str): The type of the exception.
        message (str): The exception message.
        stacktrace (str): The stack trace of the exception.
    """

    type: str
    message: str
    stacktrace: str


class SpanEvent(BaseModel):
    """Represents an event within a span.

    Attributes:
        timestamp (datetime): The timestamp of the event.
        name (str): The name of the event.
        attributes (AnyDict): Additional attributes associated with the event.
    """

    timestamp: datetime
    name: str
    attributes: AnyDict


class SpanLink(BaseModel):
    """Represents a link between spans.

    Attributes:
        trace_id (str): The trace ID of the linked span.
        span_id (str): The span ID of the linked span.
        attributes (AnyDict): Additional attributes associated with the link.
    """

    trace_id: str
    span_id: str
    attributes: AnyDict


class TraceLog(BaseModel):
    """Represents a log entry within a trace.

    Attributes:
        timestamp (datetime): The timestamp of the log entry.
        body (str): The log message body.
        severity (str): The severity level of the log.
        service (str | None): The service associated with the log.
        trace_id (str | None): The trace ID associated with the log.
        span_id (str | None): The span ID associated with the log.
        attributes (AnyDict): Additional attributes for the log.
        container (str | None): The container associated with the log.
    """

    timestamp: datetime
    body: str
    severity: str
    service: str | None
    trace_id: str | None
    span_id: str | None
    attributes: AnyDict
    container: str | None


class TraceSpan(BaseModel):
    """Represents a span within a trace.

    Attributes:
        timestamp (datetime): The timestamp of the span.
        duration (int): The duration of the span in milliseconds.
        trace_id (str): The trace ID of the span.
        span_id (str): The span ID of the span.
        parent_span_id (str | None): The parent span ID, if any.
        service_name (str | None): The name of the service associated with the span.
        status (SpanStatus): The status of the span.
        exception (SpanException | None): The exception associated with the span, if any.
        name (str): The name of the span.
        attributes (AnyDict): Additional attributes for the span.
        resource_attributes (AnyDict): Resource-specific attributes for the span.
        events (list[SpanEvent]): A list of events associated with the span.
        links (list[SpanLink]): A list of links to other spans.
    """

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
    """Represents a metric associated with a task or span.

    Attributes:
        value (float): The value of the metric.
        step (int): The step number associated with the metric.
        timestamp (datetime): The timestamp of the metric.
        attributes (AnyDict): Additional attributes for the metric.
    """

    value: float
    step: int
    timestamp: datetime
    attributes: AnyDict


class ObjectRef(BaseModel):
    """Represents a reference to an object.

    Attributes:
        name (str): The name of the object.
        label (str): The label of the object.
        hash (str): The hash of the object.
    """

    name: str
    label: str
    hash: str


class ObjectUri(BaseModel):
    """Represents an object stored as a URI.

    Attributes:
        hash (str): The hash of the object.
        schema_hash (str): The schema hash of the object.
        uri (str): The URI of the object.
        size (int): The size of the object in bytes.
        type (Literal["uri"]): The type of the object, always "uri".
    """

    hash: str
    schema_hash: str
    uri: str
    size: int
    type: t.Literal["uri"]


class ObjectVal(BaseModel):
    """Represents an object stored as a value.

    Attributes:
        hash (str): The hash of the object.
        schema_hash (str): The schema hash of the object.
        value (Any): The value of the object.
        type (Literal["val"]): The type of the object, always "val".
    """

    hash: str
    schema_hash: str
    value: t.Any
    type: t.Literal["val"]


Object = ObjectUri | ObjectVal


class V0Object(BaseModel):
    """Represents a backward-compatible object.

    Attributes:
        name (str): The name of the object.
        label (str): The label of the object.
        value (Any): The value of the object.
    """

    name: str
    label: str
    value: t.Any


class Run(BaseModel):
    """Represents a run of a task or process.

    Attributes:
        id (ULID): The unique identifier for the run.
        name (str): The name of the run.
        span_id (str): The span ID associated with the run.
        trace_id (str): The trace ID associated with the run.
        timestamp (datetime): The timestamp of the run.
        duration (int): The duration of the run in milliseconds.
        status (SpanStatus): The status of the run.
        exception (SpanException | None): The exception associated with the run, if any.
        tags (set[str]): A set of tags associated with the run.
        params (AnyDict): Parameters associated with the run.
        metrics (dict[str, list[Metric]]): Metrics associated with the run.
        inputs (list[ObjectRef]): Input objects for the run.
        outputs (list[ObjectRef]): Output objects for the run.
        objects (dict[str, Object]): Additional objects associated with the run.
        object_schemas (AnyDict): Schemas for the objects.
        schema_ (AnyDict): The schema of the run.
    """

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
    """Represents a task within the system.

    Attributes:
        name (str): The name of the task.
        span_id (str): The span ID associated with the task.
        trace_id (str): The trace ID associated with the task.
        parent_span_id (str | None): The parent span ID, if any.
        parent_task_span_id (str | None): The parent task span ID, if any.
        timestamp (datetime): The timestamp of the task.
        duration (int): The duration of the task in milliseconds.
        status (SpanStatus): The status of the task.
        exception (SpanException | None): The exception associated with the task, if any.
        tags (set[str]): A set of tags associated with the task.
        params (AnyDict): Parameters associated with the task.
        metrics (dict[str, list[Metric]]): Metrics associated with the task.
        inputs (list[ObjectRef] | list[V0Object]): Input objects for the task.
        outputs (list[ObjectRef] | list[V0Object]): Output objects for the task.
        schema_ (AnyDict): The schema of the task.
        attributes (AnyDict): Additional attributes for the task.
        resource_attributes (AnyDict): Resource-specific attributes for the task.
        events (list[SpanEvent]): A list of events associated with the task.
        links (list[SpanLink]): A list of links to other spans.
    """

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
    """Represents a project within the system.

    Attributes:
        id (UUID): The unique identifier for the project.
        key (str): The key associated with the project.
        name (str): The name of the project.
        description (str | None): A description of the project.
        created_at (datetime): The timestamp when the project was created.
        updated_at (datetime): The timestamp when the project was last updated.
        run_count (int): The number of runs associated with the project.
        last_run (Run | None): The last run associated with the project, if any.
    """

    id: UUID
    key: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run: Run | None


class TaskTree(BaseModel):
    """Represents a tree structure of tasks.

    Attributes:
        task (Task): The root task of the tree.
        children (list[TaskTree]): The child tasks of the root task.
    """

    task: Task
    children: list["TaskTree"] = []


class SpanTree(BaseModel):
    """Represents a tree structure of spans.

    Attributes:
        span (Task | TraceSpan): The root span of the tree.
        children (list[SpanTree]): The child spans of the root span.
    """

    span: Task | TraceSpan
    children: list["SpanTree"] = []


class UserDataCredentials(BaseModel):
    """Represents user data credentials for accessing resources.

    Attributes:
        access_key_id (str): The access key ID.
        secret_access_key (str): The secret access key.
        session_token (str): The session token.
        expiration (datetime): The expiration time of the credentials.
        region (str): The region associated with the credentials.
        bucket (str): The bucket associated with the credentials.
        prefix (str): The prefix associated with the credentials.
        endpoint (str | None): The endpoint associated with the credentials, if any.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: datetime
    region: str
    bucket: str
    prefix: str
    endpoint: str | None
