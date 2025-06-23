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
"""Status of a span in the trace"""

ExportFormat = t.Literal["csv", "json", "jsonl", "parquet"]
"""Available export formats for traces and runs"""
StatusFilter = t.Literal["all", "completed", "failed"]
"""Filter for trace and run statuses"""
TimeAxisType = t.Literal["wall", "relative", "step"]
"""Type of time axis for traces and runs"""
TimeAggregationType = t.Literal["max", "min", "sum", "count"]
"""How to aggregate time in traces and runs"""
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
"""How to aggregate metrics in traces and runs"""


class SpanException(BaseModel):
    """Exception details for a span in a trace."""

    type: str
    message: str
    stacktrace: str


class SpanEvent(BaseModel):
    """OTEL event for a span in a trace."""

    timestamp: datetime
    name: str
    attributes: AnyDict


class SpanLink(BaseModel):
    """OTEL link for a span in a trace."""

    trace_id: str
    span_id: str
    attributes: AnyDict


class TraceSpan(BaseModel):
    """Span in a trace, representing a single operation or task."""

    timestamp: datetime
    """Timestamp when the span started."""
    duration: int
    """Duration of the span in milliseconds."""
    trace_id: str = Field(repr=False)
    """Unique identifier for the trace this span belongs to."""
    span_id: str
    """Unique identifier for the span."""
    parent_span_id: str | None = Field(repr=False)
    """ID of the parent span, if any."""
    service_name: str | None = Field(repr=False)
    """Name of the service that generated this span."""
    status: SpanStatus
    """Status of the span, e.g., 'completed', 'failed'."""
    exception: SpanException | None
    """Exception details if the span failed."""
    name: str
    """Name of the operation or task represented by the span."""
    attributes: AnyDict = Field(repr=False)
    """Attributes associated with the span."""
    resource_attributes: AnyDict = Field(repr=False)
    """Resource attributes for the span, e.g., host, service version."""
    events: list[SpanEvent] = Field(repr=False)
    """Events associated with the span, e.g., logs, checkpoints."""
    links: list[SpanLink] = Field(repr=False)
    """Links to other spans or resources related to this span."""


class Metric(BaseModel):
    """Metric data for a span in a trace."""

    value: float | None
    """Value of the metric."""
    step: int
    """Step or iteration number for the metric."""
    timestamp: datetime
    """Timestamp when the metric was recorded."""
    attributes: AnyDict
    """Attributes associated with the metric, e.g., labels, tags."""


class ObjectRef(BaseModel):
    """Reference to an object in a run or task."""

    name: str
    """Name of the object."""
    label: str
    """Label for the object."""
    hash: str
    """Hash of the object, used for deduplication and content tracking."""


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
    """Represents a value object in a run or task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """Name of the object."""
    label: str
    """Label for the object."""
    hash: str = Field(repr=False)
    """Hash of the object, used for deduplication and content tracking."""
    schema_: AnyDict
    """Schema of the object, describing its structure."""
    schema_hash: str = Field(repr=False)
    """Hash of the schema, used for deduplication."""
    value: t.Any
    """The actual value of the object, can be any type."""

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            with contextlib.suppress(ValidationError):
                return TypeAdapter(t.Any).validate_json(value)

        return value


class ObjectUri(BaseModel):
    """Represents a URI object in a run or task - stored in a remote filesystem."""

    name: str
    """Name of the object."""
    label: str
    """Label for the object."""
    hash: str = Field(repr=False)
    """Hash of the object, used for deduplication and content tracking."""
    schema_: AnyDict
    """Schema of the object, describing its structure."""
    schema_hash: str = Field(repr=False)
    """Hash of the schema, used for deduplication."""
    uri: str
    """URI where the object is stored (e.g. s3://...)."""
    size: int
    """Size of the object in bytes."""

    _value: t.Any = PrivateAttr(default=None)

    @cached_property
    def value(self) -> t.Any:
        """
        The actual value of the object, fetched from the URI if not already cached.
        """
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
"""Represents an object (input/output) in a run or task."""


class ArtifactFile(BaseModel):
    """Represents a file entry for artifacts."""

    hash: str
    """Hash of the file, used for deduplication."""
    uri: str
    """URI where the file is stored (e.g. s3://...)."""
    size_bytes: int
    """Size of the file in bytes."""
    final_real_path: str
    """Real path of the original file."""


class ArtifactDir(BaseModel):
    """Represents a directory entry for artifacts."""

    dir_path: str
    """Path to the directory."""
    hash: str
    """Hash of the directory, used for deduplication."""
    children: list[t.Union["ArtifactDir", ArtifactFile]]
    """List of child artifacts, which can be files or subdirectories."""


class RunSummary(BaseModel):
    """Summary of a run, containing metadata and basic information."""

    id: ULID | str
    """Unique identifier for the run."""
    name: str
    """Name of the run."""
    span_id: str = Field(repr=False)
    """Unique identifier for the run's span in the trace."""
    trace_id: str = Field(repr=False)
    """Unique identifier for the trace this run belongs to."""
    timestamp: datetime
    """Timestamp when the run started."""
    duration: int
    """Duration of the run in milliseconds."""
    status: SpanStatus
    """Status of the run, e.g., 'completed', 'failed'."""
    exception: SpanException | None
    """Exception details if the run failed."""
    tags: set[str]
    """Set of tags associated with the run."""
    params: AnyDict = Field(repr=False)
    """Parameters logged for the run with log_param()."""
    metrics: dict[str, list[Metric]] = Field(repr=False)
    """Metrics logged for the run with log_metric()."""


class RawRun(RunSummary):
    inputs: list[ObjectRef] = Field(repr=False)
    outputs: list[ObjectRef] = Field(repr=False)
    objects: dict[str, RawObject] = Field(repr=False)
    object_schemas: AnyDict = Field(repr=False)
    artifacts: list[ArtifactDir] = Field(repr=False)
    schema_: AnyDict = Field(alias="schema", repr=False)


class Run(RunSummary):
    """Detailed information about a run, including inputs, outputs, and artifacts."""

    inputs: dict[str, Object] = Field(repr=False)
    """Inputs logged for the run with log_input()."""
    outputs: dict[str, Object] = Field(repr=False)
    """Outputs logged for the run with log_output()."""
    artifacts: list[ArtifactDir] = Field(repr=False)
    """Artifacts associated with the run, including files and directories."""
    schema_: AnyDict = Field(alias="schema", repr=False)


class _Task(BaseModel):
    name: str
    """Name of the task."""
    span_id: str
    """Unique identifier for the task's span in the trace."""
    trace_id: str = Field(repr=False)
    """Unique identifier for the trace this task belongs to."""
    parent_span_id: str | None = Field(repr=False)
    """ID of the parent span, if any."""
    parent_task_span_id: str | None = Field(repr=False)
    """ID of the parent task's span, if any."""
    timestamp: datetime
    """Timestamp when the task started."""
    duration: int
    """Duration of the task in milliseconds."""
    status: SpanStatus
    """Status of the task, e.g., 'completed', 'failed'."""
    exception: SpanException | None
    """Exception details if the task failed."""
    tags: set[str]
    """Set of tags associated with the task."""
    params: AnyDict = Field(repr=False)
    """Parameters logged for the task with log_param()."""
    metrics: dict[str, list[Metric]] = Field(repr=False)
    """Metrics logged for the task with log_metric()."""
    attributes: AnyDict = Field(repr=False)
    """Attributes associated with the task, e.g., labels, tags."""
    resource_attributes: AnyDict = Field(repr=False)
    """Resource attributes for the task, e.g., host, service version."""
    events: list[SpanEvent] = Field(repr=False)
    """OTEL Events associated with the task span."""
    links: list[SpanLink] = Field(repr=False)
    """OTEL Links associated with the task span."""

    schema_: AnyDict = Field(alias="schema", repr=False)


class RawTask(_Task):
    inputs: list[ObjectRef] | list[V0Object] = Field(repr=False)
    outputs: list[ObjectRef] | list[V0Object] = Field(repr=False)


class Task(_Task):
    """Detailed information about a task, including inputs and outputs."""

    inputs: dict[str, Object] = Field(repr=False)
    """Inputs logged for the task with log_input() or autologging."""
    outputs: dict[str, Object] = Field(repr=False)
    """Outputs logged for the task with log_output() or autologging."""


class Project(BaseModel):
    """Project metadata, containing information about the project."""

    id: UUID = Field(repr=False)
    """Unique identifier for the project."""
    key: str
    """Key for the project, used for authentication."""
    name: str
    """Name of the project."""
    description: str | None = Field(repr=False)
    """Description of the project."""
    created_at: datetime
    """Timestamp when the project was created."""
    updated_at: datetime
    """Timestamp when the project was last updated."""
    run_count: int
    """Number of runs associated with the project."""
    last_run: RawRun | None = Field(repr=False)
    """Last run associated with the project, if any."""


# Derived types


class TaskTree(BaseModel):
    """Tree structure representing tasks and their relationships in a trace."""

    task: Task
    """Task at this node."""
    children: list["TaskTree"] = []
    """Children of this task."""


class TraceTree(BaseModel):
    """Tree structure representing spans and their relationships in a trace."""

    span: Task | TraceSpan
    """Span at this node, can be a Task or a TraceSpan."""
    children: list["TraceTree"] = []
    """Children of this span, representing nested spans or tasks."""


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
