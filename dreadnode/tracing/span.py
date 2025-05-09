"""
This module provides classes and utilities for tracing spans in the Dreadnode SDK.

It includes the `Span` base class and its specialized subclasses such as `RunSpan`,
`RunUpdateSpan`, and `TaskSpan`. These classes are used to manage and log tracing
information, metrics, parameters, and artifacts during the execution of tasks and runs.
"""

import logging
import re
import types
import typing as t
from contextvars import ContextVar, Token
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import typing_extensions as te
from fsspec import AbstractFileSystem  # type: ignore [import-untyped]
from logfire._internal.json_encoder import logfire_json_dumps as json_dumps
from logfire._internal.json_schema import (
    JsonSchemaProperties,
    attributes_json_schema,
    create_json_schema,
)
from logfire._internal.tracer import OPEN_SPANS
from logfire._internal.utils import uniquify_sequence
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Tracer
from opentelemetry.util import types as otel_types
from ulid import ULID

from dreadnode.artifact.merger import ArtifactMerger
from dreadnode.artifact.storage import ArtifactStorage
from dreadnode.artifact.tree_builder import ArtifactTreeBuilder, DirectoryNode
from dreadnode.constants import MAX_INLINE_OBJECT_BYTES
from dreadnode.metric import Metric, MetricAggMode, MetricDict
from dreadnode.object import Object, ObjectRef, ObjectUri, ObjectVal
from dreadnode.serialization import Serialized, serialize
from dreadnode.types import UNSET, AnyDict, JsonDict, JsonValue, Unset
from dreadnode.version import VERSION

from .constants import (
    EVENT_ATTRIBUTE_LINK_HASH,
    EVENT_ATTRIBUTE_OBJECT_HASH,
    EVENT_ATTRIBUTE_OBJECT_LABEL,
    EVENT_ATTRIBUTE_ORIGIN_SPAN_ID,
    EVENT_NAME_OBJECT,
    EVENT_NAME_OBJECT_INPUT,
    EVENT_NAME_OBJECT_LINK,
    EVENT_NAME_OBJECT_METRIC,
    EVENT_NAME_OBJECT_OUTPUT,
    METRIC_ATTRIBUTE_SOURCE_HASH,
    SPAN_ATTRIBUTE_ARTIFACTS,
    SPAN_ATTRIBUTE_INPUTS,
    SPAN_ATTRIBUTE_LABEL,
    SPAN_ATTRIBUTE_LARGE_ATTRIBUTES,
    SPAN_ATTRIBUTE_METRICS,
    SPAN_ATTRIBUTE_OBJECT_SCHEMAS,
    SPAN_ATTRIBUTE_OBJECTS,
    SPAN_ATTRIBUTE_OUTPUTS,
    SPAN_ATTRIBUTE_PARAMS,
    SPAN_ATTRIBUTE_PARENT_TASK_ID,
    SPAN_ATTRIBUTE_PROJECT,
    SPAN_ATTRIBUTE_RUN_ID,
    SPAN_ATTRIBUTE_SCHEMA,
    SPAN_ATTRIBUTE_TAGS_,
    SPAN_ATTRIBUTE_TYPE,
    SPAN_ATTRIBUTE_VERSION,
    SpanType,
)

logger = logging.getLogger(__name__)

R = t.TypeVar("R")


current_task_span: ContextVar["TaskSpan[t.Any] | None"] = ContextVar(
    "current_task_span",
    default=None,
)
current_run_span: ContextVar["RunSpan | None"] = ContextVar(
    "current_run_span",
    default=None,
)


class Span(ReadableSpan):
    """
    Base class for tracing spans.

    Attributes:
        name (str): The name of the span.
        attributes (AnyDict): Attributes associated with the span.
        tracer (Tracer): The tracer instance used for the span.
        label (str | None): Optional label for the span.
        type (SpanType): The type of the span (e.g., "span").
        tags (t.Sequence[str] | None): Optional tags for the span.
    """

    def __init__(
        self,
        name: str,
        attributes: AnyDict,
        tracer: Tracer,
        *,
        label: str | None = None,
        type: SpanType = "span",
        tags: t.Sequence[str] | None = None,
    ) -> None:
        """
        Initializes a Span instance.

        Args:
            name (str): The name of the span.
            attributes (AnyDict): Attributes associated with the span.
            tracer (Tracer): The tracer instance used for the span.
            label (str | None, optional): Optional label for the span. Defaults to None.
            type (SpanType, optional): The type of the span. Defaults to "span".
            tags (t.Sequence[str] | None, optional): Optional tags for the span. Defaults to None.
        """
        self._label = label or ""
        self._span_name = name
        self._pre_attributes = {
            SPAN_ATTRIBUTE_VERSION: VERSION,
            SPAN_ATTRIBUTE_TYPE: type,
            SPAN_ATTRIBUTE_LABEL: self._label,
            SPAN_ATTRIBUTE_TAGS_: uniquify_sequence(tags or ()),
            **attributes,
        }
        self._tracer = tracer

        self._schema: JsonSchemaProperties = JsonSchemaProperties({})
        self._token: object | None = None  # trace sdk context
        self._span: trace_api.Span | None = None

    if not t.TYPE_CHECKING:

        def __getattr__(self, name: str) -> t.Any:
            return getattr(self._span, name)

    def __enter__(self) -> te.Self:
        """
        Enters the span context.

        Returns:
            te.Self: The span instance.
        """
        if self._span is None:
            self._span = self._tracer.start_span(
                name=self._span_name,
                attributes=prepare_otlp_attributes(self._pre_attributes),
            )

        self._span.__enter__()

        OPEN_SPANS.add(self._span)  # type: ignore [arg-type]

        if self._token is None:
            self._token = context_api.attach(trace_api.set_span_in_context(self._span))

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """
        Exits the span context.

        Args:
            exc_type (type[BaseException] | None): The exception type, if any.
            exc_value (BaseException | None): The exception value, if any.
            traceback (types.TracebackType | None): The traceback, if any.
        """
        if self._token is None or self._span is None:
            return

        context_api.detach(self._token)  # type: ignore [arg-type]
        self._token = None

        if not self._span.is_recording():
            return

        self._span.set_attribute(
            SPAN_ATTRIBUTE_SCHEMA,
            attributes_json_schema(self._schema) if self._schema else r"{}",
        )
        self._span.__exit__(exc_type, exc_value, traceback)

        OPEN_SPANS.discard(self._span)  # type: ignore [arg-type]

    @property
    def span_id(self) -> str:
        """
        Returns the span ID.

        Returns:
            str: The span ID.

        Raises:
            ValueError: If the span is not active.
        """
        if self._span is None:
            raise ValueError("Span is not active")
        return trace_api.format_span_id(self._span.get_span_context().span_id)

    @property
    def trace_id(self) -> str:
        """
        Returns the trace ID.

        Returns:
            str: The trace ID.

        Raises:
            ValueError: If the span is not active.
        """
        if self._span is None:
            raise ValueError("Span is not active")
        return trace_api.format_trace_id(self._span.get_span_context().trace_id)

    @property
    def is_recording(self) -> bool:
        """
        Checks if the span is recording.

        Returns:
            bool: True if the span is recording, False otherwise.
        """
        if self._span is None:
            return False
        return self._span.is_recording()

    @property
    def tags(self) -> tuple[str, ...]:
        """
        Gets the tags associated with the span.

        Returns:
            tuple[str, ...]: The tags.
        """
        return tuple(self.get_attribute(SPAN_ATTRIBUTE_TAGS_, ()))

    @tags.setter
    def tags(self, new_tags: t.Sequence[str]) -> None:
        """
        Sets the tags for the span.

        Args:
            new_tags (t.Sequence[str]): The new tags.
        """
        self.set_attribute(SPAN_ATTRIBUTE_TAGS_, uniquify_sequence(new_tags))

    def set_attribute(
        self,
        key: str,
        value: t.Any,
        *,
        schema: bool = True,
        raw: bool = False,
    ) -> None:
        """
        Sets an attribute for the span.

        Args:
            key (str): The attribute key.
            value (t.Any): The attribute value.
            schema (bool, optional): Whether to include the attribute in the schema. Defaults to True.
            raw (bool, optional): Whether to store the raw value. Defaults to False.
        """
        self._added_attributes = True
        if schema and raw is False:
            self._schema[key] = create_json_schema(value, set())
        otel_value = self._pre_attributes[key] = value if raw else prepare_otlp_attribute(value)
        if self._span is not None:
            self._span.set_attribute(key, otel_value)
        self._pre_attributes[key] = otel_value

    def set_attributes(self, attributes: AnyDict) -> None:
        """
        Sets multiple attributes for the span.

        Args:
            attributes (AnyDict): A dictionary of attributes to set.
        """
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def get_attributes(self) -> AnyDict:
        """
        Gets all attributes of the span.

        Returns:
            AnyDict: A dictionary of attributes.
        """
        if self._span is not None:
            return getattr(self._span, "attributes", {})
        return self._pre_attributes

    def get_attribute(self, key: str, default: t.Any) -> t.Any:
        """
        Gets a specific attribute of the span.

        Args:
            key (str): The attribute key.
            default (t.Any): The default value if the attribute is not found.

        Returns:
            t.Any: The attribute value or the default value.
        """
        return self.get_attributes().get(key, default)

    def log_event(
        self,
        name: str,
        attributes: AnyDict | None = None,
    ) -> None:
        """
        Logs an event to the span.

        Args:
            name (str): The name of the event.
            attributes (AnyDict | None, optional): Attributes associated with the event. Defaults to None.
        """
        if self._span is not None:
            self._span.add_event(
                name,
                attributes=prepare_otlp_attributes(attributes or {}),
            )


class RunUpdateSpan(Span):
    """
    A specialized span for updating run information.

    Attributes:
        run_id (str): The ID of the run.
        tracer (Tracer): The tracer instance.
        project (str): The project name.
        metrics (MetricDict | None): Optional metrics for the run.
        params (JsonDict | None): Optional parameters for the run.
        inputs (JsonDict | None): Optional inputs for the run.
    """

    def __init__(
        self,
        run_id: str,
        tracer: Tracer,
        project: str,
        *,
        metrics: MetricDict | None = None,
        params: JsonDict | None = None,
        inputs: JsonDict | None = None,
    ) -> None:
        """
        Initializes a RunUpdateSpan instance.

        Args:
            run_id (str): The ID of the run.
            tracer (Tracer): The tracer instance.
            project (str): The project name.
            metrics (MetricDict | None, optional): Optional metrics for the run. Defaults to None.
            params (JsonDict | None, optional): Optional parameters for the run. Defaults to None.
            inputs (JsonDict | None, optional): Optional inputs for the run. Defaults to None.
        """
        attributes: AnyDict = {
            SPAN_ATTRIBUTE_RUN_ID: run_id,
            SPAN_ATTRIBUTE_PROJECT: project,
        }

        if metrics:
            attributes[SPAN_ATTRIBUTE_METRICS] = metrics
        if params:
            attributes[SPAN_ATTRIBUTE_PARAMS] = params
        if inputs:
            attributes[SPAN_ATTRIBUTE_INPUTS] = inputs

        super().__init__(f"run.{run_id}.update", attributes, tracer, type="run_update")


class RunSpan(Span):
    """
    A specialized span for managing and logging tracing information for a run.

    Attributes:
        project (str): The project name associated with the run.
        params (AnyDict): Parameters associated with the run.
        metrics (MetricDict): Metrics associated with the run.
        inputs (list[ObjectRef]): Input objects for the run.
        outputs (list[ObjectRef]): Output objects for the run.
        artifacts (list[DirectoryNode]): Artifacts associated with the run.
        run_id (str): The unique identifier for the run.
    """

    def __init__(
        self,
        name: str,
        project: str,
        attributes: AnyDict,
        tracer: Tracer,
        file_system: AbstractFileSystem,
        prefix_path: str,
        params: AnyDict | None = None,
        metrics: MetricDict | None = None,
        run_id: str | None = None,
        tags: t.Sequence[str] | None = None,
    ) -> None:
        """
        Initializes a RunSpan instance.

        Args:
            name (str): The name of the span.
            project (str): The project name associated with the run.
            attributes (AnyDict): Attributes associated with the span.
            tracer (Tracer): The tracer instance used for the span.
            file_system (AbstractFileSystem): The file system for artifact storage.
            prefix_path (str): The prefix path for artifact storage.
            params (AnyDict | None, optional): Parameters for the run. Defaults to None.
            metrics (MetricDict | None, optional): Metrics for the run. Defaults to None.
            run_id (str | None, optional): The unique identifier for the run. Defaults to None.
            tags (t.Sequence[str] | None, optional): Tags for the span. Defaults to None.
        """
        self._params = params or {}
        self._metrics = metrics or {}
        self._objects: dict[str, Object] = {}
        self._object_schemas: dict[str, JsonDict] = {}
        self._inputs: list[ObjectRef] = []
        self._outputs: list[ObjectRef] = []
        self._artifact_storage = ArtifactStorage(file_system=file_system)
        self._artifacts: list[DirectoryNode] = []
        self._artifact_merger = ArtifactMerger()
        self._artifact_tree_builder = ArtifactTreeBuilder(
            storage=self._artifact_storage,
            prefix_path=prefix_path,
        )
        self.project = project

        self._last_pushed_params = deepcopy(self._params)
        self._last_pushed_metrics = deepcopy(self._metrics)

        self._context_token: Token[RunSpan | None] | None = None  # contextvars context
        self._file_system = file_system
        self._prefix_path = prefix_path

        attributes = {
            SPAN_ATTRIBUTE_RUN_ID: str(run_id or ULID()),
            SPAN_ATTRIBUTE_PROJECT: project,
            SPAN_ATTRIBUTE_PARAMS: self._params,
            SPAN_ATTRIBUTE_METRICS: self._metrics,
            **attributes,
        }
        super().__init__(name, attributes, tracer, type="run", tags=tags)

    def __enter__(self) -> te.Self:
        if current_run_span.get() is not None:
            raise RuntimeError("You cannot start a run span within another run")

        self._context_token = current_run_span.set(self)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.set_attribute(SPAN_ATTRIBUTE_PARAMS, self._params)
        self.set_attribute(SPAN_ATTRIBUTE_INPUTS, self._inputs, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OUTPUTS, self._outputs, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_METRICS, self._metrics, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OBJECTS, self._objects, schema=False)
        self.set_attribute(
            SPAN_ATTRIBUTE_OBJECT_SCHEMAS,
            self._object_schemas,
            schema=False,
        )
        self.set_attribute(SPAN_ATTRIBUTE_ARTIFACTS, self._artifacts, schema=False)

        # Mark our objects attribute as large so it's stored separately
        self.set_attribute(
            SPAN_ATTRIBUTE_LARGE_ATTRIBUTES,
            [SPAN_ATTRIBUTE_OBJECTS, SPAN_ATTRIBUTE_OBJECT_SCHEMAS],
            raw=True,
        )

        super().__exit__(exc_type, exc_value, traceback)
        if self._context_token is not None:
            current_run_span.reset(self._context_token)

    def push_update(self) -> None:
        """
        Pushes updates for the run's parameters and metrics to the tracing system.

        If there are no changes to the parameters or metrics since the last push,
        this method does nothing.
        """
        if self._span is None:
            return

        metrics: MetricDict | None = None
        if self._last_pushed_metrics != self._metrics:
            metrics = self._metrics
            self._last_pushed_metrics = deepcopy(self._metrics)

        params: JsonDict | None = None
        if self._last_pushed_params != self._params:
            params = self._params
            self._last_pushed_params = deepcopy(self._params)

        if metrics is None and params is None:
            return

        with RunUpdateSpan(
            run_id=self.run_id,
            project=self.project,
            tracer=self._tracer,
            params=params,
            metrics=metrics,
        ):
            pass

    @property
    def run_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_RUN_ID, ""))

    def log_object(
        self,
        value: t.Any,
        *,
        label: str | None = None,
        event_name: str = EVENT_NAME_OBJECT,
        **attributes: JsonValue,
    ) -> str:
        """
        Logs an object to the span.

        Args:
            value (t.Any): The object to log.
            label (str | None, optional): An optional label for the object. Defaults to None.
            event_name (str, optional): The name of the event. Defaults to EVENT_NAME_OBJECT.
            **attributes (JsonValue): Additional attributes for the event.

        Returns:
            str: The hash of the logged object.
        """
        serialized = serialize(value)
        data_hash = serialized.data_hash
        schema_hash = serialized.schema_hash

        # Store object if we haven't already
        if data_hash not in self._objects:
            self._objects[data_hash] = self._create_object(serialized)

        object_ = self._objects[data_hash]

        # Store schema if new
        if schema_hash not in self._object_schemas:
            self._object_schemas[schema_hash] = serialized.schema

        # Build event attributes
        event_attributes = {
            **attributes,
            EVENT_ATTRIBUTE_OBJECT_HASH: object_.hash,
            EVENT_ATTRIBUTE_ORIGIN_SPAN_ID: trace_api.format_span_id(
                trace_api.get_current_span().get_span_context().span_id,
            ),
        }
        if label is not None:
            event_attributes[EVENT_ATTRIBUTE_OBJECT_LABEL] = label

        self.log_event(name=event_name, attributes=event_attributes)
        return object_.hash

    def _store_file_by_hash(self, data: bytes, full_path: str) -> str:
        """
        Writes data to the given full_path in the object store if it doesn't already exist.

        Args:
            data: Content to write.
            full_path: The path in the object store (e.g., S3 key or local path).

        Returns:
            The unstrip_protocol version of the full path (for object store URI).
        """
        if not self._file_system.exists(full_path):
            logger.debug("Storing new object at: %s", full_path)
            with self._file_system.open(full_path, "wb") as f:
                f.write(data)

        return str(self._file_system.unstrip_protocol(full_path))

    def _create_object(self, serialized: Serialized) -> Object:
        """
        Creates an ObjectVal or ObjectUri depending on the size of the serialized data.

        Args:
            serialized (Serialized): The serialized object data.

        Returns:
            Object: The created object (either ObjectVal or ObjectUri).
        """
        data = serialized.data
        data_bytes = serialized.data_bytes
        data_len = serialized.data_len
        data_hash = serialized.data_hash
        schema_hash = serialized.schema_hash

        if data is None or data_bytes is None or data_len <= MAX_INLINE_OBJECT_BYTES:
            return ObjectVal(
                hash=data_hash,
                value=data,
                schema_hash=schema_hash,
            )

        # Offload to file system (e.g., S3)
        full_path = f"{self._prefix_path.rstrip('/')}/{data_hash}"
        object_uri = self._store_file_by_hash(data_bytes, full_path)

        return ObjectUri(
            hash=data_hash,
            uri=object_uri,
            schema_hash=schema_hash,
            size=data_len,
        )

    def get_object(self, hash_: str) -> t.Any:
        """
        Retrieves an object by its hash.

        Args:
            hash_ (str): The hash of the object to retrieve.

        Returns:
            t.Any: The retrieved object.
        """
        return self._objects[hash_]

    def link_objects(
        self,
        object_hash: str,
        link_hash: str,
        **attributes: JsonValue,
    ) -> None:
        """
        Logs a link between two objects in the span.

        Args:
            object_hash (str): The hash of the source object.
            link_hash (str): The hash of the linked object.
            **attributes (JsonValue): Additional attributes for the link event.
        """
        self.log_event(
            name=EVENT_NAME_OBJECT_LINK,
            attributes={
                **attributes,
                EVENT_ATTRIBUTE_OBJECT_HASH: object_hash,
                EVENT_ATTRIBUTE_LINK_HASH: link_hash,
                EVENT_ATTRIBUTE_ORIGIN_SPAN_ID: (
                    trace_api.format_span_id(
                        trace_api.get_current_span().get_span_context().span_id,
                    )
                ),
            },
        )

    @property
    def params(self) -> AnyDict:
        return self._params

    def log_param(self, key: str, value: t.Any) -> None:
        self.log_params(**{key: value})

    def log_params(self, **params: t.Any) -> None:
        for key, value in params.items():
            self._params[key] = value

        # Always push updates for run params
        self.push_update()

    @property
    def inputs(self) -> AnyDict:
        return {ref.name: self.get_object(ref.hash) for ref in self._inputs}

    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        **attributes: JsonValue,
    ) -> None:
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_INPUT,
            **attributes,
        )
        self._inputs.append(ObjectRef(name, label=label, hash=hash_))

    def log_artifact(
        self,
        local_uri: str | Path,
    ) -> None:
        """
        Logs a local file or directory as an artifact to the object store.
        Preserves directory structure and uses content hashing for deduplication.

        Args:
            local_uri: Path to the local file or directory

        Returns:
            DirectoryNode representing the artifact's tree structure

        Raises:
            FileNotFoundError: If the path doesn't exist
        """

        artifact_tree = self._artifact_tree_builder.process_artifact(local_uri)

        self._artifact_merger.add_tree(artifact_tree)

        self._artifacts = self._artifact_merger.get_merged_trees()

    @property
    def metrics(self) -> MetricDict:
        return self._metrics

    @t.overload
    def log_metric(
        self,
        key: str,
        value: float | bool,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> None: ...

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
    ) -> None: ...

    def log_metric(
        self,
        key: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> None:
        """
        Logs a metric to the span.

        Args:
            key (str): The key for the metric.
            value (float | bool | Metric): The value of the metric.
            step (int, optional): The step associated with the metric. Defaults to 0.
            origin (t.Any | None, optional): The origin of the metric. Defaults to None.
            timestamp (datetime | None, optional): The timestamp of the metric. Defaults to None.
            mode (MetricAggMode | None, optional): The aggregation mode for the metric. Defaults to None.
            attributes (JsonDict | None, optional): Additional attributes for the metric. Defaults to None.
        """
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value), step, timestamp or datetime.now(timezone.utc), attributes or {}
            )
        )

        if origin is not None:
            origin_hash = self.log_object(
                origin,
                label=key,
                event_name=EVENT_NAME_OBJECT_METRIC,
            )
            metric.attributes[METRIC_ATTRIBUTE_SOURCE_HASH] = origin_hash

        metrics = self._metrics.setdefault(key, [])
        if mode is not None:
            metric = metric.apply_mode(mode, metrics)
        metrics.append(metric)

    @property
    def outputs(self) -> AnyDict:
        return {ref.name: self.get_object(ref.hash) for ref in self._outputs}

    def log_output(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        **attributes: JsonValue,
    ) -> None:
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_OUTPUT,
            **attributes,
        )
        self._outputs.append(ObjectRef(name, label=label, hash=hash_))


class TaskSpan(Span, t.Generic[R]):
    """
    A specialized span for managing and logging tracing information for a task.

    Attributes:
        name (str): The name of the task span.
        attributes (AnyDict): Attributes associated with the task span.
        run_id (str): The unique identifier for the run associated with the task.
        tracer (Tracer): The tracer instance used for the task span.
        label (str | None): An optional label for the task span.
        params (AnyDict): Parameters associated with the task.
        metrics (MetricDict): Metrics associated with the task.
        tags (t.Sequence[str] | None): Optional tags for the task span.
        inputs (list[ObjectRef]): Input objects for the task.
        outputs (list[ObjectRef]): Output objects for the task.
        output (R | Unset): The Python output of the task.
    """

    def __init__(
        self,
        name: str,
        attributes: AnyDict,
        run_id: str,
        tracer: Tracer,
        *,
        label: str | None = None,
        params: AnyDict | None = None,
        metrics: MetricDict | None = None,
        tags: t.Sequence[str] | None = None,
    ) -> None:
        """
        Initializes a TaskSpan instance.

        Args:
            name (str): The name of the task span.
            attributes (AnyDict): Attributes associated with the task span.
            run_id (str): The unique identifier for the run associated with the task.
            tracer (Tracer): The tracer instance used for the task span.
            label (str | None, optional): An optional label for the task span. Defaults to None.
            params (AnyDict | None, optional): Parameters for the task. Defaults to None.
            metrics (MetricDict | None, optional): Metrics for the task. Defaults to None.
            tags (t.Sequence[str] | None, optional): Tags for the task span. Defaults to None.
        """
        self._params = params or {}
        self._metrics = metrics or {}
        self._inputs: list[ObjectRef] = []
        self._outputs: list[ObjectRef] = []

        self._output: R | Unset = UNSET  # For the python output

        self._context_token: Token[TaskSpan[t.Any] | None] | None = None  # contextvars context

        attributes = {
            SPAN_ATTRIBUTE_RUN_ID: str(run_id),
            SPAN_ATTRIBUTE_PARAMS: self._params,
            SPAN_ATTRIBUTE_INPUTS: self._inputs,
            SPAN_ATTRIBUTE_METRICS: self._metrics,
            SPAN_ATTRIBUTE_OUTPUTS: self._outputs,
            **attributes,
        }
        super().__init__(name, attributes, tracer, type="task", label=label, tags=tags)

    def __enter__(self) -> te.Self:
        """
        Enters the task span context.

        Returns:
            te.Self: The task span instance.

        Raises:
            RuntimeError: If a task span is started without an active run.
        """
        self._parent_task = current_task_span.get()
        if self._parent_task is not None:
            self.set_attribute(SPAN_ATTRIBUTE_PARENT_TASK_ID, self._parent_task.span_id)

        self._run = current_run_span.get()
        if self._run is None:
            raise RuntimeError("You cannot start a task span without a run")

        self._context_token = current_task_span.set(self)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """
        Exits the task span context.

        Args:
            exc_type (type[BaseException] | None): The exception type, if any.
            exc_value (BaseException | None): The exception value, if any.
            traceback (types.TracebackType | None): The traceback, if any.
        """
        self.set_attribute(SPAN_ATTRIBUTE_PARAMS, self._params)
        self.set_attribute(SPAN_ATTRIBUTE_INPUTS, self._inputs, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_METRICS, self._metrics, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OUTPUTS, self._outputs, schema=False)
        super().__exit__(exc_type, exc_value, traceback)
        if self._context_token is not None:
            current_task_span.reset(self._context_token)

    @property
    def run_id(self) -> str:
        """
        Gets the run ID associated with the task span.

        Returns:
            str: The run ID.
        """
        return str(self.get_attribute(SPAN_ATTRIBUTE_RUN_ID, ""))

    @property
    def parent_task_id(self) -> str:
        """
        Gets the parent task ID, if any.

        Returns:
            str: The parent task ID.
        """
        return str(self.get_attribute(SPAN_ATTRIBUTE_PARENT_TASK_ID, ""))

    @property
    def run(self) -> RunSpan:
        """
        Gets the active run span associated with the task.

        Returns:
            RunSpan: The active run span.

        Raises:
            ValueError: If the task span is not in an active run.
        """
        if self._run is None:
            raise ValueError("Task span is not in an active run")
        return self._run

    @property
    def outputs(self) -> AnyDict:
        """
        Gets the output objects logged to the task span.

        Returns:
            AnyDict: A dictionary of output objects.
        """
        return {ref.name: self.run.get_object(ref.hash) for ref in self._outputs}

    @property
    def output(self) -> R:
        """
        Gets the Python output of the task.

        Returns:
            R: The Python output.

        Raises:
            TypeError: If the task output is not set.
        """
        if isinstance(self._output, Unset):
            raise TypeError("Task output is not set")
        return self._output

    @output.setter
    def output(self, value: R) -> None:
        self._output = value

    def log_output(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        **attributes: JsonValue,
    ) -> str:
        """
        Logs an output object to the task span.

        Args:
            name (str): The name of the output.
            value (t.Any): The value of the output.
            label (str | None, optional): An optional label for the output. Defaults to None.
            **attributes (JsonValue): Additional attributes for the output.

        Returns:
            str: The hash of the logged output.
        """
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_OUTPUT,
            **attributes,
        )
        self._outputs.append(ObjectRef(name, label=label, hash=hash_))
        return hash_

    @property
    def params(self) -> AnyDict:
        """
        Gets the parameters associated with the task.

        Returns:
            AnyDict: A dictionary of task parameters.
        """
        return self._params

    def log_param(self, key: str, value: t.Any) -> None:
        """
        Logs a single parameter to the task span.

        Args:
            key (str): The parameter key.
            value (t.Any): The parameter value.
        """
        self.log_params(**{key: value})

    def log_params(self, **params: t.Any) -> None:
        """
        Logs multiple parameters to the task span.

        Args:
            **params (t.Any): The parameters to log.
        """
        self._params.update(params)

    @property
    def inputs(self) -> AnyDict:
        """
        Gets the input objects logged to the task span.

        Returns:
            AnyDict: A dictionary of input objects.
        """
        return {ref.name: self.run.get_object(ref.hash) for ref in self._inputs}

    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        **attributes: JsonValue,
    ) -> str:
        """
        Logs an input object to the task span.

        Args:
            name (str): The name of the input.
            value (t.Any): The value of the input.
            label (str | None, optional): An optional label for the input. Defaults to None.
            **attributes (JsonValue): Additional attributes for the input.

        Returns:
            str: The hash of the logged input.
        """
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_INPUT,
            **attributes,
        )
        self._inputs.append(ObjectRef(name, label=label, hash=hash_))
        return hash_

    @property
    def metrics(self) -> dict[str, list[Metric]]:
        """
        Gets the metrics logged to the task span.

        Returns:
            dict[str, list[Metric]]: A dictionary of metrics.
        """
        return self._metrics

    @t.overload
    def log_metric(
        self,
        key: str,
        value: float | bool,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> None: ...

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
    ) -> None: ...

    def log_metric(
        self,
        key: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> None:
        """
        Logs a metric to the task span.

        Args:
            key (str): The key for the metric.
            value (float | bool | Metric): The value of the metric.
            step (int, optional): The step associated with the metric. Defaults to 0.
            origin (t.Any | None, optional): The origin of the metric. Defaults to None.
            timestamp (datetime | None, optional): The timestamp of the metric. Defaults to None.
            mode (MetricAggMode | None, optional): The aggregation mode for the metric. Defaults to None.
            attributes (JsonDict | None, optional): Additional attributes for the metric. Defaults to None.
        """
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value), step, timestamp or datetime.now(timezone.utc), attributes or {}
            )
        )

        if origin is not None:
            origin_hash = self.run.log_object(
                origin,
                label=key,
                event_name=EVENT_NAME_OBJECT_METRIC,
            )
            metric.attributes[METRIC_ATTRIBUTE_SOURCE_HASH] = origin_hash

        metrics = self._metrics.setdefault(key, [])
        if mode is not None:
            metric = metric.apply_mode(mode, metrics)
        metrics.append(metric)

        # For every metric we log, also log it to the run
        # with our `label` as a prefix.
        #
        # Don't include `source` and `mode` as we handled it here.
        if (run := current_run_span.get()) is not None:
            run.log_metric(f"{self._label}.{key}", metric)

    def get_average_metric_value(self, key: str | None = None) -> float:
        """
        Calculates the average value of a metric or all metrics.

        Args:
            key (str | None, optional): The key of the metric to calculate the average for.
                If None, calculates the average for all metrics. Defaults to None.

        Returns:
            float: The average value of the metric(s).
        """
        metrics = (
            self._metrics.get(key, [])
            if key is not None
            else [m for ms in self._metrics.values() for m in ms]
        )
        return sum(metric.value for metric in metrics) / len(
            metrics,
        )


def prepare_otlp_attributes(
    attributes: AnyDict,
) -> dict[str, otel_types.AttributeValue]:
    """
    Prepares attributes for OTLP export.
    Converts attributes to a format suitable for OTLP export.

    Args:
        attributes (AnyDict): A dictionary of attributes to prepare.
    Returns:
        dict[str, otel_types.AttributeValue]: A dictionary of prepared attributes.
    """
    return {key: prepare_otlp_attribute(value) for key, value in attributes.items()}


def prepare_otlp_attribute(value: t.Any) -> otel_types.AttributeValue:
    """
    Prepares a single attribute for OTLP export.
    Converts the attribute to a format suitable for OTLP export.

    Args:
        value (t.Any): The attribute value to prepare.
    Returns:
        otel_types.AttributeValue: The prepared attribute value.
    """
    if isinstance(value, str | int | bool | float):
        return value
    return json_dumps(value)
