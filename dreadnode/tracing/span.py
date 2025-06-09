import hashlib
import logging
import time
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
from opentelemetry import propagate
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Tracer
from opentelemetry.util import types as otel_types
from ulid import ULID

from dreadnode.artifact.merger import ArtifactMerger
from dreadnode.artifact.storage import ArtifactStorage
from dreadnode.artifact.tree_builder import ArtifactTreeBuilder, DirectoryNode
from dreadnode.constants import MAX_INLINE_OBJECT_BYTES
from dreadnode.metric import Metric, MetricAggMode, MetricsDict
from dreadnode.object import Object, ObjectRef, ObjectUri, ObjectVal
from dreadnode.serialization import Serialized, serialize
from dreadnode.types import UNSET, AnyDict, JsonDict, JsonValue, Unset
from dreadnode.util import clean_str
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
        self._label = label or ""
        self._span_name = name

        tags = [tags] if isinstance(tags, str) else list(tags or [])
        tags = [clean_str(t) for t in tags]
        self.tags: tuple[str, ...] = uniquify_sequence(tags)

        self._pre_attributes = {
            SPAN_ATTRIBUTE_VERSION: VERSION,
            SPAN_ATTRIBUTE_TYPE: type,
            SPAN_ATTRIBUTE_LABEL: self._label,
            SPAN_ATTRIBUTE_TAGS_: self.tags,
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
        self._span.set_attribute(SPAN_ATTRIBUTE_TAGS_, self.tags)

        self._span.__exit__(exc_type, exc_value, traceback)

        OPEN_SPANS.discard(self._span)  # type: ignore [arg-type]

    @property
    def span_id(self) -> str:
        if self._span is None:
            raise ValueError("Span is not active")
        return trace_api.format_span_id(self._span.get_span_context().span_id)

    @property
    def trace_id(self) -> str:
        if self._span is None:
            raise ValueError("Span is not active")
        return trace_api.format_trace_id(self._span.get_span_context().trace_id)

    @property
    def is_recording(self) -> bool:
        if self._span is None:
            return False
        return self._span.is_recording()

    def set_tags(self, tags: t.Sequence[str]) -> None:
        tags = [tags] if isinstance(tags, str) else list(tags)
        tags = [clean_str(t) for t in tags]
        self.tags = uniquify_sequence(tags)

    def add_tags(self, tags: t.Sequence[str]) -> None:
        tags = [tags] if isinstance(tags, str) else list(tags)
        self.set_tags([*self.tags, *tags])

    def set_attribute(
        self,
        key: str,
        value: t.Any,
        *,
        schema: bool = True,
        raw: bool = False,
    ) -> None:
        self._added_attributes = True
        if schema and raw is False:
            self._schema[key] = create_json_schema(value, set())
        otel_value = self._pre_attributes[key] = value if raw else prepare_otlp_attribute(value)
        if self._span is not None:
            self._span.set_attribute(key, otel_value)
        self._pre_attributes[key] = otel_value

    def set_attributes(self, attributes: AnyDict) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def get_attributes(self) -> AnyDict:
        if self._span is not None:
            return getattr(self._span, "attributes", {})
        return self._pre_attributes

    def get_attribute(self, key: str, default: t.Any) -> t.Any:
        return self.get_attributes().get(key, default)

    def log_event(
        self,
        name: str,
        attributes: AnyDict | None = None,
    ) -> None:
        if self._span is not None:
            self._span.add_event(
                name,
                attributes=prepare_otlp_attributes(attributes or {}),
            )


class RunContext(te.TypedDict):
    """Context for transferring and continuing runs in other places."""

    run_id: str
    run_name: str
    project: str
    trace_context: dict[str, str]


class RunUpdateSpan(Span):
    def __init__(
        self,
        run_id: str,
        tracer: Tracer,
        project: str,
        *,
        metrics: MetricsDict | None = None,
        params: JsonDict | None = None,
        inputs: list[ObjectRef] | None = None,
        outputs: list[ObjectRef] | None = None,
        objects: dict[str, Object] | None = None,
        object_schemas: dict[str, JsonDict] | None = None,
    ) -> None:
        attributes: AnyDict = {
            SPAN_ATTRIBUTE_RUN_ID: run_id,
            SPAN_ATTRIBUTE_PROJECT: project,
            **({SPAN_ATTRIBUTE_METRICS: metrics} if metrics else {}),
            **({SPAN_ATTRIBUTE_PARAMS: params} if params else {}),
            **({SPAN_ATTRIBUTE_INPUTS: inputs} if inputs else {}),
            **({SPAN_ATTRIBUTE_OUTPUTS: outputs} if outputs else {}),
            **({SPAN_ATTRIBUTE_OBJECTS: objects} if objects else {}),
            **({SPAN_ATTRIBUTE_OBJECT_SCHEMAS: object_schemas} if object_schemas else {}),
        }

        # Mark objects and schemas as large attributes if present
        if objects or object_schemas:
            large_attrs = []
            if objects:
                large_attrs.append(SPAN_ATTRIBUTE_OBJECTS)
            if object_schemas:
                large_attrs.append(SPAN_ATTRIBUTE_OBJECT_SCHEMAS)
            attributes[SPAN_ATTRIBUTE_LARGE_ATTRIBUTES] = large_attrs

        super().__init__(f"run.{run_id}.update", attributes, tracer, type="run_update")


class RunSpan(Span):
    def __init__(
        self,
        name: str,
        project: str,
        attributes: AnyDict,
        tracer: Tracer,
        file_system: AbstractFileSystem,
        prefix_path: str,
        *,
        params: AnyDict | None = None,
        metrics: MetricsDict | None = None,
        tags: t.Sequence[str] | None = None,
        autolog: bool = True,
        update_frequency: int = 5,
        run_id: str | ULID | None = None,
        type: SpanType = "run",
    ) -> None:
        self.autolog = autolog
        self.project = project

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

        # Update mechanics
        self._last_update_time = time.time()
        self._update_frequency = update_frequency
        self._pending_params = deepcopy(self._params)
        self._pending_inputs = deepcopy(self._inputs)
        self._pending_outputs = deepcopy(self._outputs)
        self._pending_metrics = deepcopy(self._metrics)
        self._pending_objects = deepcopy(self._objects)
        self._pending_object_schemas = deepcopy(self._object_schemas)

        self._context_token: Token[RunSpan | None] | None = None  # contextvars context
        self._remote_context: dict[str, str] | None = None  # remote run trace context
        self._remote_token: object | None = None
        self._file_system = file_system
        self._prefix_path = prefix_path

        attributes = {
            SPAN_ATTRIBUTE_RUN_ID: str(run_id or ULID()),
            SPAN_ATTRIBUTE_PROJECT: project,
            **attributes,
        }
        super().__init__(name, attributes, tracer, type=type, tags=tags)

    @classmethod
    def from_context(
        cls,
        context: RunContext,
        tracer: Tracer,
        file_system: AbstractFileSystem,
        prefix_path: str,
    ) -> "RunSpan":
        self = RunSpan(
            name=f"run.{context['run_id']}.fragment",
            project=context["project"],
            attributes={},
            tracer=tracer,
            file_system=file_system,
            prefix_path=prefix_path,
            type="run_fragment",
            run_id=context["run_id"],
        )

        self._remote_context = context["trace_context"]

        return self

    def __enter__(self) -> te.Self:
        if current_run_span.get() is not None:
            raise RuntimeError("You cannot start a run span within another run")

        if self._remote_context is not None:
            otel_context = propagate.extract(carrier=self._remote_context)
            self._remote_token = context_api.attach(otel_context)
        else:
            super().__enter__()

        self._context_token = current_run_span.set(self)
        self.push_update(force=True)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self._remote_context is not None:
            super().__enter__()  # Now we can open our actually span

        # When we finally close out the final span, include all the
        # full data attributes, so we can skip the update spans during
        # db queries later.
        self.set_attribute(SPAN_ATTRIBUTE_PARAMS, self._params, schema=False)
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

        if self._remote_token is not None:
            context_api.detach(self._remote_token)  # type: ignore [arg-type]

        if self._context_token is not None:
            current_run_span.reset(self._context_token)

    def push_update(self, *, force: bool = False) -> None:
        if self._span is None:
            return

        current_time = time.time()
        force_update = force or (current_time - self._last_update_time >= self._update_frequency)
        should_update = force_update and (
            self._pending_params
            or self._pending_inputs
            or self._pending_outputs
            or self._pending_metrics
            or self._pending_objects
            or self._pending_object_schemas
        )

        if not should_update:
            return

        with RunUpdateSpan(
            run_id=self.run_id,
            project=self.project,
            tracer=self._tracer,
            metrics=self._pending_metrics if self._pending_metrics else None,
            params=self._pending_params if self._pending_params else None,
            inputs=self._pending_inputs if self._pending_inputs else None,
            outputs=self._pending_outputs if self._pending_outputs else None,
            objects=self._pending_objects if self._pending_objects else None,
            object_schemas=self._pending_object_schemas if self._pending_object_schemas else None,
        ):
            pass

        self._pending_metrics.clear()
        self._pending_params.clear()
        self._pending_inputs.clear()
        self._pending_outputs.clear()
        self._pending_objects.clear()
        self._pending_object_schemas.clear()

        self._last_update_time = current_time

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
        serialized = serialize(value)
        data_hash = serialized.data_hash
        schema_hash = serialized.schema_hash

        # Create a composite key that represents both data and schema
        hash_input = f"{data_hash}:{schema_hash}"
        composite_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:16]  # noqa: S324 # nosec

        # Store schema if new
        if schema_hash not in self._object_schemas:
            self._object_schemas[schema_hash] = serialized.schema
            self._pending_object_schemas[schema_hash] = serialized.schema

        # Check if we already have this exact composite hash
        if composite_hash not in self._objects:
            # Create a new object, but use the data_hash for deduplication of storage
            obj = self._create_object_by_hash(serialized, composite_hash)

            # Store with composite hash so we can look it up by the combination
            self._objects[composite_hash] = obj
            self._pending_objects[composite_hash] = obj

        # Build event attributes, use composite hash in events
        event_attributes = {
            **attributes,
            EVENT_ATTRIBUTE_OBJECT_HASH: composite_hash,
            EVENT_ATTRIBUTE_ORIGIN_SPAN_ID: trace_api.format_span_id(
                trace_api.get_current_span().get_span_context().span_id,
            ),
        }
        if label is not None:
            event_attributes[EVENT_ATTRIBUTE_OBJECT_LABEL] = label

        self.log_event(name=event_name, attributes=event_attributes)
        self.push_update()

        return composite_hash

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

    def _create_object_by_hash(self, serialized: Serialized, object_hash: str) -> Object:
        """Create an ObjectVal or ObjectUri depending on size with a specific hash."""
        data = serialized.data
        data_bytes = serialized.data_bytes
        data_len = serialized.data_len
        data_hash = serialized.data_hash
        schema_hash = serialized.schema_hash

        if data is None or data_bytes is None or data_len <= MAX_INLINE_OBJECT_BYTES:
            return ObjectVal(
                hash=object_hash,
                value=data,
                schema_hash=schema_hash,
            )

        # Offload to file system (e.g., S3)
        # For storage efficiency, still use just the data_hash for the file path
        # This ensures we don't duplicate storage for the same data
        full_path = f"{self._prefix_path.rstrip('/')}/{data_hash}"
        object_uri = self._store_file_by_hash(data_bytes, full_path)

        return ObjectUri(
            hash=object_hash,
            uri=object_uri,
            schema_hash=schema_hash,
            size=data_len,
        )

    def get_object(self, hash_: str) -> t.Any:
        return self._objects[hash_]

    def link_objects(
        self,
        object_hash: str,
        link_hash: str,
        **attributes: JsonValue,
    ) -> None:
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
            self._pending_params[key] = value

        # Params should get pushed immediately
        self.push_update(force=True)

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
        label = label or clean_str(name)
        hash_ = self.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_INPUT,
        )
        object_ref = ObjectRef(name, label=label, hash=hash_, attributes=attributes)
        self._inputs.append(object_ref)
        self._pending_inputs.append(object_ref)

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
    def metrics(self) -> MetricsDict:
        return self._metrics

    @t.overload
    def log_metric(
        self,
        name: str,
        value: float | bool,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        prefix: str | None = None,
        attributes: JsonDict | None = None,
    ) -> Metric: ...

    @t.overload
    def log_metric(
        self,
        name: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
        prefix: str | None = None,
    ) -> Metric: ...

    def log_metric(
        self,
        name: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        prefix: str | None = None,
        attributes: JsonDict | None = None,
    ) -> Metric:
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value), step, timestamp or datetime.now(timezone.utc), attributes or {}
            )
        )

        key = clean_str(name)
        if prefix is not None:
            key = f"{prefix}.{key}"

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
        self._pending_metrics.setdefault(key, []).append(metric)

        return metric

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
        label = label or clean_str(name)
        hash_ = self.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_OUTPUT,
        )
        object_ref = ObjectRef(name, label=label, hash=hash_, attributes=attributes)
        self._outputs.append(object_ref)
        self._pending_outputs.append(object_ref)


class TaskSpan(Span, t.Generic[R]):
    def __init__(
        self,
        name: str,
        attributes: AnyDict,
        run_id: str,
        tracer: Tracer,
        *,
        label: str | None = None,
        params: AnyDict | None = None,
        metrics: MetricsDict | None = None,
        tags: t.Sequence[str] | None = None,
    ) -> None:
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
        self.set_attribute(SPAN_ATTRIBUTE_PARAMS, self._params)
        self.set_attribute(SPAN_ATTRIBUTE_INPUTS, self._inputs, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_METRICS, self._metrics, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OUTPUTS, self._outputs, schema=False)
        super().__exit__(exc_type, exc_value, traceback)
        if self._context_token is not None:
            current_task_span.reset(self._context_token)

    @property
    def run_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_RUN_ID, ""))

    @property
    def parent_task_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_PARENT_TASK_ID, ""))

    @property
    def run(self) -> RunSpan:
        if self._run is None:
            raise ValueError("Task span is not in an active run")
        return self._run

    @property
    def outputs(self) -> AnyDict:
        return {ref.name: self.run.get_object(ref.hash) for ref in self._outputs}

    @property
    def output(self) -> R:
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
        label = label or clean_str(name)
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_OUTPUT,
        )
        self._outputs.append(ObjectRef(name, label=label, hash=hash_, attributes=attributes))
        return hash_

    @property
    def params(self) -> AnyDict:
        return self._params

    def log_param(self, key: str, value: t.Any) -> None:
        self.log_params(**{key: value})

    def log_params(self, **params: t.Any) -> None:
        self._params.update(params)

    @property
    def inputs(self) -> AnyDict:
        return {ref.name: self.run.get_object(ref.hash) for ref in self._inputs}

    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        **attributes: JsonValue,
    ) -> str:
        label = label or clean_str(name)
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_INPUT,
        )
        self._inputs.append(ObjectRef(name, label=label, hash=hash_, attributes=attributes))
        return hash_

    @property
    def metrics(self) -> dict[str, list[Metric]]:
        return self._metrics

    @t.overload
    def log_metric(
        self,
        name: str,
        value: float | bool,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> Metric: ...

    @t.overload
    def log_metric(
        self,
        name: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
    ) -> Metric: ...

    def log_metric(
        self,
        name: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: JsonDict | None = None,
    ) -> Metric:
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value), step, timestamp or datetime.now(timezone.utc), attributes or {}
            )
        )

        key = clean_str(name)

        # For every metric we log, also log it to the run
        # with our `label` as a prefix.
        #
        # Let the run handle the origin and mode aggregation
        # for us as we don't have access to the other times
        # this task-metric was logged here.

        if (run := current_run_span.get()) is not None:
            metric = run.log_metric(key, metric, prefix=self._label, origin=origin, mode=mode)

        self._metrics.setdefault(key, []).append(metric)

        return metric

    def get_average_metric_value(self, key: str | None = None) -> float:
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
    return {key: prepare_otlp_attribute(value) for key, value in attributes.items()}


def prepare_otlp_attribute(value: t.Any) -> otel_types.AttributeValue:
    if isinstance(value, str | int | bool | float):
        return value
    return json_dumps(value)
