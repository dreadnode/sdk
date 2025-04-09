import re
import types
import typing as t
from contextvars import ContextVar, Token
from copy import deepcopy
from datetime import datetime, timezone

import typing_extensions as te
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

from dreadnode.metric import Metric, MetricDict
from dreadnode.object import Object, ObjectRef, ObjectVal
from dreadnode.serialization import serialize
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

R = t.TypeVar("R")


current_task_span: ContextVar["TaskSpan[t.Any] | None"] = ContextVar(
    "current_task_span",
    default=None,
)
current_run_span: ContextVar["RunSpan | None"] = ContextVar("current_run_span", default=None)


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

        context_api.detach(self._token)
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

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(self.get_attribute(SPAN_ATTRIBUTE_TAGS_, ()))

    @tags.setter
    def tags(self, new_tags: t.Sequence[str]) -> None:
        self.set_attribute(SPAN_ATTRIBUTE_TAGS_, uniquify_sequence(new_tags))

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


class RunUpdateSpan(Span):
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
    def __init__(
        self,
        name: str,
        project: str,
        attributes: AnyDict,
        tracer: Tracer,
        params: AnyDict | None = None,
        metrics: MetricDict | None = None,
        run_id: str | None = None,
        tags: t.Sequence[str] | None = None,
    ) -> None:
        self._params = params or {}
        self._metrics = metrics or {}
        self._objects: dict[str, Object] = {}
        self._object_schemas: dict[str, JsonDict] = {}
        self._inputs: list[ObjectRef] = []
        self._outputs: list[ObjectRef] = []
        self.project = project

        self._last_pushed_params = deepcopy(self._params)
        self._last_pushed_metrics = deepcopy(self._metrics)

        self._context_token: Token[RunSpan | None] | None = None  # contextvars context

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
        self.set_attribute(SPAN_ATTRIBUTE_METRICS, self._metrics, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OBJECTS, self._objects, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OBJECT_SCHEMAS, self._object_schemas, schema=False)
        self.set_attribute(SPAN_ATTRIBUTE_OUTPUTS, self._outputs, schema=False)

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
        serialized = serialize(value)
        object_ = ObjectVal(
            hash=serialized.data_hash,
            value=serialized.data,
            schema_hash=serialized.schema_hash,
        )

        # check size and offload to s3 if needed
        # if len(serialized.data) > 1 * 1024 * 1024:
        #     pass

        if serialized.data_hash not in self._objects:
            self._objects[serialized.data_hash] = object_

        if serialized.schema_hash not in self._object_schemas:
            self._object_schemas[serialized.schema_hash] = serialized.schema

        attributes = {
            **attributes,
            EVENT_ATTRIBUTE_OBJECT_HASH: object_.hash,
            EVENT_ATTRIBUTE_ORIGIN_SPAN_ID: trace_api.format_span_id(
                trace_api.get_current_span().get_span_context().span_id,
            ),
        }
        if label is not None:
            attributes[EVENT_ATTRIBUTE_OBJECT_LABEL] = label
        self.log_event(name=event_name, attributes=attributes)
        return object_.hash

    def get_object(self, hash_: str) -> t.Any:
        return self._objects[hash_]

    def link_objects(self, object_hash: str, link_hash: str, **attributes: JsonValue) -> None:
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

        if self._span is None:
            return

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
    ) -> None:
        ...

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
    ) -> None:
        ...

    def log_metric(
        self,
        key: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(float(value), step, timestamp or datetime.now(timezone.utc))
        )

        if origin is not None:
            origin_hash = self.log_object(
                origin,
                label=key,
                event_name=EVENT_NAME_OBJECT_METRIC,
            )
            metric.attributes[METRIC_ATTRIBUTE_SOURCE_HASH] = origin_hash

        self._metrics.setdefault(key, []).append(metric)
        if self._span is None:
            return

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
    ) -> None:
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_OUTPUT,
            **attributes,
        )
        self._outputs.append(ObjectRef(name, label=label, hash=hash_))

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
    ) -> None:
        label = label or re.sub(r"\W+", "_", name.lower())
        hash_ = self.run.log_object(
            value,
            label=label,
            event_name=EVENT_NAME_OBJECT_INPUT,
            **attributes,
        )
        self._inputs.append(ObjectRef(name, label=label, hash=hash_))

    @property
    def metrics(self) -> dict[str, list[Metric]]:
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
    ) -> None:
        ...

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
    ) -> None:
        ...

    def log_metric(
        self,
        key: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(float(value), step, timestamp or datetime.now(timezone.utc))
        )

        if origin is not None:
            origin_hash = self.run.log_object(
                origin,
                label=key,
                event_name=EVENT_NAME_OBJECT_METRIC,
            )
            metric.attributes[METRIC_ATTRIBUTE_SOURCE_HASH] = origin_hash

        self._metrics.setdefault(key, []).append(metric)

        # For every metric we log, also log it to the run
        # with our `label` as a prefix.
        #
        # Don't include `source` as we handled it here.
        if (run := current_run_span.get()) is not None:
            run.log_metric(f"{self._label}.{key}", metric)

    def get_average_metric_value(self, key: str | None = None) -> float:
        metrics = (
            self._metrics.get(key, [])
            if key is not None
            else [m for ms in self._metrics.values() for m in ms]
        )
        return sum(metric.value for metric in metrics) / len(
            metrics,
        )


def prepare_otlp_attributes(attributes: AnyDict) -> dict[str, otel_types.AttributeValue]:
    return {key: prepare_otlp_attribute(value) for key, value in attributes.items()}


def prepare_otlp_attribute(value: t.Any) -> otel_types.AttributeValue:
    if isinstance(value, str | int | bool | float):
        return value
    return json_dumps(value)
