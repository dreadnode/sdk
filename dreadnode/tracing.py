import typing as t
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime

import typing_extensions as te
from logfire._internal.json_encoder import logfire_json_dumps as json_dumps
from logfire._internal.json_schema import JsonSchemaProperties, attributes_json_schema, create_json_schema
from logfire._internal.utils import uniquify_sequence
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Tracer
from opentelemetry.util import types as otel_types
from ulid import ULID

from .constants import (
    SPAN_ATTRIBUTE_PARENT_TASK_ID_KEY,
    SPAN_ATTRIBUTE_PROJECT_KEY,
    SPAN_ATTRIBUTE_RUN_ID_KEY,
    SPAN_ATTRIBUTE_RUN_METRICS_KEY,
    SPAN_ATTRIBUTE_RUN_PARAMS_KEY,
    SPAN_ATTRIBUTE_SCHEMA_KEY,
    SPAN_ATTRIBUTE_TAGS_KEY,
    SPAN_ATTRIBUTE_TASK_ARGS_KEY,
    SPAN_ATTRIBUTE_TASK_OUTPUT_KEY,
    SPAN_ATTRIBUTE_TASK_SCORES_KEY,
    SPAN_ATTRIBUTE_TYPE_KEY,
    SpanType,
)

R = t.TypeVar("R")

JsonValue = t.Union[int, float, str, bool, None, list["JsonValue"], tuple["JsonValue", ...], "JsonDict"]
JsonDict = dict[str, JsonValue]

current_task_span: ContextVar["TaskSpan[t.Any] | None"] = ContextVar("current_task_span", default=None)
current_run_span: ContextVar["RunSpan | None"] = ContextVar("current_run_span", default=None)


@dataclass
class Score:
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: JsonDict = field(default_factory=dict)

    @classmethod
    def from_many(cls, name: str, values: t.Sequence[tuple[str, float, float]]) -> "Score":
        "Create a composite score from individual values and weights."
        total = sum(value * weight for _, value, weight in values)
        weight = sum(weight for _, _, weight in values)
        return cls(name, total / weight, attributes={name: value for name, value, _ in values})


@dataclass
class Metric:
    value: float
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


MetricDict = dict[str, list[Metric]]


class Span(ReadableSpan):
    def __init__(
        self,
        name: str,
        attributes: dict[str, t.Any],
        tracer: Tracer,
        type: SpanType = "span",
        tags: t.Sequence[str] | None = None,
    ) -> None:
        self._span_name = name
        self._pre_attributes = {
            SPAN_ATTRIBUTE_TYPE_KEY: type,
            SPAN_ATTRIBUTE_TAGS_KEY: tuple(uniquify_sequence(tags or ())),
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

        if self._token is None:
            self._token = context_api.attach(trace_api.set_span_in_context(self._span))

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: t.Any) -> None:
        if self._token is None or self._span is None:
            return

        context_api.detach(self._token)
        self._token = None

        if not self._span.is_recording():
            return

        self._span.set_attribute(SPAN_ATTRIBUTE_SCHEMA_KEY, attributes_json_schema(self._schema))
        self._span.__exit__(exc_type, exc_value, traceback)

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
        return tuple(self.get_attribute(SPAN_ATTRIBUTE_TAGS_KEY, ()))

    @tags.setter
    def tags(self, new_tags: t.Sequence[str]) -> None:
        self.set_attribute(SPAN_ATTRIBUTE_TAGS_KEY, tuple(uniquify_sequence(new_tags)))

    def set_attribute(self, key: str, value: t.Any) -> None:
        self._added_attributes = True
        self._schema[key] = create_json_schema(value, set())
        otel_value = self._pre_attributes[key] = prepare_otlp_attribute(value)
        if self._span is not None:
            self._span.set_attribute(key, otel_value)
        self._pre_attributes[key] = otel_value

    def set_attributes(self, attributes: dict[str, t.Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def get_attributes(self) -> dict[str, t.Any]:
        if self._span is not None:
            return getattr(self._span, "attributes", {})
        return self._pre_attributes

    def get_attribute(self, key: str, default: t.Any) -> t.Any:
        return self.get_attributes().get(key, default)


class RunUpdateSpan(Span):
    def __init__(
        self,
        run_id: str,
        tracer: Tracer,
        project: str,
        *,
        metrics: MetricDict | None = None,
        params: JsonDict | None = None,
    ) -> None:
        attributes: dict[str, t.Any] = {
            SPAN_ATTRIBUTE_RUN_ID_KEY: run_id,
            SPAN_ATTRIBUTE_PROJECT_KEY: project,
        }

        if metrics:
            attributes[SPAN_ATTRIBUTE_RUN_METRICS_KEY] = metrics
        if params:
            attributes[SPAN_ATTRIBUTE_RUN_PARAMS_KEY] = params

        super().__init__(f"run.{run_id}.update", attributes, tracer, "run_update")


class RunSpan(Span):
    def __init__(
        self,
        name: str,
        project: str,
        attributes: dict[str, t.Any],
        tracer: Tracer,
        run_id: str | None = None,
        tags: t.Sequence[str] | None = None,
    ) -> None:
        self._params: JsonDict = {}
        self._metrics: dict[str, list[Metric]] = {}
        self.scores: list[Score] = []
        self.project = project

        self._context_token: Token[RunSpan | None] | None = None  # contextvars context

        attributes = {
            SPAN_ATTRIBUTE_RUN_ID_KEY: str(run_id or ULID()),
            SPAN_ATTRIBUTE_PROJECT_KEY: project,
            **attributes,
        }
        super().__init__(name, attributes, tracer, "run", tags)

    def __enter__(self) -> te.Self:
        self._context_token = current_run_span.set(self)
        return super().__enter__()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: t.Any) -> None:
        from .score import scores_to_metrics

        score_metrics = scores_to_metrics(self.scores)
        self._metrics.update(score_metrics)

        self.set_attribute(SPAN_ATTRIBUTE_RUN_PARAMS_KEY, self._params)
        self.set_attribute(SPAN_ATTRIBUTE_RUN_METRICS_KEY, self._metrics)
        super().__exit__(exc_type, exc_value, traceback)
        if self._context_token is not None:
            current_run_span.reset(self._context_token)

    @property
    def run_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_RUN_ID_KEY, ""))

    @property
    def params(self) -> JsonDict:
        return self._params

    def log_param(self, key: str, value: JsonValue) -> None:
        self.log_params(**{key: value})

    def log_params(self, **params: JsonValue) -> None:
        for key, value in params.items():
            self.params[key] = value

        if self._span is None:
            return

        with RunUpdateSpan(run_id=self.run_id, project=self.project, tracer=self._tracer, params=self._params):
            pass

    @property
    def metrics(self) -> dict[str, list[Metric]]:
        return self._metrics

    def log_metric(self, key: str, value: float, step: int = 0, *, timestamp: datetime | None = None) -> None:
        metric = Metric(value, step, timestamp or datetime.now())
        self._metrics.setdefault(key, []).append(metric)
        if self._span is None:
            return

        with RunUpdateSpan(run_id=self.run_id, project=self.project, tracer=self._tracer, metrics=self._metrics):
            pass

    @property
    def scores(self) -> list[Score]:
        return self._scores

    @scores.setter
    def scores(self, value: list[Score]) -> None:
        self._scores = value


class TaskSpan(Span, t.Generic[R]):
    def __init__(
        self,
        name: str,
        attributes: dict[str, t.Any],
        args: dict[str, t.Any],
        run_id: str,
        tracer: Tracer,
        tags: t.Sequence[str] | None = None,
    ) -> None:
        self._args = args
        self._output: R | None = None
        self._scores: list[Score] = []

        self._context_token: Token[TaskSpan[t.Any] | None] | None = None  # contextvars context

        attributes = {
            SPAN_ATTRIBUTE_RUN_ID_KEY: str(run_id),
            SPAN_ATTRIBUTE_TASK_ARGS_KEY: args,
            **attributes,
        }
        super().__init__(name, attributes, tracer, "task", tags)

    def __enter__(self) -> te.Self:
        self._parent_task = current_task_span.get()
        if self._parent_task is not None:
            self.set_attribute(SPAN_ATTRIBUTE_PARENT_TASK_ID_KEY, self._parent_task.span_id)
        self._context_token = current_task_span.set(self)
        return super().__enter__()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: t.Any) -> None:
        self.set_attribute(SPAN_ATTRIBUTE_TASK_OUTPUT_KEY, self._output)
        self.set_attribute(SPAN_ATTRIBUTE_TASK_ARGS_KEY, self._args)
        self.set_attribute(SPAN_ATTRIBUTE_TASK_SCORES_KEY, self._scores)
        super().__exit__(exc_type, exc_value, traceback)
        if self._context_token is not None:
            current_task_span.reset(self._context_token)

    @property
    def run_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_RUN_ID_KEY, ""))

    @property
    def parent_task_id(self) -> str:
        return str(self.get_attribute(SPAN_ATTRIBUTE_PARENT_TASK_ID_KEY, ""))

    @property
    def output(self) -> R | None:
        return self._output

    @output.setter
    def output(self, value: R | None) -> None:
        self._output = value

    @property
    def args(self) -> dict[str, t.Any]:
        return self._args

    @args.setter
    def args(self, value: dict[str, t.Any]) -> None:
        self._args = value

    @property
    def scores(self) -> list[Score]:
        return self._scores

    @scores.setter
    def scores(self, value: list[Score]) -> None:
        self._scores = value

    @property
    def average_score(self) -> float:
        if not self._scores:
            return 0.0
        return sum(score.value for score in self._scores) / len(self._scores)


def prepare_otlp_attributes(attributes: dict[str, t.Any]) -> dict[str, otel_types.AttributeValue]:
    return {key: prepare_otlp_attribute(value) for key, value in attributes.items()}


def prepare_otlp_attribute(value: t.Any) -> otel_types.AttributeValue:
    if isinstance(value, str | int | bool | float):
        return value
    return json_dumps(value)
