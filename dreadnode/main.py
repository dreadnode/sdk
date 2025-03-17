from __future__ import annotations

import contextlib
import inspect
import os
import random
import re
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import coolname  # type: ignore [import-untyped]
import logfire
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from logfire._internal.stack_info import get_filepath_attribute, warn_at_user_stacklevel
from logfire._internal.utils import safe_repr
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .api.client import ApiClient
from .constants import (
    DEFAULT_SERVER_URL,
    ENV_API_KEY,
    ENV_API_TOKEN,
    ENV_LOCAL_DIR,
    ENV_PROJECT,
    ENV_SERVER,
    ENV_SERVER_URL,
    AnyDict,
    JsonValue,
)
from .exporters import FileExportConfig, FileMetricReader, FileSpanExporter
from .metric import Metric, Scorer, ScorerCallable, T
from .task import P, R, Task
from .tracing import (
    RunSpan,
    Span,
    TaskSpan,
    current_run_span,
    current_task_span,
)
from .version import VERSION

if t.TYPE_CHECKING:
    from opentelemetry.sdk.metrics.export import MetricReader
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Tracer

ToObject = t.Literal["task-or-run", "run"]


class DreadnodeConfigWarning(UserWarning):
    pass


@dataclass
class Dreadnode:
    server: str | None
    token: str | None
    local_dir: str | Path | t.Literal[False]
    project: str | None
    service_name: str | None
    service_version: str | None
    console: logfire.ConsoleOptions | t.Literal[False, True]
    send_to_logfire: bool | t.Literal["if-token-present"]
    otel_scope: str

    def __init__(
        self,
        *,
        server: str | None = None,
        token: str | None = None,
        local_dir: str | Path | t.Literal[False] = False,
        project: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | t.Literal[False, True] = True,
        send_to_logfire: bool | t.Literal["if-token-present"] = "if-token-present",
        otel_scope: str = "dreadnode",
    ) -> None:
        self.server = server
        self.token = token
        self.local_dir = local_dir
        self.project = project
        self.service_name = service_name
        self.service_version = service_version
        self.console = console
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self._api: ApiClient | None = None

        self._logfire = logfire.DEFAULT_LOGFIRE_INSTANCE
        self._logfire.config.ignore_no_config = True

        self._initialized = False

    def configure(
        self,
        server: str | None = None,
        token: str | None = None,
        local_dir: str | Path | t.Literal[False] = False,  # noqa: FBT002
        project: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | t.Literal[False, True] = True,  # noqa: FBT002
        send_to_logfire: bool | t.Literal["if-token-present"] = "if-token-present",
        otel_scope: str = "dreadnode",
    ) -> None:
        self._initialized = False

        self.server = server or os.environ.get(ENV_SERVER_URL) or os.environ.get(ENV_SERVER)
        self.token = token or os.environ.get(ENV_API_TOKEN) or os.environ.get(ENV_API_KEY)

        if local_dir is False and ENV_LOCAL_DIR in os.environ:
            env_local_dir = os.environ.get(ENV_LOCAL_DIR)
            if env_local_dir:
                self.local_dir = Path(env_local_dir)
            else:
                self.local_dir = False
        else:
            self.local_dir = local_dir

        self.project = project or os.environ.get(ENV_PROJECT)
        self.service_name = service_name
        self.service_version = service_version
        self.console = console
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self.initialize()

    def initialize(self) -> None:
        if self._initialized:
            return

        span_processors: list[SpanProcessor] = []
        metric_readers: list[MetricReader] = []

        if self.server is None and self.local_dir is False and self.send_to_logfire is not True:
            warn_at_user_stacklevel(
                "Your current configuration won't persist run data anywhere. "
                f"Use `dreadnode.init(server=..., token=...)`, `dreadnode.init(local_dir=...)`, or use environment variables ({ENV_SERVER_URL}, {ENV_API_TOKEN}, {ENV_LOCAL_DIR}).",
                category=DreadnodeConfigWarning,
            )

        if self.local_dir is not False:
            config = FileExportConfig(
                base_path=self.local_dir,
                prefix=self.project + "-" if self.project else "",
            )
            span_processors.append(BatchSpanProcessor(FileSpanExporter(config)))
            metric_readers.append(FileMetricReader(config))

        if self.token is not None:
            self.server = self.server or DEFAULT_SERVER_URL
            self._api = ApiClient(self.server, self.token)

            headers = {"User-Agent": f"dreadnode/{VERSION}", "X-Api-Key": self.token}
            span_processors.append(
                BatchSpanProcessor(
                    RemovePendingSpansExporter(  # This will tell Logfire to emit pending spans to us as well
                        OTLPSpanExporter(
                            endpoint=urljoin(self.server, "/api/otel/traces"),
                            headers=headers,
                            compression=Compression.Gzip,
                        ),
                    ),
                ),
            )
            # TODO(nick): Metrics
            # https://linear.app/dreadnode/issue/ENG-1310/sdk-add-metrics-exports
            # metric_readers.append(
            #     PeriodicExportingMetricReader(
            #         OTLPMetricExporter(
            #             endpoint=urljoin(self.server, "/v1/metrics"),
            #             headers=headers,
            #             compression=Compression.Gzip,
            #             # preferred_temporality
            #         )
            #     )
            # )

        self._logfire = logfire.configure(
            local=not self.is_default,
            send_to_logfire=self.send_to_logfire,
            additional_span_processors=span_processors,
            metrics=logfire.MetricsOptions(additional_readers=metric_readers),
            service_name=self.service_name,
            service_version=self.service_version,
            console=logfire.ConsoleOptions() if self.console is True else self.console,
            scrubbing=False,
        )
        self._logfire.config.ignore_no_config = True

        self._initialized = True

    @property
    def is_default(self) -> bool:
        return self is DEFAULT_INSTANCE

    # I'd like to feel like a property as well,
    # but it won't work well for our lazy initialization
    def api(self, *, server: str | None = None, token: str | None = None) -> ApiClient:
        if server is not None and token is not None:
            return ApiClient(server, token)

        if not self._initialized:
            raise RuntimeError("Call .configure() before accessing the API")

        if self._api is None:
            raise RuntimeError("API is not available without a server configuration")

        return self._api

    def _get_tracer(self, *, is_span_tracer: bool = True) -> Tracer:
        return self._logfire._tracer_provider.get_tracer(  # noqa: SLF001
            self.otel_scope,
            VERSION,
            is_span_tracer=is_span_tracer,
        )

    def shutdown(self) -> None:
        if not self._initialized:
            return

        self._logfire.shutdown()

    def span(
        self,
        name: str,
        *,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> Span:
        return Span(
            name=name,
            attributes=attributes,
            tracer=self._get_tracer(),
            tags=tags,
        )

    @t.overload
    def task(
        self,
        *,
        scorers: None = None,
        name: str | None = None,
        kind: str | None = None,
        log_params: t.Sequence[str] | t.Literal[True] | None = None,
        log_inputs: t.Sequence[str] | t.Literal[True] | None = None,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]] | t.Callable[P, R]], Task[P, R]]:
        ...

    @t.overload
    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]],
        name: str | None = None,
        kind: str | None = None,
        log_params: t.Sequence[str] | t.Literal[True] | None = None,
        log_inputs: t.Sequence[str] | t.Literal[True] | None = None,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]] | t.Callable[P, R]], Task[P, R]]:
        ...

    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[t.Any] | ScorerCallable[R]] | None = None,
        name: str | None = None,
        kind: str | None = None,
        log_params: t.Sequence[str] | t.Literal[True] | None = None,
        log_inputs: t.Sequence[str] | t.Literal[True] | None = None,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]] | t.Callable[P, R]], Task[P, R]]:
        def make_task(func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R]) -> Task[P, R]:
            func = inspect.unwrap(func)

            if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
                raise TypeError("@task cannot be applied to generators")

            func_name = getattr(
                func,
                "__qualname__",
                getattr(func, "__name__", safe_repr(func)),
            )

            _name = name or func_name
            _kind = kind or func_name

            # conform our kind for sanity
            _kind = re.sub(r"[\W_]+", "_", _kind.lower())

            _attributes = attributes or {}
            _attributes["code.function"] = func_name
            with contextlib.suppress(Exception):
                _attributes["code.lineno"] = func.__code__.co_firstlineno
            with contextlib.suppress(Exception):
                _attributes.update(
                    get_filepath_attribute(inspect.getsourcefile(func)),  # type: ignore [arg-type]
                )

            return Task(
                tracer=self._get_tracer(),
                name=_name,
                attributes=_attributes,
                func=t.cast(t.Callable[P, R], func),
                scorers=[
                    scorer
                    if isinstance(scorer, Scorer)
                    else Scorer.from_callable(self._get_tracer(), scorer)
                    for scorer in scorers or []
                ],
                tags=list(tags or []),
                log_params=log_params,
                log_inputs=log_inputs,
                log_output=log_output,
                kind=_kind,
            )

        return make_task

    def task_span(
        self,
        name: str,
        *,
        kind: str | None = None,
        params: AnyDict | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> TaskSpan[t.Any]:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("task_span() must be called within a run")

        kind = kind or re.sub(r"[\W_]+", "_", name.lower())
        return TaskSpan(
            name=name,
            kind=kind,
            attributes=attributes,
            params=params,
            tags=tags,
            run_id=run.run_id,
            tracer=self._get_tracer(),
        )

    def scorer(
        self,
        *,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[ScorerCallable[T]], Scorer[T]]:
        def make_scorer(func: ScorerCallable[T]) -> Scorer[T]:
            return Scorer.from_callable(
                self._get_tracer(),
                func,
                name=name,
                tags=tags,
                attributes=attributes,
            )

        return make_scorer

    def run(
        self,
        name: str | None = None,
        *,
        tags: t.Sequence[str] | None = None,
        params: AnyDict | None = None,
        project: str | None = None,
        **attributes: t.Any,
    ) -> RunSpan:
        if not self._initialized:
            self.initialize()

        if name is None:
            name = f"{coolname.generate_slug(2)}-{random.randint(100, 999)}"  # noqa: S311

        return RunSpan(
            name=name,
            project=project or self.project or "default",
            attributes=attributes,
            tracer=self._get_tracer(),
            params=params,
            tags=tags,
        )

    def log_param(self, key: str, value: JsonValue, *, to: ToObject = "task-or-run") -> None:
        self.log_params(to=to, **{key: value})

    def log_params(self, to: ToObject = "task-or-run", **params: JsonValue) -> None:
        task = current_task_span.get()
        run = current_run_span.get()

        if to == "task-or-run":
            target = task or run
            if target is None:
                raise RuntimeError(
                    "log_params() with to='task-or-run' must be called within a run or a task",
                )
            target.log_params(**params)

        elif to == "run":
            if run is None:
                raise RuntimeError("log_params() with to='run' must be called within a run")
            run.log_params(**params)

    @t.overload
    def log_metric(
        self,
        key: str,
        value: float | bool,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        to: ToObject = "task-or-run",
    ) -> None:
        ...

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        to: ToObject = "task-or-run",
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
        to: ToObject = "task-or-run",
    ) -> None:
        task = current_task_span.get()
        run = current_run_span.get()

        metric = (
            value
            if isinstance(value, Metric)
            else Metric(float(value), step, timestamp or datetime.now(timezone.utc))
        )

        if to == "task-or-run":
            target = task or run
            if target is None:
                raise RuntimeError(
                    "log_metric() with to='task-or-run' must be called within a run or a task",
                )
            target.log_metric(key, metric, origin=origin)

        elif to == "run":
            if run is None:
                raise RuntimeError("log_metric() with to='run' must be called within a run")
            run.log_metric(key, metric, origin=origin)

    def log_input(
        self,
        name: str,
        value: JsonValue,
        *,
        kind: str | None = None,
        to: ToObject = "task-or-run",
        **attributes: t.Any,
    ) -> None:
        task = current_task_span.get()
        run = current_run_span.get()

        if to == "task-or-run":
            target = task or run
            if target is None:
                raise RuntimeError(
                    "log_inputs() with to='task-or-run' must be called within a run or a task",
                )
            target.log_input(name, value, kind=kind, **attributes)

        elif to == "run":
            if run is None:
                raise RuntimeError("log_inputs() with to='run' must be called within a run")
            run.log_input(name, value, kind=kind, **attributes)

    def log_inputs(
        self,
        to: ToObject = "task-or-run",
        **inputs: JsonValue,
    ) -> None:
        for name, value in inputs.items():
            self.log_input(name, value, to=to)

    def log_output(
        self,
        name: str,
        value: t.Any,
        *,
        kind: str | None = None,
        to: ToObject = "task-or-run",
        **attributes: JsonValue,
    ) -> None:
        task = current_task_span.get()
        run = current_run_span.get()

        if to == "task-or-run":
            target = task or run
            if target is None:
                raise RuntimeError(
                    "log_output() with to='task-or-run' must be called within a run or a task",
                )
            target.log_output(name, value, kind=kind, **attributes)

        elif to == "run":
            if run is None:
                raise RuntimeError("log_output() with to='run' must be called within a run")
            run.log_output(name, value, kind=kind, **attributes)

    def log_outputs(
        self,
        to: ToObject = "task-or-run",
        **outputs: JsonValue,
    ) -> None:
        for name, value in outputs.items():
            self.log_output(name, value, to=to)

    def link_objects(self, origin: t.Any, link: t.Any, **attributes: JsonValue) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("link() must be called within a run")

        origin_hash = run.log_object(origin)
        link_hash = run.log_object(link)
        run.link_objects(origin_hash, link_hash, **attributes)


DEFAULT_INSTANCE = Dreadnode()
