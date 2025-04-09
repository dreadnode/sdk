from __future__ import annotations

import contextlib
import inspect
import os
import random
import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import coolname  # type: ignore
import logfire
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from logfire._internal.stack_info import get_filepath_attribute, warn_at_user_stacklevel
from logfire._internal.utils import safe_repr
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from .api.client import ApiClient
from .constants import ENV_API_TOKEN, ENV_LOCAL_DIR, ENV_PROJECT, ENV_SERVER_URL
from .exporters import FileExportConfig, FileMetricReader, FileSpanExporter
from .score import Scorer, ScorerCallable, T
from .task import P, R, Task
from .tracing import JsonValue, RunSpan, Score, Span, TaskSpan, current_run_span, current_task_span
from .version import VERSION


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
        local_dir: str | Path | t.Literal[False] = False,
        project: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | t.Literal[False, True] = True,
        send_to_logfire: bool | t.Literal["if-token-present"] = "if-token-present",
        otel_scope: str = "dreadnode",
    ) -> None:
        self._initialized = False

        self.server = server or os.environ.get(ENV_SERVER_URL)
        self.token = token or os.environ.get(ENV_API_TOKEN)

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
            config = FileExportConfig(base_path=self.local_dir, prefix=self.project + "-" if self.project else "")
            span_processors.append(BatchSpanProcessor(FileSpanExporter(config)))
            metric_readers.append(FileMetricReader(config))

        if self.server is not None:
            if self.token is None:
                raise ValueError(f"Token ({ENV_API_TOKEN}) must be provided when server is set")

            self._api = ApiClient(self.server, self.token)

            headers = {"User-Agent": f"dreadnode/{VERSION}", "X-Api-Key": self.token}
            span_processors.append(
                BatchSpanProcessor(
                    RemovePendingSpansExporter(  # This will tell Logfire to emit pending spans to us as well
                        OTLPSpanExporter(
                            endpoint=urljoin(self.server, "/api/otel/traces"),
                            headers=headers,
                            compression=Compression.Gzip,
                        )
                    )
                )
            )
            # TODO: Metrics
            # metric_readers.append(
            #     PeriodicExportingMetricReader(
            #         OTLPMetricExporter(
            #             endpoint=urljoin(self.server, "/v1/metrics"),
            #             headers=headers,
            #             compression=Compression.Gzip,
            #             # TODO: preferred_temporality
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
    def api(self, *, base_url: str | None = None, token: str | None = None) -> ApiClient:
        if base_url is not None and token is not None:
            return ApiClient(base_url, token)

        if not self._initialized:
            raise RuntimeError("Call .configure() before accessing the API")

        if self._api is None:
            raise RuntimeError("API is not available without a server configuration")

        return self._api

    def _get_tracer(self, *, is_span_tracer: bool = True) -> Tracer:
        return self._logfire._tracer_provider.get_tracer(
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
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]]], Task[P, R]]: ...

    @t.overload
    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]],
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]]], Task[P, R]]: ...

    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[t.Any] | ScorerCallable[R]] | None = None,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[t.Callable[P, t.Awaitable[R]]], Task[P, R]]:
        def make_task(func: t.Callable[P, t.Awaitable[R]]) -> Task[P, R]:
            func = inspect.unwrap(func)

            qualified_func_name = func_name = getattr(func, "__qualname__", getattr(func, "__name__", safe_repr(func)))

            with contextlib.suppress(Exception):
                qualified_func_name = f"{inspect.getmodule(func).__name__}.{func_name}"  # type: ignore

            _name = name or qualified_func_name

            _attributes = attributes or {}
            _attributes["code.function"] = func_name
            with contextlib.suppress(Exception):
                _attributes["code.lineno"] = func.__code__.co_firstlineno
            with contextlib.suppress(Exception):
                _attributes.update(get_filepath_attribute(inspect.getsourcefile(func)))  # type: ignore

            return Task(
                tracer=self._get_tracer(),
                name=_name,
                attributes=_attributes,
                func=func,
                scorers=[
                    scorer if isinstance(scorer, Scorer) else Scorer.from_callable(self._get_tracer(), scorer)
                    for scorer in scorers or []
                ],
                tags=list(tags or []),
            )

        return make_task

    def task_span(
        self,
        name: str,
        *,
        tags: t.Sequence[str] | None = None,
        args: dict[str, t.Any] | None = None,
        **attributes: t.Any,
    ) -> TaskSpan[t.Any]:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Task spans must be created within a run")

        return TaskSpan(
            name=name,
            args=args or {},
            attributes=attributes,
            tracer=self._get_tracer(),
            run_id=run.run_id,
            tags=tags,
        )

    def scorer(
        self,
        *,
        name: str | None = None,
        **attributes: t.Any,
    ) -> t.Callable[[ScorerCallable[T]], Scorer[T]]:
        def make_scorer(func: ScorerCallable[T]) -> Scorer[T]:
            return Scorer.from_callable(self._get_tracer(), func, name=name, attributes=attributes)

        return make_scorer

    def run(
        self,
        name: str | None = None,
        *,
        tags: t.Sequence[str] | None = None,
        project: str | None = None,
        **attributes: t.Any,
    ) -> RunSpan:
        if not self._initialized:
            self.initialize()

        if name is None:
            name = f"{coolname.generate_slug(2)}-{random.randint(100, 999)}"

        return RunSpan(
            name=name,
            project=project or self.project or "default",
            attributes=attributes,
            tracer=self._get_tracer(),
            tags=tags,
        )

    def push_update(self) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Run updates must be pushed within a run")

        run.push_update()

    def log_param(self, key: str, value: JsonValue) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Params must be set within a run")
        run.log_param(key, value)

    def log_params(self, **params: JsonValue) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Params must be set within a run")
        run.log_params(**params)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int = 0,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Metrics must be logged within a run")
        run.log_metric(key, value, step=step, timestamp=timestamp)

    def log_score(self, score: Score) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Scores must be logged within a run")

        run.scores.append(score)

        if (task := current_task_span.get()) is None:
            return

        task.scores.append(score)


DEFAULT_INSTANCE = Dreadnode()
