from __future__ import annotations

import contextlib
import inspect
import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import logfire
from logfire._internal.stack_info import get_filepath_attribute
from logfire._internal.utils import safe_repr
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from .exporters import FileExportConfig, FileMetricReader, FileSpanExporter
from .task import P, R, Scorer, Task
from .tracing import JsonValue, RunSpan, Span, current_run_span
from .version import VERSION


@dataclass
class Dreadnode:
    server: str | None
    token: str | None
    local_dir: str | Path | t.Literal[False]
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
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | t.Literal[False, True] = True,
        send_to_logfire: bool | t.Literal["if-token-present"] = "if-token-present",
        otel_scope: str = "dreadnode",
    ) -> None:
        self.server = server
        self.token = token
        self.local_dir = local_dir
        self.service_name = service_name
        self.service_version = service_version
        self.console = console
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self._logfire = logfire.DEFAULT_LOGFIRE_INSTANCE
        self._initialized = False

    def configure(
        self,
        server: str | None = None,
        token: str | None = None,
        local_dir: str | Path | t.Literal[False] = False,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | t.Literal[False, True] = True,
        send_to_logfire: bool | t.Literal["if-token-present"] = "if-token-present",
        otel_scope: str = "dreadnode",
    ) -> None:
        self._initialized = False

        self.server = server
        self.token = token
        self.local_dir = local_dir
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

        if self.local_dir is not False:
            config = FileExportConfig(base_path=self.local_dir)
            span_processors.append(BatchSpanProcessor(FileSpanExporter(config)))
            metric_readers.append(FileMetricReader(config))

        if self.server is not None:
            if self.token is None:
                raise ValueError("Token must be provided when server is set")
            headers = {"User-Agent": f"dreadnode/{VERSION}", "X-Api-Key": self.token}
            span_processors.append(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=urljoin(self.server, "/api/otel/traces"),
                        headers=headers,
                        compression=Compression.Gzip,
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
        )

        self._initialized = True

    @property
    def is_default(self) -> bool:
        return self is DEFAULT_INSTANCE

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
        /,
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

    def task(
        self,
        /,
        *,
        name: str | None = None,
        scorers: list[Scorer[R]] | None = None,
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

            return Task[P, R](
                tracer=self._get_tracer(),
                name=_name,
                attributes=_attributes,
                func=func,
                scorers=scorers or [],
                tags=tags or [],
            )

        return make_task

    def run(
        self,
        name: str,
        /,
        *,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> RunSpan:
        return RunSpan(
            name=name,
            attributes=attributes,
            tracer=self._get_tracer(),
            tags=tags,
        )

    def log_param(self, key: str, value: JsonValue) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Params must be set within a run")
        run.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = 0, *, timestamp: datetime | None = None) -> None:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Metrics must be set within a run")
        run.log_metric(key, value, step=step, timestamp=timestamp)


DEFAULT_INSTANCE = Dreadnode()
