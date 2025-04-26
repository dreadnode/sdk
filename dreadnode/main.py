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
from fsspec.implementations.local import (  # type: ignore [import-untyped]
    LocalFileSystem,
)
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from logfire._internal.stack_info import get_filepath_attribute, warn_at_user_stacklevel
from logfire._internal.utils import safe_repr
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from s3fs import S3FileSystem  # type: ignore [import-untyped]

from dreadnode.api.client import ApiClient
from dreadnode.constants import (
    DEFAULT_SERVER_URL,
    ENV_API_KEY,
    ENV_API_TOKEN,
    ENV_LOCAL_DIR,
    ENV_PROJECT,
    ENV_SERVER,
    ENV_SERVER_URL,
)
from dreadnode.metric import Metric, MetricAggMode, Scorer, ScorerCallable, T
from dreadnode.task import P, R, Task
from dreadnode.tracing.exporters import (
    FileExportConfig,
    FileMetricReader,
    FileSpanExporter,
)
from dreadnode.tracing.span import (
    RunSpan,
    Span,
    TaskSpan,
    current_run_span,
    current_task_span,
)
from dreadnode.types import (
    AnyDict,
    JsonValue,
)
from dreadnode.util import handle_internal_errors
from dreadnode.version import VERSION

if t.TYPE_CHECKING:
    from fsspec import AbstractFileSystem  # type: ignore [import-untyped]
    from opentelemetry.sdk.metrics.export import MetricReader
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Tracer


ToObject = t.Literal["task-or-run", "run"]


class DreadnodeConfigWarning(UserWarning):
    pass


class DreadnodeUsageWarning(UserWarning):
    pass


@dataclass
class Dreadnode:
    """
    The core Dreadnode SDK class.

    A default instance of this class is created and can be used directly with `dreadnode.*`.

    Otherwise, you can create your own instance and configure it with `configure()`.
    """

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

        self._fs: AbstractFileSystem = LocalFileSystem(auto_mkdir=True)
        self._fs_prefix: str = ".dreadnode/storage/"

        self._initialized = False

    def configure(
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
        """
        Configure the Dreadnode SDK and call `initialize()`.

        This method should always be called before using the SDK.

        If `server` and `token` are not provided, the SDK will look in
        the associated environment variables:

        - `DREADNODE_SERVER_URL` or `DREADNODE_SERVER`
        - `DREADNODE_API_TOKEN` or `DREADNODE_API_KEY`

        Args:
            server: The Dreadnode server URL.
            token: The Dreadnode API token.
            local_dir: The local directory to store data in.
            project: The defautlt project name to associate all runs with.
            service_name: The service name to use for OpenTelemetry.
            service_version: The service version to use for OpenTelemetry.
            console: Whether to log span information to the console.
            send_to_logfire: Whether to send data to Logfire.
            otel_scope: The OpenTelemetry scope name.
        """

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
        """
        Initialize the Dreadnode SDK.

        This method is called automatically when you call `configure()`.
        """
        if self._initialized:
            return

        span_processors: list[SpanProcessor] = []
        metric_readers: list[MetricReader] = []

        self.server = self.server or DEFAULT_SERVER_URL
        if self.server is None and self.local_dir is False:
            warn_at_user_stacklevel(
                "Your current configuration won't persist run data anywhere. "
                "Use `dreadnode.init(server=..., token=...)`, `dreadnode.init(local_dir=...)`, "
                f"or use environment variables ({ENV_SERVER_URL}, {ENV_API_TOKEN}, {ENV_LOCAL_DIR}).",
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
            self._api = ApiClient(self.server, self.token)

            try:
                self._api.list_projects()
            except Exception as e:
                raise RuntimeError(
                    "Failed to authenticate with the provided server and token",
                ) from e

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

            credentials = self._api.get_user_data_credentials()
            self._fs = S3FileSystem(
                key=credentials.access_key_id,
                secret=credentials.secret_access_key,
                token=credentials.session_token,
                client_kwargs={
                    "endpoint_url": credentials.endpoint,
                    "region_name": credentials.region,
                },
            )
            self._fs_prefix = f"{credentials.bucket}/{credentials.prefix}/"

        self._logfire = logfire.configure(
            local=not self.is_default,
            send_to_logfire=self.send_to_logfire,
            additional_span_processors=span_processors,
            metrics=logfire.MetricsOptions(additional_readers=metric_readers),
            service_name=self.service_name,
            service_version=self.service_version,
            console=logfire.ConsoleOptions() if self.console is True else self.console,
            scrubbing=False,
            inspect_arguments=False,
        )
        self._logfire.config.ignore_no_config = True

        self._initialized = True

    @property
    def is_default(self) -> bool:
        return self is DEFAULT_INSTANCE

    def api(self, *, server: str | None = None, token: str | None = None) -> ApiClient:
        """
        Get an API client based on the current configuration or the provided server and token.

        If the server and token are not provided, the method will use the current configuration
        and `configure()` needs to be called first.

        Args:
            server: The server URL to use for the API client.
            token: The API token to use for authentication.

        Returns:
            An ApiClient instance.
        """
        if server is not None and token is not None:
            return ApiClient(server, token)

        if not self._initialized:
            raise RuntimeError("Call .configure() before accessing the API")

        if self._api is None:
            raise RuntimeError("API is not available without a server configuration")

        return self._api

    def _get_tracer(self, *, is_span_tracer: bool = True) -> "Tracer":
        return self._logfire._tracer_provider.get_tracer(  # noqa: SLF001
            self.otel_scope,
            VERSION,
            is_span_tracer=is_span_tracer,
        )

    @handle_internal_errors()
    def shutdown(self) -> None:
        """
        Shutdown any associate OpenTelemetry components and flush any pending spans.

        It is not required to call this method, as the SDK will automatically
        flush and shutdown when the process exits.

        However, if you want to ensure that all spans are flushed before
        exiting, you can call this method manually.
        """
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
        """
        Create a new OpenTelemety span.

        Spans are more lightweight than tasks, but still let you track
        work being performed and view it in the UI. You cannot
        log parameters, inputs, or outputs to spans.

        Example:
            ```
            with dreadnode.span("my_span") as span:
                # do some work here
                pass
            ```

        Args:
            name: The name of the span.
            tags: A list of tags to attach to the span.
            **attributes: A dictionary of attributes to attach to the span.

        Returns:
            A Span object.
        """
        return Span(
            name=name,
            attributes=attributes,
            tracer=self._get_tracer(),
            tags=tags,
        )

    # Some excessive typing here to ensure we can properly
    # overload our decorator for sync/async and cases
    # where we need the return type of the task to align
    # with the scorer inputs

    class TaskDecorator(t.Protocol):
        @t.overload
        def __call__(
            self,
            func: t.Callable[P, t.Awaitable[R]],
        ) -> Task[P, R]: ...

        @t.overload
        def __call__(
            self,
            func: t.Callable[P, R],
        ) -> Task[P, R]: ...

        def __call__(
            self,
            func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
        ) -> Task[P, R]: ...

    class ScoredTaskDecorator(t.Protocol, t.Generic[R]):
        @t.overload
        def __call__(
            self,
            func: t.Callable[P, t.Awaitable[R]],
        ) -> Task[P, R]: ...

        @t.overload
        def __call__(
            self,
            func: t.Callable[P, R],
        ) -> Task[P, R]: ...

        def __call__(
            self,
            func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
        ) -> Task[P, R]: ...

    @t.overload
    def task(
        self,
        *,
        scorers: None = None,
        name: str | None = None,
        label: str | None = None,
        log_params: t.Sequence[str] | bool = False,
        log_inputs: t.Sequence[str] | bool = True,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> TaskDecorator: ...

    @t.overload
    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]],
        name: str | None = None,
        label: str | None = None,
        log_params: t.Sequence[str] | bool = False,
        log_inputs: t.Sequence[str] | bool = True,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> ScoredTaskDecorator[R]: ...

    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[t.Any] | ScorerCallable[t.Any]] | None = None,
        name: str | None = None,
        label: str | None = None,
        log_params: t.Sequence[str] | bool = False,
        log_inputs: t.Sequence[str] | bool = True,
        log_output: bool = True,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> TaskDecorator:
        """
        Create a new task from a function.

        Example:
            ```
            @dreadnode.task(name="my_task")
            async def my_task(x: int) -> int:
                return x * 2

            await my_task(2)
            ```

        Args:
            scorers: A list of scorers to attach to the task. These will be called after every execution
                of the task and will be passed the task's output.
            name: The name of the task.
            label: The label of the task - useful for filtering in the UI.
            log_params: Whether to log all, or specific, incoming arguments to the function as parameters.
            log_inputs: Whether to log all, or specific, incoming arguments to the function as inputs.
            log_output: Whether to log the result of the function as an output.
            tags: A list of tags to attach to the task span.
            **attributes: A dictionary of attributes to attach to the task span.

        Returns:
            A new Task object.
        """

        def make_task(
            func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
        ) -> Task[P, R]:
            unwrapped = inspect.unwrap(func)

            if inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(
                unwrapped,
            ):
                raise TypeError("@task cannot be applied to generators")

            func_name = getattr(
                unwrapped,
                "__qualname__",
                getattr(func, "__name__", safe_repr(func)),
            )

            _name = name or func_name
            _label = label or func_name

            # conform our label for sanity
            _label = re.sub(r"[\W_]+", "_", _label.lower())

            _attributes = attributes or {}
            _attributes["code.function"] = func_name
            with contextlib.suppress(Exception):
                _attributes["code.lineno"] = unwrapped.__code__.co_firstlineno
            with contextlib.suppress(Exception):
                _attributes.update(
                    get_filepath_attribute(
                        inspect.getsourcefile(unwrapped),  # type: ignore [arg-type]
                    ),
                )

            return Task(
                tracer=self._get_tracer(),
                name=_name,
                attributes=_attributes,
                func=t.cast("t.Callable[P, R]", func),
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
                label=_label,
            )

        return make_task

    def task_span(
        self,
        name: str,
        *,
        label: str | None = None,
        params: AnyDict | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> TaskSpan[t.Any]:
        """
        Create a task span without an explicit associated function.

        This is useful for creating tasks on the fly without having to
        define a function.

        Example:
            ```
            async with dreadnode.task_span("my_task") as task:
                # do some work here
                pass
            ```
        Args:
            name: The name of the task.
            label: The label of the task - useful for filtering in the UI.
            params: A dictionary of parameters to attach to the task span.
            tags: A list of tags to attach to the task span.
            **attributes: A dictionary of attributes to attach to the task span.

        Returns:
            A TaskSpan object.
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Task spans must be created within a run")

        label = label or re.sub(r"[\W_]+", "_", name.lower())
        return TaskSpan(
            name=name,
            label=label,
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
        """
        Make a scorer from a callable function.

        This is useful when you want to change the name of the scorer
        or add additional attributes to it.

        Example:
            ```
            @dreadnode.scorer(name="my_scorer")
            async def my_scorer(x: int) -> float:
                return x * 2

            @dreadnode.task(scorers=[my_scorer])
            async def my_task(x: int) -> int:
                return x * 2

            await my_task(2)
            ```

        Args:
            name: The name of the scorer.
            tags: A list of tags to attach to the scorer.
            **attributes: A dictionary of attributes to attach to the scorer.

        Returns:
            A new Scorer object.
        """

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
        """
        Create a new run.

        Runs are the main way to track work in Dreadnode. They are
        associated with a specific project and can have parameters,
        inputs, and outputs logged to them.

        You cannot create runs inside other runs.

        Example:
            ```
            with dreadnode.run("my_run"):
                # do some work here
                pass
            ```

        Args:
            name: The name of the run. If not provided, a random name will be generated.
            tags: A list of tags to attach to the run.
            params: A dictionary of parameters to attach to the run.
            project: The project name to associate the run with. If not provided,
                the project passed to `configure()` will be used, or the
                run will be associated with a default project.
            **attributes: Additional attributes to attach to the run span.
        """
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
            file_system=self._fs,
            prefix_path=self._fs_prefix,
        )

    @handle_internal_errors()
    def push_update(self) -> None:
        """
        Push any pending metric or parameter data to the server.

        This is useful for ensuring that the UI is up to date with the
        latest data. Otherwise, all data for the run will be pushed
        automatically when the run is closed.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_params(...)
                dreadnode.log_metric(...)
                dreadnode.push_update()
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("Run updates must be pushed within a run")

        run.push_update()

    @handle_internal_errors()
    def log_param(
        self,
        key: str,
        value: JsonValue,
        *,
        to: ToObject = "task-or-run",
    ) -> None:
        """
        Log a single parameter to the current task or run.

        Parameters are key-value pairs that are associated with the task or run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run") as run:
                run.log_param("param_name", "param_value")
            ```

        Args:
            key: The name of the parameter.
            value: The value of the parameter.
            to: The target object to log the parameter to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the parameter will be logged
                to the current task or run, whichever is the nearest ancestor.
        """
        self.log_params(to=to, **{key: value})

    @handle_internal_errors()
    def log_params(self, to: ToObject = "run", **params: JsonValue) -> None:
        """
        Log multiple parameters to the current task or run.

        Parameters are key-value pairs that are associated with the task or run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run") as run:
                run.log_params(
                    param1="value1",
                    param2="value2"
                )
            ```

        Args:
            to: The target object to log the parameters to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the parameters will be logged
                to the current task or run, whichever is the nearest ancestor.
            **params: The parameters to log. Each parameter is a key-value pair.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            raise RuntimeError("log_params() must be called within a run")

        target.log_params(**params)

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
        to: ToObject = "task-or-run",
    ) -> None:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run") as run:
                run.log_metric("metric_name", 42.0)
            ```

        Args:
            key: The name of the metric.
            value: The value of the metric.
            step: The step of the metric.
            origin: The origin of the metric - can be provided any object which was logged
                as an input or output anywhere in the run.
            timestamp: The timestamp of the metric - defaults to the current time.
            mode: The aggregation mode to use for the metric. Helpful when you want to let
                the library take care of translating your raw values into better representations.
                - direct: do not modify the value at all (default)
                - min: the lowest observed value reported for this metric
                - max: the highest observed value reported for this metric
                - avg: the average of all reported values for this metric
                - sum: the cumulative sum of all reported values for this metric
                - count: increment every time this metric is logged - disregard value
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.
        """

    @t.overload
    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
        to: ToObject = "task-or-run",
    ) -> None:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run") as run:
                run.log_metric("metric_name", 42.0)
            ```

        Args:
            key: The name of the metric.
            value: The metric object.
            origin: The origin of the metric - can be provided any object which was logged
                as an input or output anywhere in the run.
            mode: The aggregation mode to use for the metric. Helpful when you want to let
                the library take care of translating your raw values into better representations.
                - min: always report the lowest ovbserved value for this metric
                - max: always report the highest observed value for this metric
                - avg: report the average of all values for this metric
                - sum: report a rolling sum of all values for this metric
                - count: report the number of times this metric has been logged
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.
        """

    @handle_internal_errors()
    def log_metric(
        self,
        key: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        to: ToObject = "task-or-run",
    ) -> None:
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            raise RuntimeError("log_metric() must be called within a run")

        metric = (
            value
            if isinstance(value, Metric)
            else Metric(float(value), step, timestamp or datetime.now(timezone.utc))
        )
        target.log_metric(key, metric, origin=origin, mode=mode)

    @handle_internal_errors()
    def log_artifact(
        self,
        local_uri: str | Path,
    ) -> None:
        """
        Log a file or directory artifact to the current run.

        This method uploads a local file or directory to the artifact storage associated with the run.

        Examples:
            Log a single file:
            ```
            with dreadnode.run("my_run") as run:
                # Save a file
                with open("results.json", "w") as f:
                    json.dump(results, f)

                # Log it as an artifact
                run.log_artifact("results.json")
            ```

            Log a directory:
            ```
            with dreadnode.run("my_run") as run:
                # Create a directory with model files
                os.makedirs("model_output", exist_ok=True)
                save_model("model_output/model.pkl")
                save_config("model_output/config.yaml")

                # Log the entire directory as an artifact
                run.log_artifact("model_output")
            ```

        Args:
            local_uri: The local path to the file to upload.
            to: The target object to log the artifact to. Only "run" is supported.
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("log_artifact() must be called within a run")

        run.log_artifact(local_uri=local_uri)

    @handle_internal_errors()
    def log_input(
        self,
        name: str,
        value: JsonValue,
        *,
        label: str | None = None,
        to: ToObject = "task-or-run",
        **attributes: t.Any,
    ) -> None:
        """
        Log a single input to the current task or run.

        Inputs can be any runtime object, which are serialized, stored, and tracked
        in the Dreadnode UI.

        Example:
            ```
            @dreadnode.task
            async def my_task(x: int) -> int:
                dreadnode.log_input("input_name", x)
                return x * 2

            with dreadnode.run("my_run"):
                dreadnode.log_input("input_name", some_dataframe)

                await my_task(2)
            ```
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            raise RuntimeError("log_inputs() must be called within a run")

        target.log_input(name, value, label=label, **attributes)

    @handle_internal_errors()
    def log_inputs(
        self,
        to: ToObject = "task-or-run",
        **inputs: JsonValue,
    ) -> None:
        """
        Log multiple inputs to the current task or run.

        See `log_input()` for more details.
        """
        for name, value in inputs.items():
            self.log_input(name, value, to=to)

    @handle_internal_errors()
    def log_output(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        to: ToObject = "task-or-run",
        **attributes: JsonValue,
    ) -> None:
        """
        Log a single output to the current task or run.

        Outputs can be any runtime object, which are serialized, stored, and tracked
        in the Dreadnode UI.

        Example:
            ```
            @dreadnode.task
            async def my_task(x: int) -> int:
                result = x * 2
                dreadnode.log_output("result", x * 2)
                return result

            with dreadnode.run("my_run"):
                await my_task(2)

                dreadnode.log_output("other", 123)
            ```
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            raise RuntimeError(
                "log_output() must be called within a run or a task",
            )

        target.log_output(name, value, label=label, **attributes)

    @handle_internal_errors()
    def log_outputs(
        self,
        to: ToObject = "task-or-run",
        **outputs: JsonValue,
    ) -> None:
        """
        Log multiple outputs to the current task or run.

        See `log_output()` for more details.
        """
        for name, value in outputs.items():
            self.log_output(name, value, to=to)

    @handle_internal_errors()
    def link_objects(self, origin: t.Any, link: t.Any, **attributes: JsonValue) -> None:
        """
        Associate two runtime objects with each other.

        This is useful for linking any two objects which are related to
        each other, such as a model and its training data, or an input
        prompt and the resulting output.

        Example:
            ```
            with dreadnode.run("my_run") as run:
                model = SomeModel()
                data = SomeData()

                run.link_objects(model, data)
            ```

        Args:
            origin: The origin object to link from.
            link: The linked object to link to.
            **attributes: Additional attributes to attach to the link.
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("link() must be called within a run")

        origin_hash = run.log_object(origin)
        link_hash = run.log_object(link)
        run.link_objects(origin_hash, link_hash, **attributes)


DEFAULT_INSTANCE = Dreadnode()
