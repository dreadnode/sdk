import contextlib
import inspect
import os
import random
import socket
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

import coolname  # type: ignore [import-untyped]
import logfire
from fsspec.implementations.local import (  # type: ignore [import-untyped]
    LocalFileSystem,
)
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from logfire._internal.stack_info import get_filepath_attribute, warn_at_user_stacklevel
from logfire._internal.utils import safe_repr
from opentelemetry import propagate
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
from dreadnode.metric import (
    Metric,
    MetricAggMode,
    MetricDict,
    Scorer,
    ScorerCallable,
    T,
)
from dreadnode.task import P, R, Task
from dreadnode.tracing.exporters import (
    FileExportConfig,
    FileMetricReader,
    FileSpanExporter,
)
from dreadnode.tracing.span import (
    RunContext,
    RunSpan,
    Span,
    TaskSpan,
    current_run_span,
    current_task_span,
)
from dreadnode.types import (
    INHERITED,
    AnyDict,
    Inherited,
    JsonValue,
)
from dreadnode.util import clean_str, handle_internal_errors, logger
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
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
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

    @staticmethod
    def _resolve_endpoint(endpoint: str | None) -> str | None:
        """Automatically resolve endpoints based on environment

        Args:
            endpoint: The endpoint URL to resolve.

        Returns:
            str: The resolved endpoint URL.

        Raises:
            ValueError: If the endpoint URL is invalid.
        """
        if not endpoint:
            return None
        parsed = urlparse(endpoint)

        # If it's a real domain (has dots), use as-is
        if not parsed.hostname:
            raise ValueError(f"Invalid endpoint URL: {endpoint}")

        if "." in parsed.hostname:
            return endpoint

        # If it's a service name, try to resolve it
        if Dreadnode._is_docker_service_name(parsed.hostname):
            return Dreadnode._resolve_docker_service(endpoint, parsed)

        return endpoint

    @staticmethod
    def _is_docker_service_name(hostname: str) -> bool:
        """Check if this looks like a Docker service name

        Args:
            hostname: The hostname to check.

        Returns:
            bool: True if the hostname looks like a Docker service name, False otherwise.
        """
        return bool(hostname and "." not in hostname and hostname != "localhost")

    @staticmethod
    def _resolve_docker_service(original_endpoint: str, parsed: ParseResult) -> str:
        """Try different resolution strategies for Docker services

        Args:
            original_endpoint: The original endpoint URL.
            parsed: The parsed URL object.

        Returns:
            str: The resolved endpoint URL.

        Raises:
            RuntimeError: If no valid endpoint is found.
        """
        strategies = [
            original_endpoint,  # Try original first (works if running in same network)
            f"{parsed.scheme}://localhost:{parsed.port}",  # Try localhost
            f"{parsed.scheme}://host.docker.internal:{parsed.port}",  # Docker Desktop
            f"{parsed.scheme}://172.17.0.1:{parsed.port}",  # Docker bridge IP
        ]

        for endpoint in strategies:
            if Dreadnode._test_connection(endpoint):
                logger.warning(
                    f"Resolved Docker service for s3 connection '{parsed.hostname}' to '{endpoint}'."
                )
                return str(endpoint)

        # If nothing works, return original and let it fail with a helpful error
        raise RuntimeError(f"Failed to connect to the Dreadnode Artifact storage at {endpoint}.")

    @staticmethod
    def _test_connection(endpoint: str) -> bool:
        """Quick connectivity test

        Args:
            endpoint: The endpoint URL to test.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            parsed = urlparse(endpoint)
            socket.create_connection((parsed.hostname, parsed.port or 443), timeout=1)
        except Exception:  # noqa: BLE001
            return False

        return True

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
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
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
            project: The default project name to associate all runs with.
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

        self.server = self.server or (DEFAULT_SERVER_URL if self.token else None)
        if not (self.server or self.token or self.local_dir):
            warn_at_user_stacklevel(
                "Your current configuration won't persist run data anywhere. "
                "Use `dreadnode.init(server=..., token=...)`, `dreadnode.init(local_dir=...)`, "
                f"or use environment variables ({ENV_SERVER_URL}, {ENV_API_TOKEN}, {ENV_LOCAL_DIR}).",
                category=DreadnodeConfigWarning,
            )

        if self.local_dir:
            config = FileExportConfig(
                base_path=self.local_dir,
                prefix=self.project + "-" if self.project else "",
            )
            span_processors.append(BatchSpanProcessor(FileSpanExporter(config)))
            metric_readers.append(FileMetricReader(config))

        if self.token and self.server:
            try:
                parsed_url = urlparse(self.server)
                if not parsed_url.scheme:
                    netloc = parsed_url.path.split("/")[0]
                    path = "/".join(parsed_url.path.split("/")[1:])
                    parsed_new = parsed_url._replace(
                        scheme="https", netloc=netloc, path=f"/{path}" if path else ""
                    )
                    self.server = urlunparse(parsed_new)

                self._api = ApiClient(self.server, self.token)

                self._api.list_projects()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to the Dreadnode server: {e}",
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
            resolved_endpoint = self._resolve_endpoint(credentials.endpoint)
            self._fs = S3FileSystem(
                key=credentials.access_key_id,
                secret=credentials.secret_access_key,
                token=credentials.session_token,
                client_kwargs={
                    "endpoint_url": resolved_endpoint,
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
            distributed_tracing=False,
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
        attributes: AnyDict | None = None,
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
            attributes: A dictionary of attributes to attach to the span.

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
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
    ) -> TaskDecorator: ...

    @t.overload
    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]],
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
    ) -> ScoredTaskDecorator[R]: ...

    def task(
        self,
        *,
        scorers: t.Sequence[Scorer[t.Any] | ScorerCallable[t.Any]] | None = None,
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
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
            log_inputs: Log all, or specific, incoming arguments to the function as inputs.
            log_output: Log the result of the function as an output.
            log_execution_metrics: Log execution metrics for the task, such as success rate and run count.
            tags: A list of tags to attach to the task span.
            attributes: A dictionary of attributes to attach to the task span.

        Returns:
            A new Task object.
        """

        def make_task(
            func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
        ) -> Task[P, R]:
            if isinstance(func, Task):
                return func.with_(
                    name=name,
                    label=label,
                    log_inputs=log_inputs,
                    log_output=log_output,
                    log_execution_metrics=log_execution_metrics,
                    tags=tags,
                    attributes=attributes,
                    append=True,
                )

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
            _label = clean_str(_label)

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
                    scorer if isinstance(scorer, Scorer) else Scorer.from_callable(scorer)
                    for scorer in scorers or []
                ],
                tags=list(tags or []),
                log_inputs=log_inputs,
                log_output=log_output,
                log_execution_metrics=log_execution_metrics,
                label=_label,
            )

        return make_task

    def task_span(
        self,
        name: str,
        *,
        label: str | None = None,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
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
            tags: A list of tags to attach to the task span.
            attributes: A dictionary of attributes to attach to the task span.

        Returns:
            A TaskSpan object.
        """
        run = current_run_span.get()
        label = clean_str(label or name)

        return TaskSpan(
            name=name,
            label=label,
            attributes=attributes,
            tags=tags,
            run_id=run.run_id if run else "",
            tracer=self._get_tracer(),
        )

    def scorer(
        self,
        *,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
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
            attributes: A dictionary of attributes to attach to the scorer.

        Returns:
            A new Scorer object.
        """

        def make_scorer(func: ScorerCallable[T]) -> Scorer[T]:
            return Scorer.from_callable(
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
        autolog: bool = True,
        attributes: AnyDict | None = None,
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
            autolog: Whether to automatically log task inputs, outputs, and execution metrics if otherwise unspecified.
            attributes: Additional attributes to attach to the run span.

        Returns:
            A RunSpan object that can be used as a context manager.
            The run will automatically be completed when the context manager exits.
        """
        if not self._initialized:
            self.initialize()

        if name is None:
            name = f"{coolname.generate_slug(2)}-{random.randint(100, 999)}"  # noqa: S311 # nosec

        return RunSpan(
            name=name,
            project=project or self.project or "default",
            attributes=attributes,
            tracer=self._get_tracer(),
            params=params,
            tags=tags,
            file_system=self._fs,
            prefix_path=self._fs_prefix,
            autolog=autolog,
        )

    def get_run_context(self) -> RunContext:
        """
        Capture the current run context for transfer to another host, thread, or process.

        Use `continue_run()` to continue the run anywhere else.

        Returns:
            RunContext containing run state and trace propagation headers.

        Raises:
            RuntimeError: If called outside of an active run.
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("get_run_context() must be called within a run")

        # Capture OpenTelemetry trace context
        trace_context: dict[str, str] = {}
        propagate.inject(trace_context)

        return {
            "run_id": run.run_id,
            "run_name": run.name,
            "project": run.project,
            "trace_context": trace_context,
        }

    def continue_run(self, run_context: RunContext) -> RunSpan:
        """
        Continue a run from captured context on a remote host.

        Args:
            run_context: The RunContext captured from get_run_context().

        Returns:
            A RunSpan object that can be used as a context manager.
        """
        if not self._initialized:
            self.initialize()

        return RunSpan.from_context(
            context=run_context,
            tracer=self._get_tracer(),
            file_system=self._fs,
            prefix_path=self._fs_prefix,
        )

    def tag(self, *tag: str, to: ToObject = "task-or-run") -> None:
        """
        Add one or many tags to the current task or run.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.tag("my_tag")
            ```

        Args:
            tag: The tag to attach to the task or run.
            to: The target object to log the tag to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the tag will be logged
                to the current task or run, whichever is the nearest ancestor.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "tag() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        target.add_tags(tag)

    @handle_internal_errors()
    def push_update(self) -> None:
        """
        Push any pending run data to the server before run completion.

        This is useful for ensuring that the UI is up to date with the
        latest data. Data is automatically pushed periodically, but
        you can call this method to force a push.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_params(...)
                dreadnode.log_metric(...)
                dreadnode.push_update()

                # do more work
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "push_update() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.push_update(force=True)

    @handle_internal_errors()
    def log_param(
        self,
        key: str,
        value: JsonValue,
    ) -> None:
        """
        Log a single parameter to the current task or run.

        Parameters are key-value pairs that are associated with the task or run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_param("param_name", "param_value")
            ```

        Args:
            key: The name of the parameter.
            value: The value of the parameter.
        """
        self.log_params(**{key: value})

    @handle_internal_errors()
    def log_params(self, **params: JsonValue) -> None:
        """
        Log multiple parameters to the current task or run.

        Parameters are key-value pairs that are associated with the task or run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_params(
                    param1="value1",
                    param2="value2"
                )
            ```

        Args:
            **params: The parameters to log. Each parameter is a key-value pair.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "log_params() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.log_params(**params)

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
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("metric_name", 42.0)
            ```

        Args:
            name: The name of the metric.
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
            attributes: A dictionary of additional attributes to attach to the metric.
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            The logged metric object.
        """

    @t.overload
    def log_metric(
        self,
        name: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("metric_name", 42.0)
            ```

        Args:
            name: The name of the metric.
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

        Returns:
            The logged metric object.
        """

    @handle_internal_errors()
    def log_metric(
        self,
        name: str,
        value: float | bool | Metric,
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Examples:
            With a raw value:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("accuracy", 0.95, step=10)
                dreadnode.log_metric("loss", 0.05, step=10, mode="min")
            ```

            With a Metric object:
            ```
            with dreadnode.run("my_run"):
                metric = Metric(0.95, step=10, timestamp=datetime.now(timezone.utc))
                dreadnode.log_metric("accuracy", metric)
            ```

        Args:
            name: The name of the metric.
            value: The value of the metric, either as a raw float/bool or a Metric object.
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
            attributes: A dictionary of additional attributes to attach to the metric.
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            The logged metric object.
        """
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value),
                step,
                timestamp or datetime.now(timezone.utc),
                attributes or {},
            )
        )

        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "log_metric() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return metric

        return target.log_metric(name, metric, origin=origin, mode=mode)

    @t.overload
    def log_metrics(
        self,
        metrics: dict[str, float | bool],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics from a dictionary of name/value pairs.

        Examples:
            ```
            dreadnode.log_metrics(
                {
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "f1_score": 0.92
                },
                step=10
            )
            ```

        Args:
            metrics: Dictionary of name/value pairs to log as metrics.
            step: Step value for all metrics.
            timestamp: Timestamp for all metrics.
            mode: Aggregation mode for all metrics.
            attributes: Attributes for all metrics.
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

    @t.overload
    def log_metrics(
        self,
        metrics: list[MetricDict],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics from a list of metric configurations.

        Example:
            ```
            dreadnode.log_metrics(
                [
                    {"name": "accuracy", "value": 0.95},
                    {"name": "loss", "value": 0.05, "mode": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: List of metric configurations to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            mode: Default aggregation mode for metrics if not supplied.
            attributes: Default attributes for metrics if not supplied.
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

    @handle_internal_errors()
    def log_metrics(
        self,
        metrics: dict[str, float | bool] | list[MetricDict],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics to the current task or run.

        Examples:
            Log metrics from a dictionary:
            ```
            dreadnode.log_metrics(
                {
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "f1_score": 0.92
                },
                step=10
            )
            ```

            Log metrics from a list of MetricDicts:
            ```
            dreadnode.log_metrics(
                [
                    {"name": "accuracy", "value": 0.95},
                    {"name": "loss", "value": 0.05, "mode": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: Either a dictionary of name/value pairs or a list of MetricDicts to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            mode: Default aggregation mode for metrics if not supplied.
            attributes: Default attributes for metrics if not supplied.
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "log_metrics() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return []

        logged_metrics: list[Metric] = []

        # Dictionary of name/value pairs
        if isinstance(metrics, dict):
            logged_metrics = [
                target.log_metric(
                    name,
                    value,
                    step=step,
                    timestamp=timestamp,
                    mode=mode,
                    attributes=attributes,
                )
                for name, value in metrics.items()
            ]

        # List of MetricDicts
        else:
            logged_metrics = [
                target.log_metric(
                    metric["name"],
                    metric["value"],
                    step=metric.get("step", step),
                    timestamp=metric.get("timestamp", timestamp),
                    mode=metric.get("mode", mode),
                    attributes=metric.get("attributes", attributes) or {},
                )
                for metric in metrics
            ]

        return logged_metrics

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
            with dreadnode.run("my_run"):
                # Save a file
                with open("results.json", "w") as f:
                    json.dump(results, f)

                # Log it as an artifact
                dreadnode.log_artifact("results.json")
            ```

            Log a directory:
            ```
            with dreadnode.run("my_run"):
                # Create a directory with model files
                os.makedirs("model_output", exist_ok=True)
                save_model("model_output/model.pkl")
                save_config("model_output/config.yaml")

                # Log the entire directory as an artifact
                dreadnode.log_artifact("model_output")
            ```

        Args:
            local_uri: The local path to the file to upload.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "log_artifact() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.log_artifact(local_uri=local_uri)

    @handle_internal_errors()
    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        to: ToObject = "task-or-run",
        attributes: AnyDict | None = None,
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
            warn_at_user_stacklevel(
                "log_input() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        target.log_input(name, value, label=label, attributes=attributes)

    @handle_internal_errors()
    def log_inputs(
        self,
        to: ToObject = "task-or-run",
        **inputs: t.Any,
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
        attributes: AnyDict | None = None,
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

        Args:
            name: The name of the output.
            value: The value of the output.
            label: An optional label for the output, useful for filtering in the UI.
            to: The target object to log the output to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the output will be logged
                to the current task or run, whichever is the nearest ancestor.
            attributes: Additional attributes to attach to the output.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "log_output() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        target.log_output(name, value, label=label, attributes=attributes)

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
    def link_objects(
        self,
        origin: t.Any,
        link: t.Any,
        attributes: AnyDict | None = None,
    ) -> None:
        """
        Associate two runtime objects with each other.

        This is useful for linking any two objects which are related to
        each other, such as a model and its training data, or an input
        prompt and the resulting output.

        Example:
            ```
            with dreadnode.run("my_run"):
                model = SomeModel()
                data = SomeData()

                dreadnode.link_objects(model, data)
            ```

        Args:
            origin: The origin object to link from.
            link: The linked object to link to.
            attributes: Additional attributes to attach to the link.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "link_objects() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        origin_hash = run.log_object(origin)
        link_hash = run.log_object(link)
        run.link_objects(origin_hash, link_hash, attributes=attributes)


DEFAULT_INSTANCE = Dreadnode()
