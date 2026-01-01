import contextlib
import random
import sys
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import coolname
import logfire
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from opentelemetry import propagate
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from ulid import ULID

from dreadnode.core.api.client import ApiClient
from dreadnode.core.api.models import Organization, Project, Workspace
from dreadnode.core.api.session import Session, UserConfig
from dreadnode.core.exceptions import (
    DreadnodeUsageWarning,
    handle_internal_errors,
    warn_at_user_stacklevel,
)
from dreadnode.core.log import configure_logging
from dreadnode.core.metric import (
    Metric,
    MetricAggMode,
    MetricDict,
    MetricsLike,
)
from dreadnode.core.packaging.package import (
    BuildResult,
    Package,
    PackageType,
    PullResult,
    PushResult,
)
from dreadnode.core.scorer import ScorersLike
from dreadnode.core.storage import Storage
from dreadnode.core.task import P, R, ScoredTaskDecorator, Task, TaskDecorator
from dreadnode.core.tracing.exporter import CustomOTLPSpanExporter
from dreadnode.core.tracing.exporters import (
    FileMetricReader,
    FileSpanExporter,
    TraceExportConfig,
)
from dreadnode.core.tracing.span import (
    RunContext,
    RunSpan,
    Span,
    TaskSpan,
    current_run_span,
    current_task_span,
)
from dreadnode.core.types.common import (
    INHERITED,
    AnyDict,
    Inherited,
    JsonValue,
)
from dreadnode.core.util import (
    clean_str,
)
from dreadnode.core.settings import (
    settings,
)
from dreadnode.version import VERSION

if t.TYPE_CHECKING:
    from opentelemetry.sdk.metrics.export import MetricReader
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Tracer

    from dreadnode.core.tracing.constants import SpanType


ToObject = t.Literal["task-or-run", "run"]


configure_logging()


@dataclass
class Dreadnode:
    """
    The core Dreadnode SDK class.

    A default instance of this class is created and can be used directly with `dreadnode.*`.

    Otherwise, you can create your own instance and configure it with `configure()`.
    """

    def __init__(
        self,
        *,
        server: str | None = None,
        token: str | None = None,
        organization: str | ULID | None = None,
        workspace: str | ULID | None = None,
        project: str | None = None,
        cache: str | None = None,
        storage: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | bool = False,
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
        otel_scope: str = "dreadnode",
    ) -> None:
        self.server = server
        self.token = token
        self.organization = organization
        self.workspace = workspace
        self.project = project
        self.cache = cache
        self.storage = storage
        self.service_name = service_name
        self.service_version = service_version
        self.console = console
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self._api: ApiClient | None = None
        self._session: Session | None = None
        self._logfire = logfire.DEFAULT_LOGFIRE_INSTANCE
        self._logfire.config.ignore_no_config = True

        self._initialized = False

        self._version = VERSION

    @property
    def resolved_organization(self) -> Organization:
        """Get the resolved organization from the session."""
        if self._session is None:
            raise RuntimeError("Call configure() first")
        return self._session.organization

    @property
    def resolved_workspace(self) -> Workspace:
        """Get the resolved workspace from the session."""
        if self._session is None:
            raise RuntimeError("Call configure() first")
        return self._session.workspace

    @property
    def resolved_project(self) -> Project:
        """Get the resolved project from the session."""
        if self._session is None:
            raise RuntimeError("Call configure() first")
        return self._session.project

    @property
    def api(self) -> ApiClient:
        """Get the API client."""
        if self._api is None:
            raise RuntimeError("Call configure() first")
        return self._api

    def configure(
        self,
        *,
        server: str | None = None,
        token: str | None = None,
        profile: str | None = None,
        organization: str | ULID | None = None,
        workspace: str | ULID | None = None,
        project: str | None = None,
        cache: str | None = None,
        storage_provider: t.Literal["s3", "r2", "minio", "local"] | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | bool | None = None,
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
        otel_scope: str = "dreadnode",
    ) -> None:
        """
        Configure the Dreadnode SDK and call `initialize()`.

        This method should always be called before using the SDK.

        For local-only operations (packaging without a server), use:
            dn.configure(server="local")

        If `server` and `token` are not provided, the SDK will look for them
        in the following order:

        1. Environment variables:
           - `DREADNODE_SERVER_URL` or `DREADNODE_SERVER`
           - `DREADNODE_API_TOKEN` or `DREADNODE_API_KEY`
           - `DREADNODE_ORGANIZATION`
           - `DREADNODE_WORKSPACE`
           - `DREADNODE_PROJECT`

        2. Dreadnode profile (from `dreadnode login`)
           - Uses `profile` parameter if provided
           - Falls back to `DREADNODE_PROFILE` environment variable
           - Defaults to active profile

        Args:
            server: The Dreadnode server URL. Use "local" for local-only mode.
            token: The Dreadnode API token.
            profile: The Dreadnode profile name to use (only used if env vars are not set).
            organization: The default organization name or ID to use.
            workspace: The default workspace name or ID to use.
            project: The default project name to associate all runs with. This can also be in the format `org/workspace/project` using the keys.
            cache: The local cache directory to use.
            storage_provider: The storage provider to use.
            service_name: The service name to use for OpenTelemetry.
            service_version: The service version to use for OpenTelemetry.
            console: Log span information to the console.
            send_to_logfire: Send data to Logfire.
            otel_scope: The OpenTelemetry scope name.
        """

        self._initialized = False

        # Skip during testing
        if "pytest" in sys.modules:
            self._initialized = True
            return

        # Determine configuration source and active profile for logging
        config_source = "explicit parameters"
        active_profile = None
        user_config: UserConfig | None = None

        if not server or not token:
            env_server = settings.server_url
            env_token = settings.token

            if env_server or env_token:
                config_source = "environment vars"
            else:
                # Fall back to profile
                config_source = "profile"
                with contextlib.suppress(Exception):
                    user_config = UserConfig.read()
                    profile_name = profile or settings.profile
                    active_profile = profile_name or user_config.active_profile_name

                    if active_profile:
                        config_source = f"profile: {active_profile}"

        # Read user config if we haven't already
        if user_config is None:
            with contextlib.suppress(Exception):
                user_config = UserConfig.read()

        self.server = (
            server
            or settings.server_url
            or (user_config.get_profile_server(profile) if user_config else None)
        )
        self.token = (
            token
            or settings.token
            or (user_config.get_profile_api_key(profile) if user_config else None)
        )

        if (cache is None and settings.cache_dir) or cache is None:
            self.cache = settings.cache_dir
        else:
            self.cache = Path(cache)

        _org, _workspace, _project = Session.extract_project_components(project)
        self.organization = _org or organization or settings.organization
        self.workspace = _workspace or workspace or settings.workspace
        self.project = _project or project or settings.project

        self.service_name = service_name
        self.service_version = service_version
        self.console = (
            console
            if console is not None
            else settings.console
            in [
                "true",
                "1",
                "yes",
            ]
        )
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope
        self.storage_provider = storage_provider

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the Dreadnode SDK.

        This method is called automatically when `configure()` is called.
        """

        if self._initialized:
            return

        span_processors: list[SpanProcessor] = []
        metric_readers: list[MetricReader] = []

        # Local mode: skip server connection, just use local storage
        is_local_mode = self.server == "local"

        if self.token and self.server and not is_local_mode:
            try:
                parsed_url = urlparse(self.server)
                if not parsed_url.scheme:
                    netloc = parsed_url.path.split("/")[0]
                    path = "/".join(parsed_url.path.split("/")[1:])
                    parsed_new = parsed_url._replace(
                        scheme="https", netloc=netloc, path=f"/{path}" if path else ""
                    )
                    self.server = urlunparse(parsed_new)

                self._api = ApiClient(self.server, api_key=self.token)

                # Create and resolve session for RBAC
                self._session = Session(
                    api=self._api,
                    organization=self.organization,
                    workspace=self.workspace,
                    project=self.project,
                ).resolve()

                # Update project to resolved ID for use in runs/traces
                self.project = self._session.project_id

            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to {self.server}: {e}",
                ) from e

            headers = {"X-Api-Key": self.token}
            endpoint = "/api/otel/traces"
            span_processors.append(
                BatchSpanProcessor(
                    RemovePendingSpansExporter(
                        CustomOTLPSpanExporter(
                            endpoint=urljoin(self.server, endpoint),
                            headers=headers,
                            compression=Compression.Gzip,
                        ),
                    ),
                ),
            )

        # Initialize storage (works with or without session)
        self.storage = Storage(
            session=self._session,
            cache=self.cache,
            provider=self.storage_provider,
        )

        # Add file exporters for local trace storage
        if self.cache is not None:
            from ulid import ULID

            run_id = str(ULID())
            trace_config = TraceExportConfig(
                storage=self.storage,
                run_id=run_id,
            )
            self._trace_config = trace_config
            span_processors.append(BatchSpanProcessor(FileSpanExporter(trace_config)))
            metric_readers.append(FileMetricReader(trace_config))

        self._logfire = logfire.configure(
            local=False,
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

    def get_current_run(self) -> RunSpan | None:
        return current_run_span.get()

    def get_current_task(self) -> TaskSpan[t.Any] | None:
        return current_task_span.get()

    def get_tracer(self, *, is_span_tracer: bool = True) -> "Tracer":
        """
        Get an OpenTelemetry Tracer instance.

        Args:
            is_span_tracer: Whether the tracer is for creating spans.

        Returns:
            An OpenTelemetry Tracer.
        """
        return self._logfire._tracer_provider.get_tracer(  # noqa: SLF001
            self.otel_scope,
            self._version,
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
            tracer=self.get_tracer(),
            tags=tags,
        )

    def task(
        self,
        func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R] | None = None,
        /,
        *,
        scorers: ScorersLike[t.Any] | None = None,
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
    ) -> TaskDecorator | ScoredTaskDecorator[R] | Task[P, R]:
        """Create a new task from a function. See `task()` for details."""
        from dreadnode.core.task import task as task_factory

        return task_factory(
            func,
            tracer=self.get_tracer(),
            scorers=scorers,
            name=name,
            label=label,
            log_inputs=log_inputs,
            log_output=log_output,
            log_execution_metrics=log_execution_metrics,
            tags=tags,
            attributes=attributes,
            entrypoint=entrypoint,
        )

    def task_span(
        self,
        name: str,
        *,
        type: "SpanType" = "task",
        label: str | None = None,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        _tracer: "Tracer | None" = None,
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
            type: The type of span (task, evaluation, etc.).
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
            type=type,
            label=label,
            attributes=attributes,
            tags=tags,
            run_id=run.run_id if run else "",
            tracer=_tracer or self.get_tracer(),
        )

    def scorer(
        self,
        func=None,
        *,
        name: str | None = None,
        assert_: bool = False,
        attributes: AnyDict | None = None,
    ):
        """Create a scorer decorator. See `scorer()` for details."""
        from dreadnode.core.scorer import scorer as scorer_factory

        return scorer_factory(func, name=name, assert_=assert_, attributes=attributes)

    def agent(
        self,
        func=None,
        /,
        *,
        name: str | None = None,
        model: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        label: str | None = None,
        max_steps: int = 10,
        tools: list[t.Any] | None = None,
        tool_mode: str = "auto",
        stop_conditions: list[t.Any] | None = None,
        hooks: list[t.Any] | None = None,
    ):
        """Decorator to create an Agent from a function. See `agent()` for details."""
        from dreadnode.core.agents.agent import agent as agent_factory

        return agent_factory(
            func,
            name=name,
            model=model,
            description=description,
            tags=tags,
            label=label,
            max_steps=max_steps,
            tools=tools,
            tool_mode=tool_mode,
            stop_conditions=stop_conditions,
            hooks=hooks,
        )

    def evaluation(
        self,
        func=None,
        /,
        *,
        dataset: t.Any | None = None,
        dataset_file: str | None = None,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        concurrency: int = 1,
        iterations: int = 1,
        max_errors: int | None = None,
        max_consecutive_errors: int = 10,
        dataset_input_mapping: list[str] | dict[str, str] | None = None,
        parameters: dict[str, list[t.Any]] | None = None,
        scorers: "ScorersLike[t.Any] | None" = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
    ):
        """Decorator to create an Evaluation from a function. See `evaluation()` for details."""
        from dreadnode.core.evaluations import evaluation as evaluation_factory

        return evaluation_factory(
            func,
            dataset=dataset,
            dataset_file=dataset_file,
            name=name,
            description=description,
            tags=tags,
            concurrency=concurrency,
            iterations=iterations,
            max_errors=max_errors,
            max_consecutive_errors=max_consecutive_errors,
            dataset_input_mapping=dataset_input_mapping,
            parameters=parameters,
            scorers=scorers,
            assert_scores=assert_scores,
        )

    def study(
        self,
        func=None,
        /,
        *,
        name: str | None = None,
        search_strategy: t.Any | None = None,
        dataset: t.Any | None = None,
        dataset_file: str | None = None,
        objectives: "ScorersLike[t.Any] | None" = None,
        directions: list[str] | None = None,
        constraints: "ScorersLike[t.Any] | None" = None,
        max_trials: int = 100,
        concurrency: int = 1,
        stop_conditions: list[t.Any] | None = None,
    ):
        """Decorator to create a Study from a task factory. See `study()` for details."""
        from dreadnode.core.optimization import study as study_factory

        return study_factory(
            func,
            name=name,
            search_strategy=search_strategy,
            dataset=dataset,
            dataset_file=dataset_file,
            objectives=objectives,
            directions=directions,
            constraints=constraints,
            max_trials=max_trials,
            concurrency=concurrency,
            stop_conditions=stop_conditions,
        )

    def run(
        self,
        name: str | None = None,
        *,
        tags: t.Sequence[str] | None = None,
        params: AnyDict | None = None,
        project: str | None = None,
        autolog: bool = True,
        name_prefix: str | None = None,
        attributes: AnyDict | None = None,
        _tracer: "Tracer | None" = None,
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
            autolog: Automatically log task inputs, outputs, and execution metrics if otherwise unspecified.
            attributes: Additional attributes to attach to the run span.

        Returns:
            A RunSpan object that can be used as a context manager.
            The run will automatically be completed when the context manager exits.
        """
        if not self._initialized:
            self.configure()

        name_prefix = clean_str(name_prefix or coolname.generate_slug(2), replace_with="-")
        name = name or f"{name_prefix}-{random.randint(100, 999)}"  # noqa: S311 # nosec

        return RunSpan(
            name=name,
            project=project or self.project or "default",
            attributes=attributes,
            tracer=_tracer or self.get_tracer(),
            params=params,
            tags=tags,
            storage=self.storage,
            autolog=autolog,
        )

    @contextlib.contextmanager
    def task_and_run(
        self,
        name: str,
        *,
        task_name: str | None = None,
        task_type: "SpanType" = "task",
        project: str | None = None,
        tags: t.Sequence[str] | None = None,
        params: AnyDict | None = None,
        autolog: bool = True,
        inputs: AnyDict | None = None,
        label: str | None = None,
        _tracer: "Tracer | None" = None,
    ) -> t.Iterator[TaskSpan[t.Any]]:
        """
        Create a task span within a new run if one is not already active.

        Args:
            name: Name for the run (if created) and task (if task_name not specified).
            task_name: Optional separate name for the task span. If not provided, uses name.
            task_type: The type of span to create (task, evaluation, etc.).
        """

        create_run = current_run_span.get() is None
        with contextlib.ExitStack() as stack:
            if create_run:
                stack.enter_context(
                    self.run(
                        name_prefix=name,
                        project=project,
                        tags=tags,
                        params=params,
                        autolog=autolog,
                        _tracer=_tracer,
                    )
                )
                self.log_inputs(**(inputs or {}), to="run")

            task_span = stack.enter_context(
                self.task_span(
                    task_name or name, type=task_type, label=label, tags=tags, _tracer=_tracer
                )
            )
            self.log_inputs(**(inputs or {}))
            if not create_run:
                self.log_inputs(**(params or {}))

            yield task_span

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

        trace_context: dict[str, str] = {}
        propagate.inject(trace_context)

        return {
            "run_id": run.run_id,
            "run_name": run.name,
            "project": run.project_id,
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
            self.configure()

        return RunSpan.from_context(
            context=run_context,
            tracer=self.get_tracer(),
            storage=self.storage,
        )

    def tag(self, *tag: str, to: ToObject | t.Literal["both"] = "task-or-run") -> None:
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

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "tag() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.add_tags(tag)

    def init_package(
        self,
        name: str,
        package_type: PackageType,
    ) -> "Package":
        """
        Initialize a new package in the local storage.

        Args:
            name: Package name (e.g., "my-dataset").
            package_type: Type of package (datasets, models, toolsets, agents, environments).

        Returns:
            The initialized Package.
        """
        return Package.init(name=name, package_type=package_type, storage=self.storage)

    def build_package(
        self,
        path: str | Path,
    ) -> BuildResult:
        """
        Build a local repository into a wheel.

        Args:
            path: Path to the package source directory.

        Returns:
            BuildResult with success status and wheel path.
        """
        return Package(path=Path(path)).build()

    def push_package(
        self,
        path: str | Path,
        *,
        skip_upload: bool = False,
    ) -> "PushResult":
        """
        Build and Push a local package to the Dreadnode Registry.

        This handles artifact upload to CAS and wheel upload automatically.

        Args:
            path: Path to the package source directory.
            skip_upload: Skip uploading to remote (local only).

        Returns:
            PushResult with status and details.
        """
        import warnings

        # Auto-detect local mode and warn user
        is_local_mode = self.server == "local" or self._session is None
        if is_local_mode and not skip_upload:
            warnings.warn(
                "No remote credentials configured. Artifacts will be stored locally only. "
                "Use dn.configure() with server credentials to enable remote upload.",
                stacklevel=2,
            )
            skip_upload = True

        return Package(path=Path(path)).push(storage=self.storage, skip_upload=skip_upload)

    def pull_package(
        self,
        packages: list[str],
        *,
        upgrade: bool = False,
    ) -> "PullResult":
        """
        Download packages from the registry.

        Args:
            packages: Package names to install.
            upgrade: Upgrade if already installed.

        Returns:
            PullResult with status.
        """
        return Package.pull(*packages, upgrade=upgrade, _storage=self.storage)

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
        Log a single parameter to the current run.

        Parameters are key-value pairs that are associated with the run
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
        Log multiple parameters to the current run.

        Parameters are key-value pairs that are associated with the run
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
        value: float | bool,  # noqa: FBT001
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        aggregation: MetricAggMode | None = None,
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
            aggregation: The aggregation to use for the metric. Helpful when you want to let
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
        aggregation: MetricAggMode | None = None,
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
            aggregation: The aggregation to use for the metric. Helpful when you want to let
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
        value: float | bool | Metric,  # noqa: FBT001
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        aggregation: MetricAggMode | None = None,
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
                dreadnode.log_metric("loss", 0.05, step=10, aggregation="min")
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
            aggregation: The aggregation to use for the metric. Helpful when you want to let
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

        return target.log_metric(name, metric, origin=origin, aggregation=aggregation)

    @t.overload
    def log_metrics(
        self,
        metrics: dict[str, float | bool],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        aggregation: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
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
            aggregation: Aggregation for all metrics.
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
        aggregation: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics from a list of metric configurations.

        Example:
            ```
            dreadnode.log_metrics(
                [
                    {"name": "accuracy", "value": 0.95},
                    {"name": "loss", "value": 0.05, "aggregation": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: List of metric configurations to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            aggregation: Default aggregation for metrics if not supplied.
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
        metrics: MetricsLike,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        aggregation: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
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
                    {"name": "loss", "value": 0.05, "aggregation": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: Either a dictionary of name/value pairs or a list of MetricDicts to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            aggregation: Default aggregation for metrics if not supplied.
            attributes: Default attributes for metrics if not supplied.
            origin: The origin of the metrics - can be provided any object which was logged
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
                    aggregation=aggregation,
                    attributes=attributes,
                    origin=origin,
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
                    aggregation=metric.get("aggregation", aggregation),
                    attributes=metric.get("attributes", attributes) or {},
                    origin=origin,
                )
                for metric in metrics
            ]

        return logged_metrics

    # @handle_internal_errors()
    def log_artifact(
        self,
        local_uri: str | Path,
        *,
        name: str | None = None,
    ) -> None:
        """
        Log a file or directory artifact to the current run.

        This stores the artifact in the workspace CAS and uploads it to remote storage.
        Artifact metadata is recorded in artifacts.jsonl for tracking.

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
            local_uri: The local path to the file or directory to upload.
            name: Optional name for the artifact (defaults to filename).
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "log_artifact() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        # Store/upload artifact and get metadata
        artifact_metadata = run.log_artifact(local_uri=local_uri, name=name)

        # Write metadata to artifacts.jsonl
        if artifact_metadata and self._trace_config:
            if artifact_metadata.get("type") == "directory":
                # For directories, write each file's metadata
                for file_meta in artifact_metadata.get("files", []):
                    self._trace_config.write_artifact(file_meta)
            else:
                self._trace_config.write_artifact(artifact_metadata)

    @handle_internal_errors()
    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        to: ToObject | t.Literal["both"] = "task-or-run",
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

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "log_input() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.log_input(name, value, label=label, attributes=attributes)

    @handle_internal_errors()
    def log_inputs(
        self,
        to: ToObject | t.Literal["both"] = "task-or-run",
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
        to: ToObject | t.Literal["both"] = "task-or-run",
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

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "log_output() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.log_output(name, value, label=label, attributes=attributes)

    @handle_internal_errors()
    def log_outputs(
        self,
        to: ToObject | t.Literal["both"] = "task-or-run",
        **outputs: t.Any,
    ) -> None:
        """
        Log multiple outputs to the current task or run.

        See `log_output()` for more details.
        """
        for name, value in outputs.items():
            self.log_output(name, value, to=to)

    @handle_internal_errors()
    def log_sample(
        self,
        label: str,
        input: t.Any,
        output: t.Any,
        metrics: MetricsLike | None = None,
        *,
        step: int = 0,
    ) -> None:
        """
        Convenience method to log an input/output pair with metrics as a ephemeral task.

        This is useful for logging a single sample of input and output data
        along with any metrics that were computed during the process.
        """

        with self.task_span(name=label, label=label):
            self.log_input("input", input)
            self.log_output("output", output)
            self.link_objects(output, input)
            if metrics is not None:
                self.log_metrics(metrics, step=step, origin=output)

    @handle_internal_errors()
    def log_samples(
        self,
        name: str,
        samples: list[tuple[t.Any, t.Any] | tuple[t.Any, t.Any, MetricsLike]],
    ) -> None:
        """
        Log multiple input/output samples as ephemeral tasks.

        This is useful for logging a batch of input/output pairs with metrics
        in a single run.

        Example:
            ```
            dreadnode.log_samples(
                "my_samples",
                [
                    (input1, output1, {"accuracy": 0.95}),
                    (input2, output2, {"accuracy": 0.90}),
                ]
            )
            ```

        Args:
            name: The name of the task to create for each sample.
            samples: A list of tuples containing (input, output, metrics [optional]).
        """
        for sample in samples:
            metrics: MetricsLike | None = None
            if len(sample) == 3:
                input_data, output_data, metrics = sample
            elif len(sample) == 2:
                input_data, output_data = sample
            else:
                raise ValueError(
                    "Each sample must be a tuple of (input, output) or (input, output, metrics)",
                )

            # Log each sample as an ephemeral task
            self.log_sample(name, input_data, output_data, metrics=metrics)

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
