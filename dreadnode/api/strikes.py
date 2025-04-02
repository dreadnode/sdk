import io
import typing as t
from datetime import datetime
from uuid import UUID

import pandas as pd  # type: ignore
from pydantic import BaseModel, Field
from ulid import ULID

if t.TYPE_CHECKING:
    from .client import ApiClient  # Assuming relative import is correct

# Type Aliases for Literal constraints

StrikeSpanStatus = t.Literal[
    "pending",  # A pending span has been created
    "completed",  # The span has been finished
    "failed",  # The span raised an exception
]
"""Literal type defining the possible statuses of a Strike span."""

ExportFormat = t.Literal["csv", "json", "jsonl", "parquet"]
"""Literal type defining the supported file formats for exporting data."""

StatusFilter = t.Literal["all", "completed", "failed"]
"""Literal type defining the status filters applicable when exporting runs."""

TimeAxisType = t.Literal["wall", "relative", "step"]
"""Literal type defining the types of time axes available for timeseries exports."""

TimeAggregationType = t.Literal["max", "min", "sum", "count"]
"""Literal type defining the aggregation methods for time-based data."""

MetricAggregationType = t.Literal["avg", "median", "min", "max", "sum", "first", "last", "count", "std", "var"]
"""Literal type defining the aggregation methods for metric data."""


# Models

class StrikeSpanException(BaseModel):
    """Represents details of an exception recorded within a span.

    Attributes:
        type: The type or class name of the exception (e.g., 'ValueError').
        message: The exception message.
        stacktrace: The stacktrace associated with the exception, if available.
    """
    type: str
    message: str
    stacktrace: str


class StrikeSpanEvent(BaseModel):
    """Represents an event occurring within a span's lifetime.

    Corresponds to OpenTelemetry Span Events.

    Attributes:
        timestamp: The time at which the event occurred.
        name: The name identifying the event.
        attributes: A dictionary of key-value attributes associated with the event.
    """
    timestamp: datetime
    name: str
    attributes: dict[str, str]


class StrikeSpanLink(BaseModel):
    """Represents a link from one span to another.

    Corresponds to OpenTelemetry Span Links.

    Attributes:
        trace_id: The trace ID of the linked span.
        span_id: The span ID of the linked span.
        attributes: Attributes associated with the link itself.
    """
    trace_id: str
    span_id: str
    attributes: dict[str, str]


class StrikeTraceLog(BaseModel):
    """Represents a log record potentially associated with a trace or span.

    Attributes:
        timestamp: The time the log record was created.
        body: The main content or message of the log.
        severity: The severity level of the log (e.g., 'INFO', 'ERROR').
        service: The name of the service that generated the log, if applicable.
        trace_id: The trace ID associated with this log, if any.
        span_id: The span ID associated with this log, if any.
        attributes: Additional key-value attributes associated with the log.
        container: Identifier for the container or environment where the log originated, if applicable.
    """
    timestamp: datetime
    body: str
    severity: str
    service: str | None
    trace_id: str | None
    span_id: str | None
    attributes: dict[str, str]
    container: str | None


class StrikeTraceSpan(BaseModel):
    """Represents a generic span within a trace, not necessarily a Task.

    Attributes:
        timestamp: The start time of the span.
        duration: The duration of the span in nanoseconds.
        trace_id: The ID of the trace this span belongs to.
        span_id: The unique ID of this span.
        parent_span_id: The ID of the parent span, if this is a child span.
        service_name: The name of the service where the span originated.
        status: The final status of the span ('pending', 'completed', 'failed').
        exception: Details of any exception that occurred during the span, if status is 'failed'.
        name: The name of the span.
        attributes: Key-value attributes associated with the span.
        resource_attributes: Attributes describing the resource that generated the span (e.g., host, SDK version).
        events: A list of events that occurred during the span's execution.
        links: A list of links to other spans.
    """
    timestamp: datetime
    duration: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str | None
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    name: str
    attributes: dict[str, str]
    resource_attributes: dict[str, str]
    events: list[StrikeSpanEvent]
    links: list[StrikeSpanLink]


class ProjectMetric(BaseModel):
    """Represents a single data point for a metric within a project run.

    Attributes:
        timestamp: The time associated with the metric value.
        value: The numerical value of the metric.
        step: An optional step counter, often used for metrics logged over time (e.g., in training loops).
    """
    timestamp: datetime
    value: float
    step: int


ProjectParams = dict[str, t.Any]
"""Type alias for parameters associated with a project run (a dictionary)."""


class ProjectTaskScore(BaseModel):
    """Represents a score assigned during a task execution within a run.

    Scores are typically generated by evaluation functions (scorers).

    Attributes:
        name: The name identifying the score (e.g., 'accuracy', 'latency_score').
        value: The numerical value of the score.
        attributes: Additional key-value attributes associated with the score.
        timestamp: The time the score was recorded.
    """
    name: str
    value: float
    attributes: dict[str, t.Any]
    timestamp: datetime


class StrikeProjectRunResponse(BaseModel):
    """Represents the data structure for a single project run retrieved from the API.

    Attributes:
        id: The unique ULID identifier for the run.
        name: The name assigned to the run.
        span_id: The ID of the root span representing this run.
        trace_id: The trace ID associated with this run.
        timestamp: The start time of the run.
        duration: The duration of the run in nanoseconds.
        status: The final status of the run ('pending', 'completed', 'failed').
        exception: Details of any exception if the run failed.
        tags: A set of string tags associated with the run for categorization.
        params: A dictionary of parameters logged during the run.
        metrics: A dictionary where keys are metric names and values are lists of ProjectMetric data points.
        schema_: A dictionary representing schema information, potentially related to inputs/outputs (aliased from 'schema').
    """
    id: ULID
    name: str
    span_id: str
    trace_id: str
    timestamp: datetime
    duration: int
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    tags: set[str]
    params: dict[str, t.Any]
    metrics: dict[str, list[ProjectMetric]]
    schema_: dict[str, t.Any] = Field(alias="schema")


class StrikeProjectTaskResponse(BaseModel):
    """Represents the data structure for a single task executed within a run.

    Attributes:
        name: The name of the task.
        span_id: The unique ID of the span representing this task execution.
        trace_id: The trace ID this task belongs to.
        parent_span_id: The ID of the parent span (could be the run span or another task span).
        parent_task_span_id: The ID of the direct parent task span, if nested within another task.
        timestamp: The start time of the task execution.
        duration: The duration of the task execution in nanoseconds.
        status: The final status of the task ('pending', 'completed', 'failed').
        exception: Details of any exception if the task failed.
        tags: A set of string tags associated with the task.
        args: A dictionary representing the arguments passed to the task.
        output: The output or return value of the task.
        scores: A list of scores recorded during the task execution.
        schema_: A dictionary representing schema information (aliased from 'schema').
        attributes: Key-value attributes associated specifically with this task span.
        resource_attributes: Attributes describing the resource where the task executed.
        events: A list of events recorded during the task's execution.
        links: A list of links from this task span to other spans.
    """
    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None
    parent_task_span_id: str | None
    timestamp: datetime
    duration: int
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    tags: set[str]
    args: dict[str, t.Any]
    output: t.Any
    scores: list[ProjectTaskScore]
    schema_: dict[str, t.Any] = Field(alias="schema")
    attributes: dict[str, str]
    resource_attributes: dict[str, str]
    events: list[StrikeSpanEvent]
    links: list[StrikeSpanLink]


class CreateStrikeProjectRequest(BaseModel):
    """Data model for creating a new project via the API.

    Attributes:
        name: The desired name for the new project. Must be unique.
        description: An optional description for the project.
    """
    name: str
    description: str | None = None


class UpdateStrikeProjectRequest(BaseModel):
    """Data model for updating an existing project via the API.

    Attributes:
        name: The new name for the project. Must be unique.
        description: An optional updated description for the project.
    """
    name: str
    description: str | None = None


class StrikeProjectResponse(BaseModel):
    """Represents the data structure for a project retrieved from the API.

    Attributes:
        id: The unique UUID identifier for the project.
        key: A unique key identifying the project (potentially used in paths or identifiers).
        name: The name of the project.
        description: The description of the project, if provided.
        created_at: The timestamp when the project was created.
        updated_at: The timestamp when the project was last updated.
        run_count: The total number of runs recorded for this project.
        last_run: Details of the most recently recorded run for this project, if any.
    """
    id: UUID
    key: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run: StrikeProjectRunResponse | None


# Client

class StrikesClient:
    """Client for interacting with the 'Strikes' (Project/Run/Task) part of the Dreadnode API.

    Provides methods for managing projects, listing and retrieving runs and tasks,
    and exporting various data associated with project runs. This client relies
    on an underlying configured ApiClient instance for making HTTP requests.
    """

    def __init__(self, client: "ApiClient") -> None:
        """Initializes the StrikesClient.

        Args:
            client: An initialized ApiClient instance used to make requests.
        """
        self._client = client

    def list_projects(self) -> list[StrikeProjectResponse]:
        """Lists all available projects.

        Makes a GET request to the '/strikes/projects' endpoint.

        Returns:
            A list of StrikeProjectResponse objects representing the projects.
        """
        response = self._client.request("GET", "/strikes/projects")
        # TODO: Add response validation/parsing helper
        return [StrikeProjectResponse(**project) for project in response.json()]

    def get_project(self, project: str) -> StrikeProjectResponse:
        """Retrieves a specific project by its key or ID.

        Makes a GET request to the '/strikes/projects/{project_id}' endpoint.

        Args:
            project: The key or ID of the project to retrieve.

        Returns:
            A StrikeProjectResponse object representing the requested project.
        """
        response = self._client.request("GET", f"/strikes/projects/{str(project)}")
        return StrikeProjectResponse(**response.json())

    def list_runs(self, project: str) -> list[StrikeProjectRunResponse]:
        """Lists all runs associated with a specific project.

        Makes a GET request to the '/strikes/projects/{project_id}/runs' endpoint.

        Args:
            project: The key or ID of the project whose runs are to be listed.

        Returns:
            A list of StrikeProjectRunResponse objects for the project.
        """
        response = self._client.request("GET", f"/strikes/projects/{str(project)}/runs")
        return [StrikeProjectRunResponse(**run) for run in response.json()]

    def get_run(self, run: str | ULID) -> StrikeProjectRunResponse:
        """Retrieves a specific run by its ULID.

        Makes a GET request to the '/strikes/projects/runs/{run_id}' endpoint.

        Args:
            run: The ULID (or string representation) of the run to retrieve.

        Returns:
            A StrikeProjectRunResponse object representing the requested run.
        """
        if isinstance(run, str):
            # Attempt conversion, assuming valid ULID string
            try:
                run = ULID.from_str(run)
            except ValueError as e:
                raise ValueError(f"Invalid ULID string format for run: {run}") from e
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}")
        return StrikeProjectRunResponse(**response.json())

    def get_run_tasks(self, run: str | ULID) -> list[StrikeProjectTaskResponse]:
        """Retrieves all tasks associated with a specific run.

        Makes a GET request to the '/strikes/projects/runs/{run_id}/tasks' endpoint.

        Args:
            run: The ULID (or string representation) of the run whose tasks are to be retrieved.

        Returns:
            A list of StrikeProjectTaskResponse objects for the specified run.
        """
        if isinstance(run, str):
             try:
                run = ULID.from_str(run)
             except ValueError as e:
                 raise ValueError(f"Invalid ULID string format for run: {run}") from e
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}/tasks")
        return [StrikeProjectTaskResponse(**task) for task in response.json()]

    def get_run_trace(self, run: str | ULID) -> list[StrikeProjectTaskResponse | StrikeTraceSpan]:
        """Retrieves all spans (both Task spans and generic spans) for a specific run.

        Makes a GET request to the '/strikes/projects/runs/{run_id}/spans' endpoint.
        Differentiates between Task spans and generic spans based on the presence
        of the 'parent_task_span_id' field.

        Args:
            run: The ULID (or string representation) of the run whose trace spans are required.

        Returns:
            A list containing a mix of StrikeProjectTaskResponse and StrikeTraceSpan
            objects, representing the entire trace for the run.
        """
        if isinstance(run, str):
             try:
                run = ULID.from_str(run)
             except ValueError as e:
                 raise ValueError(f"Invalid ULID string format for run: {run}") from e
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}/spans")
        spans: list[StrikeProjectTaskResponse | StrikeTraceSpan] = []
        for item in response.json():
            # Heuristic to differentiate Task spans from generic spans
            if "parent_task_span_id" in item:
                spans.append(StrikeProjectTaskResponse(**item))
            else:
                spans.append(StrikeTraceSpan(**item))
        return spans

    def export_runs(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet", # Currently hardcoded to parquet
        status: StatusFilter = "completed",
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports run-level data for a project to a pandas DataFrame.

        Exports core run information along with aggregated metrics and parameters.
        Makes a GET request to the '/strikes/projects/{project_id}/export' endpoint.
        Currently requests data in Parquet format.

        Args:
            project: The key or ID of the project to export runs from.
            filter: An optional filter string (syntax depends on API implementation)
                    to apply to the runs before exporting.
            status: Filter runs by status ('all', 'completed', 'failed'). Defaults to 'completed'.
            aggregations: Optional list of aggregation functions to apply to metrics
                          (e.g., 'avg', 'max'). If None, default aggregations might be applied by the API.

        Returns:
            A pandas DataFrame containing the exported run data.
        """
        query_params = {
            "format": "parquet", # Hardcoded for now
            "status": status,
        }
        if filter:
            query_params["filter"] = filter
        if aggregations:
            query_params["aggregations"] = aggregations # Send list directly

        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export",
            query_params=query_params,
        )
        # Assuming response content is valid Parquet bytes
        return pd.read_parquet(io.BytesIO(response.content))

    def export_metrics(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports specific metric data across runs for a project.

        Makes a GET request to the '/strikes/projects/{project_id}/export/metrics' endpoint.
        Currently requests data in Parquet format.

        Args:
            project: The key or ID of the project.
            filter: An optional filter string for runs.
            status: Filter runs by status. Defaults to 'completed'.
            metrics: An optional list of specific metric names to export. If None,
                     the API might export all available metrics or a default set.
            aggregations: Optional list of aggregation functions to apply to the
                          selected metrics (e.g., 'avg', 'max').

        Returns:
            A pandas DataFrame containing the exported metric data.
        """
        query_params = {
            "format": "parquet",
            "status": status,
        }
        if filter: query_params["filter"] = filter
        if metrics: query_params["metrics"] = metrics
        if aggregations: query_params["aggregations"] = aggregations

        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/metrics",
            query_params=query_params,
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_parameters(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        parameters: list[str] | None = None,
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports parameter data alongside optional aggregated metrics for project runs.

        Makes a GET request to the '/strikes/projects/{project_id}/export/parameters' endpoint.
        Currently requests data in Parquet format.

        Args:
            project: The key or ID of the project.
            filter: An optional filter string for runs.
            status: Filter runs by status. Defaults to 'completed'.
            parameters: An optional list of specific parameter names to export. If None,
                        the API might export all available parameters.
            metrics: An optional list of metric names whose aggregated values should
                     be included alongside the parameters.
            aggregations: Optional list of aggregation functions to apply to the
                          specified metrics.

        Returns:
            A pandas DataFrame containing the exported parameter and metric data.
        """
        query_params = {
            "format": "parquet",
            "status": status,
        }
        if filter: query_params["filter"] = filter
        if parameters: query_params["parameters"] = parameters
        if metrics: query_params["metrics"] = metrics
        if aggregations: query_params["aggregations"] = aggregations

        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/parameters",
            query_params=query_params,
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_timeseries(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        time_axis: TimeAxisType = "relative",
        aggregations: list[TimeAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports metric timeseries data across runs for a project.

        Allows specifying the time axis type (wall clock, relative to run start, or step).
        Makes a GET request to the '/strikes/projects/{project_id}/export/timeseries' endpoint.
        Currently requests data in Parquet format.

        Args:
            project: The key or ID of the project.
            filter: An optional filter string for runs.
            status: Filter runs by status. Defaults to 'completed'.
            metrics: An optional list of specific metric names to export timeseries for.
                     If None, the API might export all available metrics.
            time_axis: The type of time axis to use ('wall', 'relative', 'step').
                       Defaults to 'relative'.
            aggregations: Optional list of aggregation functions to apply over time points
                          if multiple runs match the filter (e.g., 'max', 'min').

        Returns:
            A pandas DataFrame containing the exported timeseries data.
        """
        query_params = {
            "format": "parquet",
            "status": status,
            "time_axis": time_axis,
        }
        if filter: query_params["filter"] = filter
        if metrics: query_params["metrics"] = metrics
        # Note: API parameter might be 'aggregation' (singular) based on original code query param
        if aggregations: query_params["aggregation"] = aggregations

        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/timeseries",
            query_params=query_params,
        )
        return pd.read_parquet(io.BytesIO(response.content))