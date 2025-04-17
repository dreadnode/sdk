import io
import typing as t
from datetime import datetime
from uuid import UUID

import pandas as pd  # type: ignore
from pydantic import BaseModel, Field
from ulid import ULID

if t.TYPE_CHECKING:
    from .client import ApiClient

# Models

StrikeSpanStatus = t.Literal[
    "pending",  # A pending span has been created
    "completed",  # The span has been finished
    "failed",  # The raised an exception
]

ExportFormat = t.Literal["csv", "json", "jsonl", "parquet"]
StatusFilter = t.Literal["all", "completed", "failed"]
TimeAxisType = t.Literal["wall", "relative", "step"]
TimeAggregationType = t.Literal["max", "min", "sum", "count"]
MetricAggregationType = t.Literal["avg", "median", "min", "max", "sum", "first", "last", "count", "std", "var"]


class StrikeSpanException(BaseModel):
    type: str
    message: str
    stacktrace: str


class StrikeSpanEvent(BaseModel):
    timestamp: datetime
    name: str
    attributes: dict[str, str]


class StrikeSpanLink(BaseModel):
    trace_id: str
    span_id: str
    attributes: dict[str, str]


class StrikeTraceLog(BaseModel):
    timestamp: datetime
    body: str
    severity: str
    service: str | None
    trace_id: str | None
    span_id: str | None
    attributes: dict[str, str]
    container: str | None


class StrikeTraceSpan(BaseModel):
    timestamp: datetime
    duration: int
    trace_id: str = Field(repr=False)
    span_id: str
    parent_span_id: str | None = Field(repr=False)
    service_name: str | None = Field(repr=False)
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    name: str
    attributes: dict[str, str] = Field(repr=False)
    resource_attributes: dict[str, str] = Field(repr=False)
    events: list[StrikeSpanEvent] = Field(repr=False)
    links: list[StrikeSpanLink] = Field(repr=False)


class ProjectMetric(BaseModel):
    timestamp: datetime
    value: float
    step: int


ProjectParams = dict[str, t.Any]


class ProjectTaskScore(BaseModel):
    name: str
    value: float
    attributes: dict[str, t.Any]
    timestamp: datetime


class StrikeProjectRunResponse(BaseModel):
    id: ULID
    name: str
    span_id: str
    trace_id: str = Field(repr=False)
    timestamp: datetime
    duration: int
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    tags: set[str]
    params: dict[str, t.Any] = Field(repr=False)
    metrics: dict[str, list[ProjectMetric]] = Field(repr=False)
    schema_: dict[str, t.Any] = Field(alias="schema", repr=False)


class StrikeProjectTaskResponse(BaseModel):
    name: str
    span_id: str
    trace_id: str = Field(repr=False)
    parent_span_id: str | None = Field(repr=False)
    parent_task_span_id: str | None = Field(repr=False)
    timestamp: datetime
    duration: int
    status: StrikeSpanStatus
    exception: StrikeSpanException | None
    tags: set[str]
    args: dict[str, t.Any]
    output: t.Any
    scores: list[ProjectTaskScore]
    schema_: dict[str, t.Any] = Field(alias="schema", repr=False)
    attributes: dict[str, str] = Field(repr=False)
    resource_attributes: dict[str, str] = Field(repr=False)
    events: list[StrikeSpanEvent] = Field(repr=False)
    links: list[StrikeSpanLink] = Field(repr=False)


class CreateStrikeProjectRequest(BaseModel):
    name: str
    description: str | None = None


class UpdateStrikeProjectRequest(BaseModel):
    name: str
    description: str | None = None


class StrikeProjectResponse(BaseModel):
    id: UUID
    key: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run: StrikeProjectRunResponse | None = Field(repr=False)


# Client


class StrikesClient:
    def __init__(self, client: "ApiClient") -> None:
        self._client = client

    def list_projects(self) -> list[StrikeProjectResponse]:
        response = self._client.request("GET", "/strikes/projects")
        return [StrikeProjectResponse(**project) for project in response.json()]

    def get_project(self, project: str) -> StrikeProjectResponse:
        response = self._client.request("GET", f"/strikes/projects/{str(project)}")
        return StrikeProjectResponse(**response.json())

    def list_runs(self, project: str) -> list[StrikeProjectRunResponse]:
        response = self._client.request("GET", f"/strikes/projects/{str(project)}/runs")
        return [StrikeProjectRunResponse(**run) for run in response.json()]

    def get_run(self, run: str | ULID) -> StrikeProjectRunResponse:
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}")
        return StrikeProjectRunResponse(**response.json())

    def get_run_tasks(self, run: str | ULID) -> list[StrikeProjectTaskResponse]:
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}/tasks")
        return [StrikeProjectTaskResponse(**task) for task in response.json()]

    def get_run_trace(self, run: str | ULID) -> list[StrikeProjectTaskResponse | StrikeTraceSpan]:
        response = self._client.request("GET", f"/strikes/projects/runs/{str(run)}/spans")
        spans: list[StrikeProjectTaskResponse | StrikeTraceSpan] = []
        for item in response.json():
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
        # format: ExportFormat = "parquet",
        status: StatusFilter = "all",
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export",
            query_params={
                "format": "parquet",
                "status": status,
                **({"filter": filter} if filter else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_metrics(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "all",
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/metrics",
            query_params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                **({"metrics": metrics} if metrics else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_parameters(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "all",
        parameters: list[str] | None = None,
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/parameters",
            query_params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                **({"parameters": parameters} if parameters else {}),
                **({"metrics": metrics} if metrics else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_timeseries(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "all",
        metrics: list[str] | None = None,
        time_axis: TimeAxisType = "relative",
        aggregations: list[TimeAggregationType] | None = None,
    ) -> pd.DataFrame:
        response = self._client.request(
            "GET",
            f"/strikes/projects/{str(project)}/export/timeseries",
            query_params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                "time_axis": time_axis,
                **({"metrics": metrics} if metrics else {}),
                **({"aggregation": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))
