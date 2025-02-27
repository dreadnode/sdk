import typing as t
from datetime import datetime
from uuid import UUID

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
    last_run: StrikeProjectRunResponse | None


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
