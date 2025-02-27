import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import typing_extensions as te
from opentelemetry.sdk.trace import ReadableSpan

from .constants import (
    SPAN_ATTRIBUTE_PROJECT_KEY,
    SPAN_ATTRIBUTE_RUN_ARTIFACTS_KEY,
    SPAN_ATTRIBUTE_RUN_ID_KEY,
    SPAN_ATTRIBUTE_RUN_METRICS_KEY,
    SPAN_ATTRIBUTE_RUN_PARAMS_KEY,
    SPAN_ATTRIBUTE_TAGS_KEY,
    SPAN_ATTRIBUTE_TYPE_KEY,
)
from .exporters import FileExportConfig
from .tracing import JsonDict, Metric, MetricDict, Score


@dataclass
class Run:
    """A static representation of a completed run."""

    run_id: str
    name: str
    project: str
    start_time: datetime
    end_time: datetime | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: JsonDict = field(default_factory=dict)
    metrics: MetricDict = field(default_factory=dict)
    attributes: JsonDict = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_span(cls, span: ReadableSpan) -> te.Self:
        """Create a Run object from a ReadableSpan."""
        if span.attributes.get(SPAN_ATTRIBUTE_TYPE_KEY) != "run":
            raise ValueError(f"Span is not a run span: {span.attributes.get(SPAN_ATTRIBUTE_TYPE_KEY)}")

        run_id = span.attributes.get(SPAN_ATTRIBUTE_RUN_ID_KEY, "")
        if not run_id:
            raise ValueError("Span does not have a run ID")

        # Extract basic run information
        project = span.attributes.get(SPAN_ATTRIBUTE_PROJECT_KEY, "default")
        name = span.name

        # Extract time information
        start_time = datetime.fromtimestamp(span.start_time / 1_000_000_000)
        end_time = None
        if span.end_time:
            end_time = datetime.fromtimestamp(span.end_time / 1_000_000_000)

        # Extract tags
        tags = tuple(span.attributes.get(SPAN_ATTRIBUTE_TAGS_KEY, ()))

        # Extract params, metrics, and artifacts
        params_json = span.attributes.get(SPAN_ATTRIBUTE_RUN_PARAMS_KEY, "{}")
        metrics_json = span.attributes.get(SPAN_ATTRIBUTE_RUN_METRICS_KEY, "{}")
        artifacts_json = span.attributes.get(SPAN_ATTRIBUTE_RUN_ARTIFACTS_KEY, "{}")

        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        metrics_dict = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
        artifacts = json.loads(artifacts_json) if isinstance(artifacts_json, str) else artifacts_json

        # Convert metrics from their serialized form back to MetricDict
        metrics: MetricDict = {}
        for key, metrics_list in metrics_dict.items():
            metrics[key] = [
                Metric(
                    value=m.get("value", 0.0),
                    step=m.get("step", 0),
                    timestamp=datetime.fromisoformat(m.get("timestamp", start_time.isoformat())),
                )
                for m in metrics_list
            ]

        # Create a clean copy of attributes without internal keys
        internal_keys = {
            SPAN_ATTRIBUTE_RUN_ID_KEY,
            SPAN_ATTRIBUTE_PROJECT_KEY,
            SPAN_ATTRIBUTE_TAGS_KEY,
            SPAN_ATTRIBUTE_RUN_PARAMS_KEY,
            SPAN_ATTRIBUTE_RUN_METRICS_KEY,
            SPAN_ATTRIBUTE_RUN_ARTIFACTS_KEY,
            SPAN_ATTRIBUTE_TYPE_KEY,
        }

        attributes = {k: v for k, v in span.attributes.items() if k not in internal_keys}

        return cls(
            run_id=run_id,
            name=name,
            project=project,
            start_time=start_time,
            end_time=end_time,
            tags=tags,
            params=params,
            metrics=metrics,
            attributes=attributes,
            artifact_paths=artifacts,
        )

    @classmethod
    def load(cls, run_id: str, base_path: str | Path = None) -> te.Self:
        """Load a run from disk based on its run_id."""
        if base_path is None:
            config = FileExportConfig()
            base_path = config.base_path
        else:
            config = FileExportConfig(base_path=base_path)

        traces_path = config.get_path("traces")
        if not traces_path.exists():
            raise FileNotFoundError(f"Traces file not found at {traces_path}")

        # Find the run span with the matching run_id
        matching_run_span = None

        with open(traces_path) as f:
            for line in f:
                try:
                    trace_data = json.loads(line)
                    # Process the OTLP JSON format to extract span data
                    resource_spans = trace_data.get("resourceSpans", [])

                    for resource_span in resource_spans:
                        scope_spans = resource_span.get("scopeSpans", [])

                        for scope_span in scope_spans:
                            spans = scope_span.get("spans", [])

                            for span_data in spans:
                                # Extract attributes
                                attributes = {}
                                for attr in span_data.get("attributes", []):
                                    key = attr.get("key", "")
                                    value = attr.get("value", {})

                                    # Handle different value types
                                    if "stringValue" in value:
                                        attributes[key] = value["stringValue"]
                                    elif "intValue" in value:
                                        attributes[key] = value["intValue"]
                                    elif "doubleValue" in value:
                                        attributes[key] = value["doubleValue"]
                                    elif "boolValue" in value:
                                        attributes[key] = value["boolValue"]

                                # Check if this is a run span with the matching run_id
                                if (
                                    attributes.get(SPAN_ATTRIBUTE_TYPE_KEY) == "run"
                                    and attributes.get(SPAN_ATTRIBUTE_RUN_ID_KEY) == run_id
                                ):
                                    # Convert to a ReadableSpan-like object for processing
                                    span_obj = type(
                                        "ReadableSpanLike",
                                        (),
                                        {
                                            "name": span_data.get("name", ""),
                                            "attributes": attributes,
                                            "start_time": int(span_data.get("startTimeUnixNano", 0)),
                                            "end_time": int(span_data.get("endTimeUnixNano", 0)),
                                        },
                                    )

                                    matching_run_span = span_obj
                                    break

                            if matching_run_span:
                                break

                        if matching_run_span:
                            break

                    if matching_run_span:
                        break

                except json.JSONDecodeError:
                    continue

        if not matching_run_span:
            raise ValueError(f"No run found with ID {run_id}")

        return cls.from_span(matching_run_span)

    @classmethod
    def list_runs(cls, base_path: str | Path = None, project: str = None) -> list[str]:
        """List all run IDs available in the traces file."""
        if base_path is None:
            config = FileExportConfig()
            base_path = config.base_path
        else:
            config = FileExportConfig(base_path=base_path)

        traces_path = config.get_path("traces")
        if not traces_path.exists():
            return []

        run_ids = set()

        with open(traces_path) as f:
            for line in f:
                try:
                    trace_data = json.loads(line)
                    resource_spans = trace_data.get("resourceSpans", [])

                    for resource_span in resource_spans:
                        scope_spans = resource_span.get("scopeSpans", [])

                        for scope_span in scope_spans:
                            spans = scope_span.get("spans", [])

                            for span_data in spans:
                                # Extract attributes
                                attributes = {}
                                for attr in span_data.get("attributes", []):
                                    key = attr.get("key", "")
                                    value = attr.get("value", {})

                                    # Only extract the keys we need for filtering
                                    if key in (
                                        SPAN_ATTRIBUTE_TYPE_KEY,
                                        SPAN_ATTRIBUTE_RUN_ID_KEY,
                                        SPAN_ATTRIBUTE_PROJECT_KEY,
                                    ):
                                        if "stringValue" in value:
                                            attributes[key] = value["stringValue"]

                                # Check if this is a run span
                                if attributes.get(SPAN_ATTRIBUTE_TYPE_KEY) == "run":
                                    run_id = attributes.get(SPAN_ATTRIBUTE_RUN_ID_KEY)
                                    run_project = attributes.get(SPAN_ATTRIBUTE_PROJECT_KEY)

                                    # If project filter is applied, check if it matches
                                    if project is None or run_project == project:
                                        if run_id:
                                            run_ids.add(run_id)
                except json.JSONDecodeError:
                    continue

        return sorted(list(run_ids))

    def get_scores(self) -> list[Score]:
        """Extract scores from the run's metrics."""
        scores = []
        for name, metrics_list in self.metrics.items():
            # Skip derived metrics (.avg, .cum, .count)
            if "." in name:
                continue

            for metric in metrics_list:
                score = Score(
                    name=name,
                    value=metric.value,
                    timestamp=metric.timestamp,
                )
                scores.append(score)

        return scores
