"""
Convert spans.jsonl to analysis-friendly tables.

This module reads OTLP span data from spans.jsonl and extracts typed tables
for different execution types (runs, agents, evaluations, studies, etc.).
"""

import json
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class SpanRecord:
    """Parsed span from OTLP format."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    status: str
    span_type: str  # run, agent, evaluation, study, task
    run_id: str | None

    # Dreadnode-specific attributes
    inputs: dict[str, t.Any] = field(default_factory=dict)
    outputs: dict[str, t.Any] = field(default_factory=dict)
    params: dict[str, t.Any] = field(default_factory=dict)
    metrics: dict[str, t.Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    objects: dict[str, t.Any] = field(default_factory=dict)

    # Raw attributes for debugging
    raw_attributes: dict[str, t.Any] = field(default_factory=dict)


def _parse_otlp_value(value: dict[str, t.Any]) -> t.Any:
    """Parse OTLP attribute value."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_parse_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    return value


def _parse_attributes(attrs: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    """Parse OTLP attributes list into a dictionary."""
    result = {}
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        result[key] = _parse_otlp_value(value)
    return result


def _parse_timestamp(ns: str | int) -> datetime:
    """Parse nanosecond timestamp to datetime."""
    ns_int = int(ns)
    return datetime.fromtimestamp(ns_int / 1e9)


def _parse_span(span_data: dict[str, t.Any]) -> SpanRecord:
    """Parse a single OTLP span into a SpanRecord."""
    import base64

    # Parse IDs (base64 encoded)
    trace_id = base64.b64decode(span_data.get("traceId", "")).hex()
    span_id = base64.b64decode(span_data.get("spanId", "")).hex()
    parent_span_id = None
    if span_data.get("parentSpanId"):
        parent_span_id = base64.b64decode(span_data["parentSpanId"]).hex()

    # Parse timestamps
    start_time = _parse_timestamp(span_data.get("startTimeUnixNano", 0))
    end_time = None
    duration_ms = None
    if span_data.get("endTimeUnixNano"):
        end_time = _parse_timestamp(span_data["endTimeUnixNano"])
        duration_ms = (end_time - start_time).total_seconds() * 1000

    # Parse attributes
    raw_attrs = _parse_attributes(span_data.get("attributes", []))

    # Extract dreadnode-specific attributes
    span_type = raw_attrs.get("dreadnode.type", "task")
    run_id = raw_attrs.get("dreadnode.run.id")

    # Parse JSON-encoded attributes
    inputs = {}
    outputs = {}
    params = {}
    metrics = {}
    objects = {}
    tags = raw_attrs.get("dreadnode.tags", [])

    # Parse inputs (list of references)
    inputs_raw = raw_attrs.get("dreadnode.inputs", "[]")
    if isinstance(inputs_raw, str):
        try:
            inputs_list = json.loads(inputs_raw)
            for inp in inputs_list:
                if isinstance(inp, dict) and "name" in inp:
                    inputs[inp["name"]] = inp.get("hash")
        except json.JSONDecodeError:
            pass

    # Parse outputs
    outputs_raw = raw_attrs.get("dreadnode.outputs", "[]")
    if isinstance(outputs_raw, str):
        try:
            outputs_list = json.loads(outputs_raw)
            for out in outputs_list:
                if isinstance(out, dict) and "name" in out:
                    outputs[out["name"]] = out.get("hash")
        except json.JSONDecodeError:
            pass

    # Parse params
    params_raw = raw_attrs.get("dreadnode.params", "{}")
    if isinstance(params_raw, str):
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            pass

    # Parse metrics
    metrics_raw = raw_attrs.get("dreadnode.metrics", "{}")
    if isinstance(metrics_raw, str):
        try:
            metrics = json.loads(metrics_raw)
        except json.JSONDecodeError:
            pass

    # Parse objects (the actual values)
    objects_raw = raw_attrs.get("dreadnode.objects", "{}")
    if isinstance(objects_raw, str):
        try:
            objects = json.loads(objects_raw)
        except json.JSONDecodeError:
            pass

    # Determine status
    status_data = span_data.get("status", {})
    status = "finished"
    if status_data.get("code") == 2:  # ERROR
        status = "error"

    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=span_data.get("name", ""),
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        status=status,
        span_type=span_type,
        run_id=run_id,
        inputs=inputs,
        outputs=outputs,
        params=params,
        metrics=metrics,
        tags=tags,
        objects=objects,
        raw_attributes=raw_attrs,
    )


def load_spans(spans_file: Path) -> list[SpanRecord]:
    """Load and parse spans from a JSONL file."""
    spans = []

    with open(spans_file) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                # OTLP format has resourceSpans -> scopeSpans -> spans
                for resource_span in data.get("resourceSpans", []):
                    for scope_span in resource_span.get("scopeSpans", []):
                        for span_data in scope_span.get("spans", []):
                            spans.append(_parse_span(span_data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Failed to parse span line: {e}")

    return spans


def _resolve_object_value(hash_ref: str | None, objects: dict[str, t.Any]) -> t.Any:
    """Resolve an object hash to its value."""
    if hash_ref is None:
        return None
    obj = objects.get(hash_ref)
    if obj is None:
        return hash_ref  # Return the hash if object not found
    return obj.get("value")


def extract_runs(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract run-level data."""
    runs = []
    for span in spans:
        if span.span_type == "run":
            runs.append(
                {
                    "run_id": span.run_id,
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "name": span.name,
                    "status": span.status,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "duration_ms": span.duration_ms,
                    "tags": span.tags,
                }
            )
    return runs


def _build_global_objects(spans: list[SpanRecord]) -> dict[str, t.Any]:
    """Build a global objects dictionary from all spans."""
    all_objects: dict[str, t.Any] = {}
    for span in spans:
        all_objects.update(span.objects)
    return all_objects


def extract_agents(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract agent-specific data."""
    all_objects = _build_global_objects(spans)

    agents = []
    for span in spans:
        # Only extract actual agent spans (those with agent inputs, not wrapper tasks)
        if span.span_type == "agent" and "agent_id" in span.inputs:
            # Resolve input values from global objects
            agent_id = _resolve_object_value(span.inputs.get("agent_id"), all_objects)
            agent_name = _resolve_object_value(span.inputs.get("agent_name"), all_objects)
            goal = _resolve_object_value(span.inputs.get("goal"), all_objects)
            instructions = _resolve_object_value(span.inputs.get("instructions"), all_objects)
            model = _resolve_object_value(span.inputs.get("model"), all_objects)
            tools = _resolve_object_value(span.inputs.get("tools"), all_objects)

            # Resolve output values
            actual_steps = _resolve_object_value(span.outputs.get("actual_steps"), all_objects)
            stop_reason = _resolve_object_value(span.outputs.get("stop_reason"), all_objects)
            status = _resolve_object_value(span.outputs.get("status"), all_objects)
            input_tokens = _resolve_object_value(span.outputs.get("input_tokens"), all_objects)
            output_tokens = _resolve_object_value(span.outputs.get("output_tokens"), all_objects)
            total_tokens = _resolve_object_value(span.outputs.get("total_tokens"), all_objects)

            agents.append(
                {
                    "run_id": span.run_id,
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name or span.name,
                    "model": model,
                    "goal": goal,
                    "instructions": instructions,
                    "tools": tools or [],
                    "max_steps": span.params.get("max_steps"),
                    "actual_steps": actual_steps,
                    "status": status or span.status,
                    "stop_reason": stop_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "duration_ms": span.duration_ms,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                }
            )
    return agents


def extract_evaluations(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract evaluation-specific data."""
    all_objects = _build_global_objects(spans)

    evaluations = []
    for span in spans:
        # Only extract actual evaluation spans (those with task_name input)
        if span.span_type == "evaluation" and "task_name" in span.inputs:
            # Resolve input values
            task_name = _resolve_object_value(span.inputs.get("task_name"), all_objects)
            scorers = _resolve_object_value(span.inputs.get("scorers"), all_objects)

            # Resolve output values
            dataset_size = _resolve_object_value(span.outputs.get("dataset_size"), all_objects)
            total_samples = _resolve_object_value(span.outputs.get("total_samples"), all_objects)
            passed_count = _resolve_object_value(span.outputs.get("passed_count"), all_objects)
            failed_count = _resolve_object_value(span.outputs.get("failed_count"), all_objects)
            error_count = _resolve_object_value(span.outputs.get("error_count"), all_objects)
            pass_rate = _resolve_object_value(span.outputs.get("pass_rate"), all_objects)
            mean_scores = _resolve_object_value(span.outputs.get("mean_scores"), all_objects)
            stop_reason = _resolve_object_value(span.outputs.get("stop_reason"), all_objects)

            evaluations.append(
                {
                    "run_id": span.run_id,
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "name": span.name,
                    "task_name": task_name,
                    "scorers": scorers or [],
                    "iterations": span.params.get("iterations"),
                    "concurrency": span.params.get("concurrency"),
                    "dataset_size": dataset_size,
                    "total_samples": total_samples,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "error_count": error_count,
                    "pass_rate": pass_rate,
                    "mean_scores": mean_scores,
                    "status": span.status,
                    "stop_reason": stop_reason,
                    "duration_ms": span.duration_ms,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                }
            )
    return evaluations


def extract_studies(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract study-specific data."""
    all_objects = _build_global_objects(spans)

    studies = []
    for span in spans:
        # Only extract actual study spans (those with search_strategy input)
        if span.span_type == "study" and "search_strategy" in span.inputs:
            # Resolve input values
            search_strategy = _resolve_object_value(span.inputs.get("search_strategy"), all_objects)
            objectives = _resolve_object_value(span.inputs.get("objectives"), all_objects)
            directions = _resolve_object_value(span.inputs.get("directions"), all_objects)

            # Resolve output values
            stop_reason = _resolve_object_value(span.outputs.get("stop_reason"), all_objects)
            completed_trials = _resolve_object_value(
                span.outputs.get("completed_trials"), all_objects
            )
            finished_trials = _resolve_object_value(
                span.outputs.get("finished_trials"), all_objects
            )
            failed_trials = _resolve_object_value(span.outputs.get("failed_trials"), all_objects)
            pruned_trials = _resolve_object_value(span.outputs.get("pruned_trials"), all_objects)
            best_trial_index = _resolve_object_value(
                span.outputs.get("best_trial_index"), all_objects
            )
            best_score = _resolve_object_value(span.outputs.get("best_score"), all_objects)
            best_candidate = _resolve_object_value(span.outputs.get("best_candidate"), all_objects)
            best_scores = _resolve_object_value(span.outputs.get("best_scores"), all_objects)

            studies.append(
                {
                    "run_id": span.run_id,
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "name": span.name,
                    "search_strategy": search_strategy,
                    "objectives": objectives or [],
                    "directions": directions or [],
                    "max_trials": span.params.get("max_trials"),
                    "concurrency": span.params.get("concurrency"),
                    "completed_trials": completed_trials,
                    "finished_trials": finished_trials,
                    "failed_trials": failed_trials,
                    "pruned_trials": pruned_trials,
                    "best_trial_index": best_trial_index,
                    "best_score": best_score,
                    "best_candidate": best_candidate,
                    "best_scores": best_scores,
                    "status": span.status,
                    "stop_reason": stop_reason,
                    "duration_ms": span.duration_ms,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                }
            )
    return studies


def extract_tasks(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract task/span hierarchy data."""
    tasks = []
    for span in spans:
        tasks.append(
            {
                "run_id": span.run_id,
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "type": span.span_type,
                "status": span.status,
                "tags": span.tags,
                "duration_ms": span.duration_ms,
                "start_time": span.start_time.isoformat(),
                "end_time": span.end_time.isoformat() if span.end_time else None,
            }
        )
    return tasks


def extract_params(spans: list[SpanRecord]) -> list[dict[str, t.Any]]:
    """Extract all parameters from spans."""
    params = []
    for span in spans:
        for name, value in span.params.items():
            params.append(
                {
                    "run_id": span.run_id,
                    "span_id": span.span_id,
                    "span_type": span.span_type,
                    "name": name,
                    "value": value,
                    "type": type(value).__name__,
                }
            )
    return params


def extract_inputs_outputs(spans: list[SpanRecord]) -> tuple[list[dict], list[dict]]:
    """Extract inputs and outputs with resolved values."""
    all_objects = _build_global_objects(spans)

    inputs = []
    outputs = []

    for span in spans:
        for name, hash_ref in span.inputs.items():
            value = _resolve_object_value(hash_ref, all_objects)
            inputs.append(
                {
                    "run_id": span.run_id,
                    "span_id": span.span_id,
                    "span_type": span.span_type,
                    "name": name,
                    "hash": hash_ref,
                    "value": value,
                }
            )

        for name, hash_ref in span.outputs.items():
            value = _resolve_object_value(hash_ref, all_objects)
            outputs.append(
                {
                    "run_id": span.run_id,
                    "span_id": span.span_id,
                    "span_type": span.span_type,
                    "name": name,
                    "hash": hash_ref,
                    "value": value,
                }
            )

    return inputs, outputs


def write_jsonl(data: list[dict[str, t.Any]], path: Path) -> None:
    """Write data to JSONL file."""
    with open(path, "w") as f:
        f.writelines(json.dumps(item, default=str) + "\n" for item in data)


def convert_traces(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    include_raw: bool = False,
) -> dict[str, Path]:
    """
    Convert spans.jsonl to analysis-friendly tables.

    Args:
        run_dir: Directory containing spans.jsonl
        output_dir: Output directory (defaults to run_dir)
        include_raw: Include raw params/inputs/outputs tables

    Returns:
        Dictionary mapping table name to file path.
    """
    spans_file = run_dir / "spans.jsonl"
    if not spans_file.exists():
        raise FileNotFoundError(f"No spans.jsonl found in {run_dir}")

    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading spans from {spans_file}")
    spans = load_spans(spans_file)
    logger.info(f"Loaded {len(spans)} spans")

    outputs: dict[str, Path] = {}

    # Extract typed tables
    extractors = [
        ("runs", extract_runs),
        ("tasks", extract_tasks),
        ("agents", extract_agents),
        ("evaluations", extract_evaluations),
        ("studies", extract_studies),
    ]

    for name, extractor in extractors:
        data = extractor(spans)
        if data:
            path = output_dir / f"{name}.jsonl"
            write_jsonl(data, path)
            outputs[name] = path
            logger.info(f"Wrote {len(data)} {name} to {path}")

    # Optional: raw params/inputs/outputs
    if include_raw:
        params = extract_params(spans)
        if params:
            path = output_dir / "params.jsonl"
            write_jsonl(params, path)
            outputs["params"] = path

        inputs, outputs_data = extract_inputs_outputs(spans)
        if inputs:
            path = output_dir / "inputs.jsonl"
            write_jsonl(inputs, path)
            outputs["inputs"] = path
        if outputs_data:
            path = output_dir / "outputs.jsonl"
            write_jsonl(outputs_data, path)
            outputs["outputs"] = path

    return outputs
