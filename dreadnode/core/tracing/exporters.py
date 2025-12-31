import json
import threading
import typing as t
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO

from google.protobuf import json_format
from loguru import logger
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.metrics_encoder import encode_metrics
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk._logs.export import LogExporter, LogExportResult
from opentelemetry.sdk.metrics.export import (
    MetricReader,
    MetricsData,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from dreadnode.core.storage import Storage


@dataclass
class TraceExportConfig:
    """Configuration for trace exports to Storage.

    All signals are exported as JSONL during execution for robustness.
    Parquet conversion can be done afterwards for analysis.
    """

    storage: Storage
    run_id: str
    _metrics_file: IO[str] | None = field(default=None, repr=False)
    _trajectories_file: IO[str] | None = field(default=None, repr=False)
    _artifacts_file: IO[str] | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_path(self, signal: str, ext: str = "jsonl") -> Path:
        """Get the file path for a specific signal type."""
        return self.storage.trace_path(self.run_id, f"{signal}.{ext}")

    def write_metric(self, metric: dict[str, t.Any]) -> None:
        """Write a metric to metrics.jsonl."""
        with self._lock:
            if self._metrics_file is None:
                self._metrics_file = self.get_path("metrics").open("a")
            self._metrics_file.write(json.dumps(metric) + "\n")
            self._metrics_file.flush()

    def write_trajectory(self, trajectory: dict[str, t.Any]) -> None:
        """Write a trajectory to trajectories.jsonl."""
        with self._lock:
            if self._trajectories_file is None:
                self._trajectories_file = self.get_path("trajectories").open("a")
            self._trajectories_file.write(json.dumps(trajectory) + "\n")
            self._trajectories_file.flush()

    def write_artifact(self, artifact: dict[str, t.Any]) -> None:
        """Write artifact metadata to artifacts.jsonl."""
        with self._lock:
            if self._artifacts_file is None:
                self._artifacts_file = self.get_path("artifacts").open("a")
            self._artifacts_file.write(json.dumps(artifact, default=str) + "\n")
            self._artifacts_file.flush()

    def extract_metrics_from_span(self, span: ReadableSpan) -> None:
        """Extract metrics from span attributes and write to metrics.jsonl."""
        # Look for dreadnode.metrics attribute
        metrics_attr = None
        for attr in span.attributes or {}:
            if attr == "dreadnode.metrics":
                metrics_attr = span.attributes[attr]
                break

        if not metrics_attr:
            return

        try:
            metrics_data = json.loads(metrics_attr) if isinstance(metrics_attr, str) else metrics_attr
            span_name = span.name
            span_id = format(span.context.span_id, "016x") if span.context else None
            run_id = self.run_id

            for metric_name, metric_info in metrics_data.items():
                # Handle both simple values and structured metrics
                if isinstance(metric_info, dict):
                    values = metric_info.get("values", [])
                    steps = metric_info.get("steps", [])
                    timestamps = metric_info.get("timestamps", [])

                    for i, value in enumerate(values):
                        self.write_metric({
                            "run_id": run_id,
                            "span_id": span_id,
                            "span_name": span_name,
                            "metric_name": metric_name,
                            "value": float(value) if value is not None else None,
                            "step": steps[i] if i < len(steps) else 0,
                            "timestamp": timestamps[i] if i < len(timestamps) else None,
                        })
                else:
                    self.write_metric({
                        "run_id": run_id,
                        "span_id": span_id,
                        "span_name": span_name,
                        "metric_name": metric_name,
                        "value": float(metric_info) if metric_info is not None else None,
                        "step": 0,
                        "timestamp": datetime.now().isoformat(),
                    })
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to extract metrics from span: {e}")

    def shutdown(self) -> None:
        """Close any open file handles."""
        with self._lock:
            if self._metrics_file is not None:
                self._metrics_file.close()
                self._metrics_file = None
            if self._trajectories_file is not None:
                self._trajectories_file.close()
                self._trajectories_file = None
            if self._artifacts_file is not None:
                self._artifacts_file.close()
                self._artifacts_file = None


# For backwards compatibility
FileExportConfig = TraceExportConfig


class FileMetricReader(MetricReader):
    """MetricReader that writes metrics to a file in OTLP format."""

    def __init__(self, config: TraceExportConfig):
        super().__init__()
        self.config = config
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        if not self._file:
            self._file = self.config.get_path("metrics").open("a")
        return self._file

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,  # noqa: ARG002
        **kwargs: t.Any,  # noqa: ARG002
    ) -> None:
        if metrics_data is None:
            return

        try:
            encoded = encode_metrics(metrics_data)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to export metrics: {e}")

    def shutdown(
        self,
        timeout_millis: float = 30_000,  # noqa: ARG002
        **kwargs: t.Any,  # noqa: ARG002
    ) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class FileSpanExporter(SpanExporter):
    """SpanExporter that writes spans to a file in OTLP format."""

    def __init__(self, config: TraceExportConfig):
        self.config = config
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        if not self._file:
            self._file = self.config.get_path("spans").open("a")
        return self._file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            encoded = encode_spans(spans)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()

            # Extract metrics from spans for analysis-friendly export
            for span in spans:
                self.config.extract_metrics_from_span(span)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(
        self,
        timeout_millis: float = 30_000,  # noqa: ARG002
    ) -> bool:
        return True  # We flush above

    def shutdown(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

        # Close metrics/trajectories file handles
        self.config.shutdown()


class FileLogExporter(LogExporter):
    """LogExporter that writes logs to a file in OTLP format."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        if not self._file:
            self._file = self.config.get_path("logs").open("a")
        return self._file

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            encoded = encode_logs(batch)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to export logs: {e}")
            return LogExportResult.FAILURE
        return LogExportResult.SUCCESS

    def force_flush(
        self,
        timeout_millis: float = 30_000,  # noqa: ARG002
    ) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
