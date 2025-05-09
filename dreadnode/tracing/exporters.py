"""
This module provides classes for exporting metrics, spans, and logs to JSONL files
in OTLP format. It includes configurations for file paths and exporters for
different telemetry signals.
"""

import threading
import typing as t
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from google.protobuf import json_format
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

from dreadnode.util import logger


@dataclass
class FileExportConfig:
    """Configuration for signal exports to JSONL files.

    Attributes:
        base_path (str | Path): The base directory where files will be stored.
        prefix (str): A prefix to add to the filenames.
    """

    base_path: str | Path = Path.cwd() / ".dreadnode"
    prefix: str = ""

    def get_path(self, signal: str) -> Path:
        """Get the file path for a specific signal type.

        Args:
            signal (str): The type of signal (e.g., "metrics", "traces", "logs").

        Returns:
            Path: The full path to the file for the given signal type.
        """
        base = Path(self.base_path)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.prefix}{signal}.jsonl"


class FileMetricReader(MetricReader):
    """MetricReader that writes metrics to a file in OTLP format.

    Attributes:
        config (FileExportConfig): Configuration for file export.
    """

    def __init__(self, config: FileExportConfig):
        """Initializes the FileMetricReader.

        Args:
            config (FileExportConfig): Configuration for file export.
        """
        super().__init__()
        self.config = config
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        """Lazily opens and returns the file for writing metrics.

        Returns:
            IO[str]: The file object for writing metrics.
        """
        if not self._file:
            self._file = self.config.get_path("metrics").open("a")
        return self._file

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,  # noqa: ARG002
        **kwargs: t.Any,  # noqa: ARG002
    ) -> None:
        """Receives and writes metrics data to the file.

        Args:
            metrics_data (MetricsData): The metrics data to write.
            timeout_millis (float): Timeout in milliseconds (unused).
            **kwargs: Additional arguments (unused).
        """
        if metrics_data is None:
            return

        try:
            encoded = encode_metrics(metrics_data)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()
        except Exception as e:  # noqa: BLE001
            # BLE001: Catching Exception is generally discouraged, but here we want to catch all exceptions
            logger.error(f"Failed to export metrics: {e}")

    def shutdown(
        self,
        timeout_millis: float = 30_000,  # noqa: ARG002
        **kwargs: t.Any,  # noqa: ARG002
    ) -> None:
        """Shuts down the metric reader and closes the file.

        Args:
            timeout_millis (float): Timeout in milliseconds (unused).
            **kwargs: Additional arguments (unused).
        """
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class FileSpanExporter(SpanExporter):
    """SpanExporter that writes spans to a file in OTLP format.

    Attributes:
        config (FileExportConfig): Configuration for file export.
    """

    def __init__(self, config: FileExportConfig):
        """Initializes the FileSpanExporter.

        Args:
            config (FileExportConfig): Configuration for file export.
        """
        self.config = config
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        """Lazily opens and returns the file for writing spans.

        Returns:
            IO[str]: The file object for writing spans.
        """
        if not self._file:
            self._file = self.config.get_path("traces").open("a")
        return self._file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Exports spans to the file.

        Args:
            spans (Sequence[ReadableSpan]): The spans to export.

        Returns:
            SpanExportResult: The result of the export operation.
        """
        try:
            encoded = encode_spans(spans)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()
        except Exception as e:  # noqa: BLE001
            # BLE001: Catching Exception is generally discouraged, but here we want to catch all exceptions
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 30_000) -> bool:  # noqa: ARG002
        """Forces a flush of the exporter.

        Args:
            timeout_millis (float): Timeout in milliseconds (unused).

        Returns:
            bool: Always returns True as the file is flushed during export.
        """
        return True

    def shutdown(self) -> None:
        """Shuts down the span exporter and closes the file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class FileLogExporter(LogExporter):
    """LogExporter that writes logs to a file in OTLP format.

    Attributes:
        config (FileExportConfig): Configuration for file export.
    """

    def __init__(self, config: FileExportConfig):
        """Initializes the FileLogExporter.

        Args:
            config (FileExportConfig): Configuration for file export.
        """
        self.config = config
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        """Lazily opens and returns the file for writing logs.

        Returns:
            IO[str]: The file object for writing logs.
        """
        if not self._file:
            self._file = self.config.get_path("logs").open("a")
        return self._file

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        """Exports logs to the file.

        Args:
            batch (Sequence[LogData]): The logs to export.

        Returns:
            LogExportResult: The result of the export operation.
        """
        try:
            encoded = encode_logs(batch)
            json_str = json_format.MessageToJson(encoded, indent=None)
            with self._lock:
                self.file.write(json_str + "\n")
                self.file.flush()
        except Exception as e:  # noqa: BLE001
            # BLE001: Catching Exception is generally discouraged, but here we want to catch all exceptions
            logger.error(f"Failed to export logs: {e}")
            return LogExportResult.FAILURE
        return LogExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 30_000) -> bool:  # noqa: ARG002
        """Forces a flush of the exporter.

        Args:
            timeout_millis (float): Timeout in milliseconds (unused).

        Returns:
            bool: Always returns True as the file is flushed during export.
        """
        return True

    def shutdown(self) -> None:
        """Shuts down the log exporter and closes the file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
