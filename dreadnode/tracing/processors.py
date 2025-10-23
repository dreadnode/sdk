import threading
import typing as t
from urllib.parse import urljoin

from logfire._internal.exporters.dynamic_batch import DynamicBatchSpanProcessor
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from dreadnode.tracing.constants import SPAN_RESOURCE_ATTRIBUTE_TOKEN
from dreadnode.version import VERSION

if t.TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Span


class RoutingSpanProcessor(SpanProcessor):
    """
    A SpanProcessor that routes spans to different BatchSpanProcessors based on
    a token attached to the span object.

    This allows a single application to export spans for multiple users/tokens
    to the same backend.
    """

    def __init__(
        self,
        server_url: str,
        default_token: str,
        *,
        token_header_name: str = "X-Api-Key",  # noqa: S107
        span_token_attribute_name: str = SPAN_RESOURCE_ATTRIBUTE_TOKEN,
    ):
        self._server_url = server_url
        self._default_token = default_token
        self._token_header_name = token_header_name
        self._span_token_attribute_name = span_token_attribute_name
        self._processors: dict[str, SpanProcessor] = {}
        self._lock = threading.Lock()
        self._get_or_create_processor(self._default_token)

    def _get_or_create_processor(self, token: str) -> SpanProcessor:
        """Lazily creates and caches a BatchSpanProcessor for a given token."""
        with self._lock:
            if token not in self._processors:
                headers = {"User-Agent": f"dreadnode/{VERSION}", self._token_header_name: token}
                self._processors[token] = DynamicBatchSpanProcessor(
                    RemovePendingSpansExporter(
                        OTLPSpanExporter(
                            endpoint=urljoin(self._server_url, "/api/otel/traces"),
                            headers=headers,
                            compression=Compression.Gzip,
                        )
                    )
                )
            return self._processors[token]

    def on_start(self, span: "Span", parent_context: "Context | None" = None) -> None:
        """No-op. Spans are routed on end."""

    def on_end(self, span: ReadableSpan) -> None:
        """Routes the span to the correct processor based on its token."""
        # We use the resource here to prevent it from being lost during conversions
        token = getattr(span.resource, self._span_token_attribute_name, self._default_token)
        processor = self._get_or_create_processor(token)
        processor.on_end(span)

    def shutdown(self) -> None:
        """Shuts down all managed processors."""
        with self._lock:
            for processor in self._processors.values():
                processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flushes all managed processors."""
        with self._lock:
            # OTel spec says this should return True only if all flushes succeed.
            results = [p.force_flush(timeout_millis) for p in self._processors.values()]
            return all(results)
