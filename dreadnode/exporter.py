import typing as t

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from dreadnode.constants import DEFAULT_USER_AGENT


class CustomOTLPSpanExporter(OTLPSpanExporter):
    """A custom OTLP exporter that injects our SDK version into the User-Agent."""

    def __init__(self, **kwargs: t.Any) -> None:
        super().__init__(**kwargs)

        # 2. Get the current User-Agent set by OTel (e.g., OTel-OTLP-Exporter-Python/<version>)
        otlp_user_agent = self._session.headers.get("User-Agent")
        if isinstance(otlp_user_agent, bytes):
            otlp_user_agent = otlp_user_agent.decode("utf-8")

        # 3. Combine the User-Agent strings.
        if otlp_user_agent:
            combined_user_agent = f"{DEFAULT_USER_AGENT} {otlp_user_agent}"
            self._session.headers["User-Agent"] = combined_user_agent
        else:
            # Fallback if somehow OTel didn't set a User-Agent
            self._session.headers["User-Agent"] = DEFAULT_USER_AGENT
