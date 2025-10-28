import typing as t

from dreadnode.agent.events import AgentEvent, ToolEnd, ToolStart
from dreadnode.agent.hooks import Hook
from dreadnode.meta import Config, component

if t.TYPE_CHECKING:
    from datetime import datetime


def tool_metrics(*, detailed: bool = False) -> Hook:
    """
    Creates an agent hook to log metrics about tool usage, execution time, and success rates.

    Args:
        detailed: If True, logs metrics for each specific tool in addition to general stats.
                  If False, only logs aggregate statistics across all tools.

    Returns:
        An async hook function that can be registered with an agent.
    """
    _start_times: dict[str, datetime] = {}

    @component
    async def tool_metrics(
        event: AgentEvent,
        *,
        detailed: bool = Config(
            default=detailed,
            help="If True, logs metrics for each specific tool in addition to general stats.",
        ),
    ) -> None:
        """The actual hook implementation that processes agent events."""
        from dreadnode import log_metric

        if isinstance(event, ToolStart):
            log_metric("tool/total_count", 1, step=event.step, mode="count")
            _start_times[event.tool_call.id] = event.timestamp

            if detailed:
                tool_name = event.tool_call.name
                log_metric(f"tool/count.{tool_name}", 1, step=event.step, mode="count")

        elif isinstance(event, ToolEnd):
            tool_name = event.tool_call.name
            start_time = _start_times.pop(event.tool_call.id, event.timestamp)
            duration_seconds = (event.timestamp - start_time).total_seconds()
            errored = "error" in event.message.metadata

            log_metric("tool/total_time", duration_seconds, step=event.step, mode="sum")
            log_metric("tool/success_rate", 0 if errored else 1, step=event.step, mode="avg")

            if errored:
                log_metric("tool/failed_count", 1, step=event.step, mode="count")

            if detailed:
                log_metric(
                    f"tool/time.{tool_name}",
                    duration_seconds,
                    step=event.step,
                    mode="sum",
                )
                log_metric(
                    f"tool/avg_time.{tool_name}",
                    duration_seconds,
                    step=event.step,
                    mode="avg",
                )
                log_metric(
                    f"tool/success_rate.{tool_name}",
                    0 if errored else 1,
                    step=event.step,
                    mode="avg",
                )

                if errored:
                    log_metric(
                        f"tool/failed_count.{tool_name}",
                        1,
                        step=event.step,
                        mode="count",
                    )

    return tool_metrics
