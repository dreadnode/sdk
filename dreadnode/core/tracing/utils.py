from opentelemetry.trace import Tracer

from dreadnode.core.tracing.span import RunSpan, TaskSpan


def get_default_tracer() -> Tracer:
    """Get the default tracer from the default Dreadnode instance."""
    from dreadnode import DEFAULT_INSTANCE

    return DEFAULT_INSTANCE.get_tracer()


def get_current_task_span() -> TaskSpan | None:
    """Get the current tracer from the current span."""
    from dreadnode.core.tracing.span import current_task_span

    task_span = current_task_span
    if task_span is not None:
        return task_span
    return None


def get_current_run_span() -> RunSpan | None:
    """Get the current run span from the current task span."""
    from dreadnode.core.tracing.span import current_run_span

    run_span = current_run_span
    if run_span is not None:
        return run_span
    return None
