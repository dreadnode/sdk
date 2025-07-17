import typing as t

if t.TYPE_CHECKING:
    import networkx as nx  # type: ignore [import-untyped]

    from dreadnode.tracing.span import RunSpan


def run_span_to_graph(run: "RunSpan") -> "nx.DiGraph":
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("The `networkx` package is required for graph conversion") from e

    graph = nx.DiGraph()

    graph.add_node(
        run.run_id,
        name=run.name,
        label=run.label,
        start_time=run.start_time,
        end_time=run.end_time,
        duration=run.duration,
        status="failed" if run.failed else "running" if run.is_recording else "completed",
        tags=run.tags,
    )

    for task in run.all_tasks:
        graph.add_node(
            task.span_id,
            name=task.name,
            label=task.label,
            start_time=task.start_time,
            end_time=task.end_time,
            duration=task.duration,
            status="failed" if task.failed else "running" if task.active else "completed",
            tags=task.tags,
        )

        graph.add_edge(task.parent_task.span_id if task.parent_task else run.run_id, task.span_id)

    return graph
