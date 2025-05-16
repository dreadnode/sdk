from logging import getLogger

from .models import (
    Object,
    ObjectUri,
    ObjectVal,
    RawObjectVal,
    RawRun,
    RawTask,
    Run,
    Task,
    TaskTree,
    TraceSpan,
    TraceTree,
    V0Object,
)

logger = getLogger(__name__)


def convert_flat_tasks_to_tree(tasks: list[Task]) -> list[TaskTree]:
    tree_nodes: dict[str, TaskTree] = {}
    for task in tasks:
        tree_nodes[task.span_id] = TaskTree(task=task)

    roots: list[TaskTree] = []
    for task in tasks:
        if task.parent_task_span_id not in tree_nodes:
            roots.append(tree_nodes[task.span_id])
        else:
            parent_node = tree_nodes.get(task.parent_task_span_id)
            if parent_node:
                parent_node.children.append(tree_nodes[task.span_id])

    return roots


def convert_flat_trace_to_tree(trace: list[Task | TraceSpan]) -> list[TraceTree]:
    tree_nodes: dict[str, TraceTree] = {}
    for span in trace:
        tree_nodes[span.span_id] = TraceTree(span=span)

    roots: list[TraceTree] = []
    for span in trace:
        if span.parent_span_id not in tree_nodes:
            roots.append(tree_nodes[span.span_id])
        else:
            parent_node = tree_nodes.get(span.parent_span_id)
            if parent_node:
                parent_node.children.append(tree_nodes[span.span_id])

    return roots


def process_run(run: RawRun) -> Run:
    inputs: dict[str, Object] = {}
    outputs: dict[str, Object] = {}

    for references, converted in ((run.inputs, inputs), (run.outputs, outputs)):
        for ref in references:
            if (_object := run.objects.get(ref.hash)) is None:
                if run.status != "pending":  # In-progress runs may not have all the objects ready
                    logger.error("Object %s not found in run %s", ref.hash, run.id)
                continue

            if (_schema := run.object_schemas.get(_object.schema_hash)) is None:
                if run.status != "pending":
                    logger.error("Schema for object %s not found in run %s", ref.hash, run.id)
                continue

            if isinstance(_object, RawObjectVal):
                converted[ref.name] = ObjectVal(
                    name=ref.name,
                    label=ref.label,
                    hash=ref.hash,
                    schema_=_schema,
                    schema_hash=_object.schema_hash,
                    value=_object.value,
                )
            else:
                converted[ref.name] = ObjectUri(
                    name=ref.name,
                    label=ref.label,
                    hash=ref.hash,
                    schema_=_schema,
                    schema_hash=_object.schema_hash,
                    uri=_object.uri,
                    size=_object.size,
                )

    return Run(
        id=run.id,
        name=run.name,
        span_id=run.span_id,
        trace_id=run.trace_id,
        timestamp=run.timestamp,
        duration=run.duration,
        status=run.status,
        exception=run.exception,
        tags=run.tags,
        params=run.params,
        metrics=run.metrics,
        inputs=inputs,
        outputs=outputs,
        artifacts=run.artifacts,
        schema=run.schema_,
    )


def process_task(task: RawTask, run: RawRun) -> Task:
    inputs: dict[str, Object] = {}
    outputs: dict[str, Object] = {}

    for references, converted in ((task.inputs, inputs), (task.outputs, outputs)):
        for ref in references:
            if isinstance(ref, V0Object):
                converted[ref.name] = ObjectVal(
                    name=ref.name,
                    label=ref.label,
                    hash="",
                    schema_={},
                    schema_hash="",
                    value=ref.value,
                )
                continue

            if (_object := run.objects.get(ref.hash)) is None:
                if run.status != "pending":
                    logger.error("Object %s not found in run %s", ref.hash, run.id)
                continue

            if (_schema := run.object_schemas.get(_object.schema_hash)) is None:
                if run.status != "pending":
                    logger.error("Schema for object %s not found in run %s", ref.hash, run.id)
                continue

            if isinstance(_object, RawObjectVal):
                converted[ref.name] = ObjectVal(
                    name=ref.name,
                    label=ref.label,
                    hash=ref.hash,
                    schema_=_schema,
                    schema_hash=_object.schema_hash,
                    value=_object.value,
                )
            else:
                converted[ref.name] = ObjectUri(
                    name=ref.name,
                    label=ref.label,
                    hash=ref.hash,
                    schema_=_schema,
                    schema_hash=_object.schema_hash,
                    uri=_object.uri,
                    size=_object.size,
                )

    return Task(
        name=task.name,
        span_id=task.span_id,
        trace_id=task.trace_id,
        parent_span_id=task.parent_span_id,
        parent_task_span_id=task.parent_task_span_id,
        timestamp=task.timestamp,
        duration=task.duration,
        status=task.status,
        exception=task.exception,
        tags=task.tags,
        params=task.params,
        metrics=task.metrics,
        inputs=inputs,
        outputs=outputs,
        schema=task.schema_,
        attributes=task.attributes,
        resource_attributes=task.resource_attributes,
        events=task.events,
        links=task.links,
    )
