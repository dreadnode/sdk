from dreadnode import convert, data_types, scorers
from dreadnode.data_types import Audio, Code, Image, Markdown, Object3D, Table, Text, Video
from dreadnode.main import DEFAULT_INSTANCE, Dreadnode
from dreadnode.metric import Metric, MetricDict, Scorer
from dreadnode.object import Object
from dreadnode.task import Task, TaskInput
from dreadnode.tracing.span import RunSpan, Span, TaskSpan
from dreadnode.version import VERSION

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown

api = DEFAULT_INSTANCE.api
span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
task_span = DEFAULT_INSTANCE.task_span
run = DEFAULT_INSTANCE.run
scorer = DEFAULT_INSTANCE.scorer
push_update = DEFAULT_INSTANCE.push_update
tag = DEFAULT_INSTANCE.tag
get_run_context = DEFAULT_INSTANCE.get_run_context
continue_run = DEFAULT_INSTANCE.continue_run
log_metric = DEFAULT_INSTANCE.log_metric
log_metrics = DEFAULT_INSTANCE.log_metrics
log_param = DEFAULT_INSTANCE.log_param
log_params = DEFAULT_INSTANCE.log_params
log_input = DEFAULT_INSTANCE.log_input
log_inputs = DEFAULT_INSTANCE.log_inputs
log_output = DEFAULT_INSTANCE.log_output
log_outputs = DEFAULT_INSTANCE.log_outputs
link_objects = DEFAULT_INSTANCE.link_objects
log_artifact = DEFAULT_INSTANCE.log_artifact

__version__ = VERSION

__all__ = [
    "DEFAULT_INSTANCE",
    "Audio",
    "Code",
    "Dreadnode",
    "Image",
    "Markdown",
    "Metric",
    "MetricDict",
    "Object",
    "Object3D",
    "Run",
    "RunSpan",
    "Scorer",
    "Span",
    "Table",
    "Task",
    "TaskInput",
    "TaskSpan",
    "Text",
    "Video",
    "__version__",
    "api",
    "configure",
    "continue_run",
    "convert",
    "data_types",
    "get_run_context",
    "link_objects",
    "log_artifact",
    "log_input",
    "log_inputs",
    "log_metric",
    "log_output",
    "log_param",
    "log_params",
    "push_update",
    "run",
    "scorer",
    "scorers",
    "shutdown",
    "span",
    "tag",
    "task",
    "task_span",
    "task_span",
]
