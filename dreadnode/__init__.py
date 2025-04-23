from dreadnode.main import DEFAULT_INSTANCE, Dreadnode
from dreadnode.metric import Metric, MetricDict, Scorer
from dreadnode.object import Object
from dreadnode.task import Task
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
task_span = DEFAULT_INSTANCE.task_span
push_update = DEFAULT_INSTANCE.push_update

log_metric = DEFAULT_INSTANCE.log_metric
log_param = DEFAULT_INSTANCE.log_param
log_params = DEFAULT_INSTANCE.log_params
log_input = DEFAULT_INSTANCE.log_input
log_inputs = DEFAULT_INSTANCE.log_inputs
log_output = DEFAULT_INSTANCE.log_output
link_objects = DEFAULT_INSTANCE.link_objects
log_artifact = DEFAULT_INSTANCE.log_artifact

__version__ = VERSION

__all__ = [
    "Dreadnode",
    "Metric",
    "MetricDict",
    "Object",
    "Run",
    "RunSpan",
    "Score",
    "Scorer",
    "Span",
    "Task",
    "TaskSpan",
    "__version__",
    "configure",
    "log_metric",
    "log_param",
    "run",
    "shutdown",
    "span",
    "task",
]
