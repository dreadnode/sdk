from .main import DEFAULT_INSTANCE
from .score import Scorer
from .task import Task
from .tracing import RunSpan, Score, Span, TaskSpan
from .version import VERSION

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown

api = DEFAULT_INSTANCE.api
span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
task_span = DEFAULT_INSTANCE.task_span
run = DEFAULT_INSTANCE.run
push_update = DEFAULT_INSTANCE.push_update

log_metric = DEFAULT_INSTANCE.log_metric
log_param = DEFAULT_INSTANCE.log_param
log_params = DEFAULT_INSTANCE.log_params
log_score = DEFAULT_INSTANCE.log_score

__version__ = VERSION

__all__ = [
    "configure",
    "shutdown",
    "span",
    "task",
    "run",
    "log_metric",
    "log_param",
    "Run",
    "Task",
    "Scorer",
    "Score",
    "TaskSpan",
    "Span",
    "RunSpan",
    "__version__",
]
