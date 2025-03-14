from .main import DEFAULT_INSTANCE
from .score import Scorer
from .task import Task
from .tracing import RunSpan, Span, TaskSpan
from .version import VERSION

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown

api = DEFAULT_INSTANCE.api
span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
run = DEFAULT_INSTANCE.run
scorer = DEFAULT_INSTANCE.scorer

log_metric = DEFAULT_INSTANCE.log_metric
log_param = DEFAULT_INSTANCE.log_param
log_params = DEFAULT_INSTANCE.log_params
log_input = DEFAULT_INSTANCE.log_input
log_inputs = DEFAULT_INSTANCE.log_inputs
log_output = DEFAULT_INSTANCE.log_output

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
