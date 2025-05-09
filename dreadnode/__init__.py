"""Dreadnode SDK.

This module serves as the main entry point for the Dreadnode SDK. It provides
core functionality for configuring, managing tasks, logging metrics, and
handling spans for tracing operations. The SDK is designed to facilitate
interaction with the Dreadnode platform.

Attributes:
    configure (function): Configures the Dreadnode instance.
    shutdown (function): Shuts down the Dreadnode instance.
    api (object): Provides access to the API interface of the Dreadnode instance.
    span (object): Manages tracing spans.
    task (object): Handles task-related operations.
    task_span (object): Manages task-specific spans.
    run (object): Represents the current run context.
    scorer (object): Provides scoring functionality.
    push_update (function): Pushes updates to the Dreadnode instance.
    log_metric (function): Logs a metric to the Dreadnode instance.
    log_param (function): Logs a single parameter to the Dreadnode instance.
    log_params (function): Logs multiple parameters to the Dreadnode instance.
    log_input (function): Logs a single input to the Dreadnode instance.
    log_inputs (function): Logs multiple inputs to the Dreadnode instance.
    log_output (function): Logs an output to the Dreadnode instance.
    link_objects (function): Links objects in the Dreadnode instance.
    log_artifact (function): Logs an artifact to the Dreadnode instance.
    __version__ (str): The version of the Dreadnode SDK.

Classes:
    Dreadnode: The main class for interacting with the Dreadnode platform.
    Metric: Represents a metric in the Dreadnode SDK.
    MetricDict: A dictionary-like structure for managing metrics.
    Scorer: Provides scoring capabilities.
    Object: Represents an object in the Dreadnode SDK.
    Task: Represents a task in the Dreadnode SDK.
    Span: Represents a tracing span.
    RunSpan: Represents a span for a run.
    TaskSpan: Represents a span for a task.

Example:
    Import the module and use its functionality as follows:

        from dreadnode import configure, log_metric

        configure(api_key="your_api_key")
        log_metric(name="example_metric", value=42)
"""

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
