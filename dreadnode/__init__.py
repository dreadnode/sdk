import importlib
import typing as t

from loguru import logger

from dreadnode import (
    convert,
    meta,
)
from dreadnode import logging_ as logging
from dreadnode.data_types import Code, Markdown, Object3D, Text
from dreadnode.logging_ import configure_logging
from dreadnode.main import DEFAULT_INSTANCE, Dreadnode
from dreadnode.meta import (
    Config,
    CurrentRun,
    CurrentTask,
    CurrentTrial,
    DatasetField,
    EnvVar,
    ParentTask,
    RunInput,
    RunOutput,
    RunParam,
    TaskInput,
    TaskOutput,
    TrialCandidate,
    TrialOutput,
    TrialScore,
)
from dreadnode.metric import Metric, MetricDict
from dreadnode.object import Object
from dreadnode.scorers import Scorer
from dreadnode.task import Task
from dreadnode.tracing.span import RunSpan, Span, TaskSpan
from dreadnode.version import VERSION

if t.TYPE_CHECKING:
    from dreadnode import agent, airt, eval, optimization, scorers, transforms  # noqa: A004
    from dreadnode.agent import Agent, tool, tool_method
    from dreadnode.data_types import Audio, Image, Table, Video

logger.disable("dreadnode")

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown

api = DEFAULT_INSTANCE.api
span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
task_span = DEFAULT_INSTANCE.task_span
run = DEFAULT_INSTANCE.run
task_and_run = DEFAULT_INSTANCE.task_and_run
scorer = DEFAULT_INSTANCE.scorer
score = DEFAULT_INSTANCE.score
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
log_sample = DEFAULT_INSTANCE.log_sample
log_samples = DEFAULT_INSTANCE.log_samples
link_objects = DEFAULT_INSTANCE.link_objects
log_artifact = DEFAULT_INSTANCE.log_artifact
get_current_run = DEFAULT_INSTANCE.get_current_run
get_current_task = DEFAULT_INSTANCE.get_current_task

__version__ = VERSION

__all__ = [
    "DEFAULT_INSTANCE",
    "Agent",
    "Audio",
    "Code",
    "Config",
    "CurrentRun",
    "CurrentTask",
    "CurrentTrial",
    "DatasetField",
    "Dreadnode",
    "EnvVar",
    "Image",
    "Markdown",
    "Metric",
    "MetricDict",
    "Object",
    "Object3D",
    "ParentTask",
    "RunInput",
    "RunOutput",
    "RunParam",
    "RunSpan",
    "Scorer",
    "Span",
    "Table",
    "Task",
    "TaskInput",
    "TaskOutput",
    "TaskSpan",
    "Text",
    "TrialCandidate",
    "TrialOutput",
    "TrialScore",
    "Video",
    "__version__",
    "agent",
    "airt",
    "api",
    "configure",
    "configure_logging",
    "continue_run",
    "convert",
    "eval",
    "get_run_context",
    "link_objects",
    "log_artifact",
    "log_input",
    "log_inputs",
    "log_metric",
    "log_output",
    "log_param",
    "log_params",
    "logging",
    "meta",
    "optimization",
    "push_update",
    "run",
    "scorer",
    "scorers",
    "shutdown",
    "span",
    "tag",
    "task",
    "task_and_run",
    "task_span",
    "tool",
    "tool_method",
    "transforms",
]

__lazy_submodules__: list[str] = ["scorers", "agent", "airt", "eval", "transforms", "optimization"]
__lazy_components__: dict[str, str] = {
    "Audio": "dreadnode.data_types",
    "Image": "dreadnode.data_types",
    "Table": "dreadnode.data_types",
    "Video": "dreadnode.data_types",
    "Agent": "dreadnode.agent",
    "tool": "dreadnode.agent",
    "tool_method": "dreadnode.agent",
}


def __getattr__(name: str) -> t.Any:
    if name in __lazy_submodules__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in __lazy_components__:
        module_name = __lazy_components__[name]
        module = importlib.import_module(module_name)
        component = getattr(module, name)
        globals()[name] = component
        return component

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
