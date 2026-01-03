import importlib
import typing as t

from loguru import logger

from dreadnode.core import log as logging
from dreadnode.core import meta
from dreadnode.core.agents.scorer_hook import ScorerHook, ScorerHookResult
from dreadnode.core.load import load
from dreadnode.core.log import configure_logging
from dreadnode.core.meta import (
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
from dreadnode.core.metric import Metric, MetricDict
from dreadnode.core.object import Object
from dreadnode.core.scorer import Scorer
from dreadnode.core.task import Task
from dreadnode.core.tools import tool, tool_method
from dreadnode.core.tracing import convert
from dreadnode.core.tracing.span import RunSpan, Span, TaskSpan
from dreadnode.core.tracing.spans import (
    agent_span,
    evaluation_span,
    generation_span,
    sample_span,
    scorer_span,
    study_span,
    tool_span,
    trial_span,
)
from dreadnode.core.transforms import Transform
from dreadnode.core.types import Code, Markdown, Object3D, Text
from dreadnode.datasets.dataset import Dataset
from dreadnode.evaluations import Evaluation
from dreadnode.main import DEFAULT_INSTANCE, Dreadnode
from dreadnode.version import VERSION

# Rebuild models to resolve forward references after all imports
Evaluation.model_rebuild()

if t.TYPE_CHECKING:
    from dreadnode.core.agents import Agent
    from dreadnode.core.types import Audio, Image, Table, Video

logger.disable("dreadnode")

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown
span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
task_span = DEFAULT_INSTANCE.task_span
run = DEFAULT_INSTANCE.run
task_and_run = DEFAULT_INSTANCE.task_and_run
scorer = DEFAULT_INSTANCE.scorer
agent = DEFAULT_INSTANCE.agent
evaluation = DEFAULT_INSTANCE.evaluation
study = DEFAULT_INSTANCE.study
push_update = DEFAULT_INSTANCE.push_update
init_repo = DEFAULT_INSTANCE.init_package
push_repo = DEFAULT_INSTANCE.push_package
pull_repo = DEFAULT_INSTANCE.pull_package
build_repo = DEFAULT_INSTANCE.build_package
init_package = DEFAULT_INSTANCE.init_package
push_package = DEFAULT_INSTANCE.push_package
pull_package = DEFAULT_INSTANCE.pull_package
build_package = DEFAULT_INSTANCE.build_package
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
train = DEFAULT_INSTANCE.train

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
    "Dataset",
    "DatasetField",
    "Dreadnode",
    "EnvVar",
    "Evaluation",
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
    "ScorerHook",
    "ScorerHookResult",
    "Span",
    "Table",
    "Task",
    "TaskInput",
    "TaskOutput",
    "TaskSpan",
    "Text",
    "Transform",
    "TrialCandidate",
    "TrialOutput",
    "TrialScore",
    "Video",
    "__version__",
    "agent",
    "agent_span",
    "airt",
    "api",
    "configure",
    "configure_logging",
    "continue_run",
    "convert",
    "evaluation",
    "evaluation_span",
    "generation_span",
    "get_run_context",
    "link_objects",
    "load",
    "load_dataset",
    "log_artifact",
    "log_input",
    "log_inputs",
    "log_metric",
    "log_output",
    "log_param",
    "log_params",
    "logging",
    "meta",
    "push_dataset",
    "push_update",
    "run",
    "sample_span",
    "scorer",
    "scorer_span",
    "scorers",
    "shutdown",
    "span",
    "study",
    "study_span",
    "tag",
    "task",
    "task_and_run",
    "task_span",
    "tool",
    "tool_method",
    "tool_span",
    "train",
    "training",
    "transforms",
    "trial_span",
]

__lazy_submodules__: list[str] = ["scorers", "agents", "airt", "eval", "transforms", "training"]
__lazy_components__: dict[str, str] = {
    "Audio": "dreadnode.core.types",
    "Image": "dreadnode.core.types",
    "Table": "dreadnode.core.types",
    "Video": "dreadnode.core.types",
    "Agent": "dreadnode.agents",
    "tool": "dreadnode.tools",
    "tool_method": "dreadnode.tools.tool_method",
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
