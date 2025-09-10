import importlib
import typing as t

from dreadnode.agent.tools import planning, reporting, tasking
from dreadnode.agent.tools.base import (
    AnyTool,
    Tool,
    Toolset,
    discover_tools_on_obj,
    tool,
    tool_method,
)

if t.TYPE_CHECKING:
    from dreadnode.agent.tools import fs  # noqa: F401

__all__ = [
    "AnyTool",
    "Tool",
    "Toolset",
    "discover_tools_on_obj",
    "planning",
    "reporting",
    "tasking",
    "tool",
    "tool_method",
]

__lazy_submodules__ = ["highlight", "task", "todo", "fs"]


def __getattr__(name: str) -> t.Any:
    if name in __lazy_submodules__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __lazy_submodules__)
