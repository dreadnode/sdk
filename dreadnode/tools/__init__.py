import importlib
import typing as t

from dreadnode.tools import execute, fs, memory, planning, reporting, tasking

__all__ = [
    "execute",
    "fs",
    "memory",
    "planning",
    "reporting",
    "tasking",
    "tool",
    "tool_method",
]

__lazy_submodules__: list[str] = ["fs", "planning", "reporting", "tasking", "execute", "memory"]
__lazy_components__: dict[str, str] = {}


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
