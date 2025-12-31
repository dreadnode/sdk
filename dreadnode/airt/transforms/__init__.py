import importlib
import typing as t

if t.TYPE_CHECKING:
    from dreadnode.airt.transforms import (
        cipher,
        encoding,
        image,
        perturbation,
        refine,
        stylistic,
        substitution,
        swap,
        text,
    )

__all__ = [
    "cipher",
    "encoding",
    "image",
    "perturbation",
    "refine",
    "stylistic",
    "substitution",
    "swap",
    "text",
]

__lazy_submodules__: list[str] = [
    "cipher",
    "encoding",
    "image",
    "perturbation",
    "stylistic",
    "substitution",
    "refine",
    "swap",
    "text",
]
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
