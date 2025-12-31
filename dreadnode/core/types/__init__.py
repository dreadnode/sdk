import importlib
import typing as t

from dreadnode.core.types.base import DataType, WithMeta
from dreadnode.core.types.object_3d import Object3D
from dreadnode.core.types.text import Code, Markdown, Text

if t.TYPE_CHECKING:
    from dreadnode.core.types.audio import Audio
    from dreadnode.core.types.image import Image
    from dreadnode.core.types.table import Table
    from dreadnode.core.types.video import Video

__all__ = [
    "Audio",
    "Code",
    "DataType",
    "Image",
    "Markdown",
    "Object3D",
    "Table",
    "Text",
    "Video",
    "WithMeta",
]

__lazy_submodules__: list[str] = []
__lazy_components__: dict[str, str] = {
    "Audio": "dreadnode.core.types.audio",
    "Image": "dreadnode.core.types.image",
    "Table": "dreadnode.core.types.table",
    "Video": "dreadnode.core.types.video",
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
