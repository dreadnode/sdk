import importlib
import typing as t

from dreadnode.data_types.base import WithMeta
from dreadnode.data_types.object_3d import Object3D
from dreadnode.data_types.text import Code, Markdown, Text

if t.TYPE_CHECKING:
    from dreadnode.data_types.audio import Audio
    from dreadnode.data_types.image import Image
    from dreadnode.data_types.table import Table
    from dreadnode.data_types.video import Video

__all__ = ["Audio", "Code", "Image", "Markdown", "Object3D", "Table", "Text", "Video", "WithMeta"]

__lazy_submodules__: list[str] = []
__lazy_components__: dict[str, str] = {
    "Audio": "dreadnode.data_types.audio",
    "Image": "dreadnode.data_types.image",
    "Table": "dreadnode.data_types.table",
    "Video": "dreadnode.data_types.video",
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
