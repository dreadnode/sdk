"""Model loading and storage."""

from dreadnode.models.model import Model
from dreadnode.models.model import load_model as load_package
from dreadnode.models.local import LocalModel
from dreadnode.models.local import load_model

__all__ = ["Model", "LocalModel", "load_model", "load_package"]
