"""Unified resource loading via URI schemes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from dreadnode.core.agents import Agent
    from dreadnode.agents import AgentPackage, LocalAgent
    from dreadnode.datasets import Dataset, LocalDataset
    from dreadnode.models import Model, LocalModel
    from dreadnode.core.storage.storage import Storage


# URI pattern: scheme://path or just path (scheme can include hyphens)
URI_PATTERN = re.compile(r"^(?P<scheme>[a-z][a-z0-9-]*)://(?P<path>.+)$")


@overload
def load(
    uri: str,
    *,
    type: Literal["dataset"],
    storage: Storage | None = None,
    **kwargs: Any,
) -> Dataset | LocalDataset: ...


@overload
def load(
    uri: str,
    *,
    type: Literal["agent"],
    **kwargs: Any,
) -> Agent: ...


@overload
def load(
    uri: str,
    *,
    type: Literal["model"],
    storage: Storage | None = None,
    **kwargs: Any,
) -> Model | LocalModel: ...


@overload
def load(
    uri: str,
    *,
    type: None = None,
    storage: Storage | None = None,
    **kwargs: Any,
) -> Dataset | LocalDataset | Agent | Model | LocalModel | Any: ...


def load(
    uri: str,
    *,
    type: Literal["dataset", "agent", "model"] | None = None,
    storage: Storage | None = None,
    **kwargs: Any,
) -> Any:
    """Load a resource by URI.

    Supports multiple URI schemes for loading different resource types:

    - `dataset://name` - Load a published dataset package
    - `dataset://name@version` - Load specific version
    - `model://name` - Load a published model package
    - `hf://path` - Load dataset from HuggingFace Hub
    - `hf-model://path` - Load model from HuggingFace Hub
    - `agent://name` - Load a published agent package
    - Plain path without scheme - Treated as HuggingFace dataset path

    Args:
        uri: Resource URI (e.g., "dataset://my-data", "hf://squad").
        type: Explicit type hint (optional, inferred from scheme).
        storage: Storage instance for local resources.
        **kwargs: Additional arguments passed to the loader.

    Returns:
        The loaded resource (Dataset, LocalDataset, Model, LocalModel, Agent, etc.)

    Example:
        >>> import dreadnode as dn
        >>>
        >>> # Load a published dataset package
        >>> ds = dn.load("dataset://my-org/sentiment-data")
        >>>
        >>> # Load dataset from HuggingFace Hub
        >>> ds = dn.load("hf://squad", split="train[:100]")
        >>>
        >>> # Load model from HuggingFace Hub
        >>> model = dn.load("hf-model://bert-base-uncased")
        >>>
        >>> # Load a published model package
        >>> model = dn.load("model://my-org/classifier")
        >>>
        >>> # Load an agent package
        >>> agent = dn.load("agent://my-org/agent")
        >>>
        >>> # Plain path defaults to HuggingFace dataset
        >>> ds = dn.load("imdb", split="train")
    """
    match = URI_PATTERN.match(uri)

    if match:
        scheme = match.group("scheme")
        path = match.group("path")
    else:
        # No scheme - default based on type hint or assume HuggingFace
        scheme = type or "hf"
        path = uri

    # Parse version from path (e.g., "name@1.0.0")
    version = None
    if "@" in path:
        path, version = path.rsplit("@", 1)

    # Dispatch to appropriate loader
    if scheme == "dataset":
        return _load_dataset_package(path, version=version, **kwargs)
    elif scheme == "hf":
        return _load_hf_dataset(path, storage=storage, **kwargs)
    elif scheme == "model":
        return _load_model_package(path, version=version, **kwargs)
    elif scheme in ("hf-model", "huggingface-model"):
        return _load_hf_model(path, storage=storage, **kwargs)
    elif scheme == "agent":
        return _load_agent_package(path, version=version, **kwargs)
    else:
        raise ValueError(f"Unknown URI scheme: {scheme}")


def _load_dataset_package(
    name: str,
    version: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """Load a published dataset package."""
    from dreadnode.datasets.dataset import Dataset

    # The Dataset class loads by entry point name
    # Version handling would need to be added to the loader
    ds = Dataset(name)
    return ds


def _load_hf_dataset(
    path: str,
    storage: Storage | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> LocalDataset:
    """Load a dataset from HuggingFace Hub."""
    from dreadnode.datasets.local import load_dataset

    return load_dataset(path, storage=storage, name=name, **kwargs)


def _load_agent_package(
    name: str,
    version: str | None = None,
    **kwargs: Any,
) -> Agent:
    """Load a published agent package."""
    from dreadnode.agents.loader import AgentPackage

    pkg = AgentPackage(name)
    return pkg.load(**kwargs)


def _load_model_package(
    name: str,
    version: str | None = None,
    **kwargs: Any,
) -> Model:
    """Load a published model package."""
    from dreadnode.models.model import Model

    return Model(name)


def _load_hf_model(
    path: str,
    storage: Storage | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> LocalModel:
    """Load a model from HuggingFace Hub."""
    from dreadnode.models.local import load_model

    return load_model(path, storage=storage, name=name, **kwargs)
