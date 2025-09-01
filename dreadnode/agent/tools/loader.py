from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement

from .manifest import ToolManifest


def _merge_defaults(
    schema_defaults: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    cfg = dict(schema_defaults)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


@dataclass
class LoadedTool:
    manifest: ToolManifest
    instance: Any  # Tool/Toolset object
    config: dict[str, Any]


def load_tool(
    manifest: ToolManifest, *, core_version: str, config_overrides: dict[str, Any] | None = None
) -> LoadedTool:
    # 1) compatibility
    if not manifest.compatibility.satisfied(core_version=core_version):
        raise RuntimeError(f"Tool {manifest.id} incompatible with core {core_version}")

    # 2) requirements (Python) — report missing; do not auto-install.
    missing: list[str] = []
    for spec in manifest.requirements.python.packages:
        try:
            Requirement(spec)  # validates syntax only
        except Exception as e:
            raise RuntimeError(f"Invalid requirement '{spec}': {e}")
        # Optional: attempt import heuristic by package name (best-effort)
    if missing:
        raise RuntimeError(f"Missing packages: {missing}")

    # 3) import entrypoint
    ep = manifest.entrypoint
    mod = import_module(ep.module)
    obj = None
    if ep.factory:
        obj = getattr(mod, ep.factory)
        instance = obj(**(ep.kwargs or {}))
    else:
        target = getattr(mod, ep.qualname) if ep.qualname else mod
        # If class -> instantiate; if function -> use as-is; if module -> look for "create".
        if isinstance(target, type) or callable(target):
            instance = target(**(ep.kwargs or {}))
        else:
            factory = getattr(target, "create", None)
            if not callable(factory):
                raise RuntimeError(
                    "Entrypoint must be class/function or expose a callable 'create'"
                )
            instance = factory(**(ep.kwargs or {}))

    # 4) config merge
    cfg = _merge_defaults(manifest.config_schema.defaults(), config_overrides)

    # Optional: you can inject cfg into your Tool/Toolset constructor or setter here
    if hasattr(instance, "configure") and callable(instance.configure):
        instance.configure(cfg)

    return LoadedTool(manifest=manifest, instance=instance, config=cfg)


SUPPORTED = {".yaml", ".yml"}


def load_manifest_file(path: Path) -> ToolManifest:
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported manifest extension: {ext}")
    data: dict[str, Any]
    if ext in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
