import importlib.metadata as md
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import SUPPORTED, load_manifest_file
from .manifest import ToolManifest


@dataclass(frozen=True)
class Discovered:
    id: str
    manifest: ToolManifest
    source: str
    distribution: str | None = None


class ManifestRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, Discovered] = {}

    # ----- discovery -----
    def _raise_invalid_manifest_type(self, result: Any) -> None:
        raise TypeError("manifest entry point must return dict or path-like")

    def discover_entrypoints(self, group: str = "dreadnode.manifest") -> list[Discovered]:
        found: list[Discovered] = []
        eps = md.entry_points(group=group)
        for ep in eps:
            try:
                obj = ep.load()
                result = obj() if callable(obj) else obj
                data: dict[str, Any]

                if isinstance(result, dict):
                    data = result
                elif isinstance(result, (str, Path)):
                    data = load_manifest_file(Path(result)).model_dump(mode="python")
                else:
                    self._raise_invalid_manifest_type(result)
                manifest = ToolManifest.model_validate(data)
                d = Discovered(
                    id=manifest.id,
                    manifest=manifest,
                    source=f"entrypoint:{ep.dist.name}",
                    distribution=ep.dist.name,
                )
                found.append(d)
            except Exception as e:  # robust discovery; log and continue
                print(f"[registry] skip entrypoint {ep!r}: {e}")
        return found

    def discover_paths(self, paths: Iterable[Path]) -> list[Discovered]:
        found: list[Discovered] = []
        for root in paths:
            for p in root.rglob("*"):
                if (
                    p.is_file()
                    and p.suffix.lower() in SUPPORTED
                    and p.stem in {"tool", "manifest", "tool.manifest"}
                ):
                    try:
                        m = load_manifest_file(p)
                        found.append(Discovered(id=m.id, manifest=m, source=f"path:{p}"))
                    except Exception as e:
                        print(f"[registry] skip {p}: {e}")
        return found

    def load_all(self, *, entrypoints: bool = True, extra_paths: list[Path] | None = None) -> None:
        if entrypoints:
            for d in self.discover_entrypoints():
                self._by_id[d.id] = d
        if extra_paths:
            for d in self.discover_paths(extra_paths):
                self._by_id[d.id] = d

    # ----- access -----
    def ids(self) -> list[str]:
        return sorted(self._by_id.keys())

    def get(self, tool_id: str) -> ToolManifest:
        return self._by_id[tool_id].manifest

    def items(self) -> list[Discovered]:
        return list(self._by_id.values())
