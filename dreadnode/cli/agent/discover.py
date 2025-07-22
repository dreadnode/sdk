import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rigging.agent import Agent  # Our Agent class

# (We can copy get_default_path and get_module_data_from_path almost verbatim)


def get_default_path() -> Path:
    potential_paths = ("main.py", "agent.py", "app.py")
    for full_path in potential_paths:
        path = Path(full_path)
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find a default file (e.g., main.py, agent.py).")


@dataclass
class ModuleData:
    module_import_str: str
    extra_sys_path: Path


def get_module_data_from_path(path: Path) -> ModuleData:
    use_path = path.resolve()
    module_path = use_path
    if use_path.is_file() and use_path.stem == "__init__":
        module_path = use_path.parent
    module_paths = [module_path]
    extra_sys_path = module_path.parent
    for parent in module_path.parents:
        init_path = parent / "__init__.py"
        if init_path.is_file():
            module_paths.insert(0, parent)
            extra_sys_path = parent.parent
        else:
            break

    module_str = ".".join(p.stem for p in module_paths)
    return ModuleData(
        module_import_str=module_str,
        extra_sys_path=extra_sys_path.resolve(),
        module_paths=module_paths,
    )


# This is our key new function, replacing `get_app_name`
def discover_agents_in_module(module_data: ModuleData) -> dict[str, Agent]:
    """
    Imports a module and finds all instances of rigging.agent.Agent.
    Returns a dictionary mapping their configured names to the agent instances.
    """
    agents: dict[str, Agent] = {}
    try:
        # Temporarily add the module's parent to the path to ensure correct imports
        sys.path.insert(0, str(module_data.extra_sys_path))
        mod = importlib.import_module(module_data.module_import_str)
    finally:
        sys.path.pop(0)

    for obj_name in dir(mod):
        obj = getattr(mod, obj_name)
        if isinstance(obj, Agent):
            # The agent's `name` field is the key for our registry
            agents[obj.name] = obj
    return agents


@dataclass
class DiscoveryResult:
    """Holds all discovered agents from a given path."""

    agents: dict[str, Agent] = field(default_factory=dict)
    module_data: ModuleData | None = None

    @property
    def is_empty(self) -> bool:
        return not self.agents

    @property
    def default_agent(self) -> Agent | None:
        """Returns the first agent discovered, as a default."""
        if self.is_empty:
            return None
        return next(iter(self.agents.values()))


def discover_agents(path: Path | None = None) -> DiscoveryResult:
    """The main discovery entrypoint."""
    if not path:
        path = get_default_path()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    module_data = get_module_data_from_path(path)
    agents = discover_agents_in_module(module_data)

    return DiscoveryResult(agents=agents, module_data=module_data)
