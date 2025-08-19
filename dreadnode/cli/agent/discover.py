import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dreadnode.agent.agent import Agent


@dataclass
class ModuleData:
    module_import_str: str
    extra_sys_path: Path


def _get_module_data_from_path(path: Path) -> ModuleData:
    """
    Calculates the python import string and the necessary sys.path entry
    to import a module from a given file path. Handles packages correctly.
    """
    use_path = path.resolve()

    # Start walking up from the file's directory to find the package root
    current = use_path.parent
    while current != current.parent and (current / "__init__.py").exists():
        current = current.parent

    # The path to add to sys.path is the parent of the package root
    extra_sys_path = current

    # The import string is the relative path from the package root
    relative_path = use_path.with_suffix("").relative_to(extra_sys_path)
    module_import_str = ".".join(relative_path.parts)

    return ModuleData(
        module_import_str=module_import_str,
        extra_sys_path=extra_sys_path,
    )


def _discover_agents_in_module(module_data: ModuleData) -> dict[str, Agent]:
    """
    Imports a module and finds all instances of dreadnode.agent.agent.Agent.
    """
    agents: dict[str, Agent] = {}
    try:
        # Temporarily add the module's parent to the path to ensure correct imports
        sys.path.insert(0, str(module_data.extra_sys_path))
        mod = importlib.import_module(module_data.module_import_str)
    finally:
        # Always clean up the path
        sys.path.pop(0)

    for obj_name in dir(mod):
        obj = getattr(mod, obj_name)
        if isinstance(obj, Agent):
            agents[obj.name] = obj
    return agents


@dataclass
class Discovered:
    filepath: Path
    agents: dict[str, Agent] = field(default_factory=dict)
    module_data: ModuleData | None = None


def discover_agents(file_path: Path | None = None) -> Discovered:
    """
    The main discovery entrypoint. Finds and loads agents from a file.
    If no file is provided, it searches for default filenames.
    """
    if file_path is None:
        for default_name in ("main.py", "agent.py", "app.py"):
            path = Path(default_name)
            if path.is_file():
                file_path = path
                break
        else:
            raise FileNotFoundError(
                "Could not find a default file (main.py, agent.py, app.py). Please specify a file."
            )

    if not file_path.is_file():
        raise FileNotFoundError(f"Path does not exist or is not a file: {file_path}")

    module_data = _get_module_data_from_path(file_path)
    agents = _discover_agents_in_module(module_data)

    return Discovered(filepath=file_path, agents=agents, module_data=module_data)
