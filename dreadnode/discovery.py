import importlib
import inspect
import sys
import typing as t
from dataclasses import dataclass
from pathlib import Path

from dreadnode.util import warn_at_user_stacklevel

T = t.TypeVar("T")


DEFAULT_SEARCH_PATHS = ("main.py", "agent.py", "app.py", "eval.py", "attack.py", "study.py")


class DiscoveryWarning(UserWarning):
    """Warning related to object discovery."""


@dataclass
class ModuleData:
    module_import_str: str
    extra_sys_path: Path


@dataclass
class Discovered(t.Generic[T]):
    name: str
    path: Path
    obj: T


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


def _discover_in_module(
    module_data: ModuleData,
    discovery_type: type[T],
    *,
    exclude_types: set[type] | None = None,
    ignore_errors: bool = True,
) -> dict[str, T]:
    """
    Imports a module and finds all instances of the specified discoverable type.
    """
    objects: dict[str, T] = {}
    try:
        sys.path.insert(0, str(module_data.extra_sys_path))
        mod = importlib.import_module(module_data.module_import_str)
    except Exception as e:
        error_str = f"Failed to import '{module_data.module_import_str}.py': {e}"
        if not ignore_errors:
            raise ModuleNotFoundError(error_str) from e
        warn_at_user_stacklevel(error_str, DiscoveryWarning)
        return objects
    finally:
        sys.path.pop(0)

    exclude_types = exclude_types or set()
    for obj_name in dir(mod):
        obj = getattr(mod, obj_name)
        if isinstance(obj, discovery_type) and not any(isinstance(obj, et) for et in exclude_types):
            objects[obj_name] = obj

    return objects


def _discover_from_path(
    discovery_type: type[T], path: Path | None, *, exclude_types: set[type] | None = None
) -> list[Discovered[T]]:
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"Path does not exist or is not a file: {path}")

    objects: list[Discovered[T]] = []

    if path is not None:
        module_data = _get_module_data_from_path(path)
        for name, obj in _discover_in_module(
            module_data, discovery_type, exclude_types=exclude_types, ignore_errors=False
        ).items():
            objects.append(Discovered(name=name, path=path, obj=obj))
        return objects

    for default_name in DEFAULT_SEARCH_PATHS:
        path = Path(default_name)
        if not path.is_file():
            continue

        module_data = _get_module_data_from_path(path)
        for name, obj in _discover_in_module(
            module_data, discovery_type, exclude_types=exclude_types
        ).items():
            objects.append(Discovered(name=name, path=path, obj=obj))

    return objects


def _discover_from_qualified_name(discovery_type: type[T], qualified_name: str) -> Discovered[T]:
    module_path, obj_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, obj_name)

    if not isinstance(obj, discovery_type):
        raise TypeError(
            f"Object at '{qualified_name}' is not of the expected type '{discovery_type.__name__}'."
        )

    file_path = Path(inspect.getfile(module))
    return Discovered(name=obj_name, path=file_path, obj=obj)


def discover(
    discovery_type: type[T],
    identifier: str | Path | None = None,
    *,
    exclude_types: set[type] | None = None,
) -> list[Discovered[T]]:
    """
    Discovers all objects of a specific type from a file path or FQDN.

    - If identifier is None, searches default paths.
    - If identifier looks like a path, searches that file.
    - If identifier looks like a qualified name, imports and returns that object.

    Returns a flat list of all discovered objects.
    """

    is_path_like = (
        isinstance(identifier, Path)
        or ".py" in str(identifier)
        or "/" in str(identifier)
        or "\\" in str(identifier)
    )

    if identifier is None or is_path_like:
        path = Path(identifier) if identifier is not None else None
        return _discover_from_path(discovery_type, path, exclude_types=exclude_types)

    try:
        return [_discover_from_qualified_name(discovery_type, str(identifier))]
    except (ImportError, AttributeError, TypeError):
        return []


def find(
    discovery_type: type[T],
    identifier: str | Path,
    name: str | None = None,
) -> T:
    """
    Finds a single, specific object by its identifier and optional name.

    - If `identifier` is 'my_evals.py' and `name` is 'accuracy_test', it finds that specific eval.
    - If `identifier` is 'my_evals.py:accuracy_test', it parses and finds that specific eval.
    - If `identifier` is 'my_package.evals.accuracy_test', it imports it.

    Raises a ValueError if no object or multiple objects are found.
    """
    # Handle the 'path:name' format
    if isinstance(identifier, str) and ":" in identifier:
        identifier, name = identifier.rsplit(":", 1)

    # Get all the candidates
    discovered_items = discover(discovery_type, identifier)
    if not discovered_items:
        raise ValueError(
            f"No objects of type '{discovery_type.__name__}' found for identifier: {identifier}"
        )

    # Filter by name if provided
    if name:
        candidates = [d for d in discovered_items if d.name == name]
        if not candidates:
            available = ", ".join(d.name for d in discovered_items)
            raise ValueError(
                f"Object '{name}' not found for identifier '{identifier}'. Available: [{available}]"
            )
        discovered_items = candidates

    if len(discovered_items) > 1:
        raise ValueError(
            f"Multiple objects found for identifier '{identifier}'. Please specify a name. "
            f"Found: {[d.name for d in discovered_items]}"
        )

    return discovered_items[0].obj
