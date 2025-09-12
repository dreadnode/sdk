import typing as t

from pydantic import ConfigDict
from rigging import tools
from rigging.tools.base import ToolMethod as RiggingToolMethod

from dreadnode.meta import Component, Config, Model

Tool = tools.Tool
ToolMode = tools.ToolMode

AnyTool = Tool[t.Any, t.Any]

P = t.ParamSpec("P")
R = t.TypeVar("R")

TOOL_VARIANTS_ATTR = "_tool_variants"


@t.overload
def tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], Tool[P, R]]: ...


@t.overload
def tool(
    func: t.Callable[P, R],
    /,
) -> Tool[P, R]: ...


def tool(
    func: t.Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], Tool[P, R]] | Tool[P, R]:
    """
    Decorator for creating a Tool, useful for overriding a name or description.

    Note:
        If the func contains Config or Context arguments, they will not be exposed
        as part of the tool schema, and you ensure they have default values or
        are correctly passed values.

    Args:
        func: The function to wrap.
        name: The name of the tool.
        description: The description of the tool.
        catch: Whether to catch exceptions and return them as messages.
            - `False`: Do not catch exceptions.
            - `True`: Catch all exceptions.
            - `list[type[Exception]]`: Catch only the specified exceptions.
            - `None`: By default, catches `json.JSONDecodeError` and `ValidationError`.
        truncate: If set, the maximum number of characters to truncate any tool output to.

    Returns:
        The decorated Tool object.

    Example:
        ```
        @tool(name="add_numbers", description="This is my tool")
        def add(x: int, y: int) -> int:
            return x + y
        ```
    """

    def make_tool(func: t.Callable[P, R]) -> Tool[P, R]:
        # This is purely here to inject component logic into a tool
        component = func if isinstance(func, Component) else Component(func)
        return Tool[P, R].from_callable(
            component,
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )

    return make_tool(func) if func is not None else make_tool


@t.overload
def tool_method(
    func: None = None,
    /,
    *,
    variants: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], RiggingToolMethod[P, R]]: ...


@t.overload
def tool_method(
    func: t.Callable[P, R],
    /,
) -> RiggingToolMethod[P, R]: ...


def tool_method(
    func: t.Callable[P, R] | None = None,
    /,
    *,
    variants: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], RiggingToolMethod[P, R]] | RiggingToolMethod[P, R]:
    """
    Marks a method on a Toolset as a tool, adding it to specified variants.

    This is a transparent, signature-preserving wrapper around `rigging.tool_method`.
    Use this for any method inside a class that inherits from `dreadnode.Toolset`
    to ensure it's discoverable.

    Args:
        variants: A list of variants this tool should be a part of.
                  If None, it's added to a "all" variant.
        name: Override the tool's name. Defaults to the function name.
        description: Override the tool's description. Defaults to the docstring.
        catch: Whether to catch exceptions and return them as messages.
            - `False`: Do not catch exceptions.
            - `True`: Catch all exceptions.
            - `list[type[Exception]]`: Catch only the specified exceptions.
            - `None`: By default, catches `json.JSONDecodeError` and `ValidationError`.
        truncate: The maximum number of characters for the tool's output.
    """

    def make_tool_method(func: t.Callable[P, R]) -> RiggingToolMethod[P, R]:
        tool_method_descriptor: RiggingToolMethod[P, R] = tools.tool_method(
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )(func)

        setattr(tool_method_descriptor, TOOL_VARIANTS_ATTR, variants or ["all"])

        return tool_method_descriptor

    return make_tool_method(func) if func is not None else make_tool_method


class Toolset(Model):
    """
    A Pydantic-based class for creating a collection of related, stateful tools.

    Inheriting from this class provides:
    - Pydantic's declarative syntax for defining state (fields).
    - Automatic application of the `@configurable` decorator.
    - A `get_tools` method for discovering methods decorated with `@dreadnode.tool_method`.
    """

    variant: str = Config("all")
    """The variant for filtering tools available in this toolset."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    @property
    def name(self) -> str:
        """The name of the toolset, derived from the class name."""
        return self.__class__.__name__

    def get_tools(self, *, variant: str | None = None) -> list[AnyTool]:
        variant = variant or self.variant

        tools: list[AnyTool] = []
        seen_names: set[str] = set()

        for cls in self.__class__.__mro__:
            for name, class_member in cls.__dict__.items():
                if name in seen_names or not isinstance(class_member, RiggingToolMethod):
                    continue

                variants = getattr(class_member, TOOL_VARIANTS_ATTR, [])
                if variant in variants:
                    bound_tool = t.cast("AnyTool", getattr(self, name))
                    tools.append(bound_tool)
                    seen_names.add(name)

        return tools


def discover_tools_on_obj(obj: t.Any) -> list[AnyTool]:
    tools: list[AnyTool] = []

    if not hasattr(obj, "__class__"):
        return tools

    if isinstance(obj, Toolset):
        return obj.get_tools()

    seen_names: set[str] = set()

    for cls in obj.__class__.get("__mro__", []):
        for name, class_member in cls.get("__dict__", {}).items():
            if name in seen_names or not isinstance(class_member, RiggingToolMethod):
                continue

            bound_tool = t.cast("AnyTool", getattr(obj, name))
            tools.append(bound_tool)
            seen_names.add(name)

    return tools
