import asyncio
import functools
import typing as t

from pydantic import ConfigDict, PrivateAttr
from rigging import tools
from rigging.tools.base import ToolMethod as RiggingToolMethod

from dreadnode.meta import Component, Model

Tool = tools.Tool
ToolMode = tools.ToolMode
ToolCall = tools.ToolCall
FunctionCall = tools.FunctionCall
ToolDefinition = tools.ToolDefinition
FunctionDefinition = tools.FunctionDefinition

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
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], RiggingToolMethod[P, R]]: ...


@t.overload
def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R],
    /,
) -> RiggingToolMethod[P, R]: ...


def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    /,
    *,
    variants: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> (
    t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], RiggingToolMethod[P, R]]
    | RiggingToolMethod[P, R]
):
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

    def make_tool_method(
        func: t.Callable[t.Concatenate[t.Any, P], R],
    ) -> RiggingToolMethod[P, R]:
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
    - Support for async context management, with automatic re-entrancy handling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    variant: str | None = None
    """The variant for filtering tools available in this toolset."""

    # Context manager magic
    _entry_ref_count: int = PrivateAttr(default=0)
    _context_handle: object = PrivateAttr(default=None)
    _entry_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def name(self) -> str:
        """The name of the toolset, derived from the class name."""
        return self.__class__.__name__

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        # This essentially ensures that if the Toolset is any kind of context manager,
        # it will be re-entrant, and only actually enter/exit once. This means we can
        # safely build auto-entry/exit logic into our Agent class without worrying about
        # breaking the code if the user happens to enter a toolset manually before using
        # it in an agent.

        original_aenter = cls.__dict__.get("__aenter__")
        original_enter = cls.__dict__.get("__enter__")
        original_aexit = cls.__dict__.get("__aexit__")
        original_exit = cls.__dict__.get("__exit__")

        has_enter = callable(original_aenter) or callable(original_enter)
        has_exit = callable(original_aexit) or callable(original_exit)

        if has_enter and not has_exit:
            raise TypeError(
                f"{cls.__name__} defining __aenter__ or __enter__ must also define __aexit__ or __exit__"
            )
        if has_exit and not has_enter:
            raise TypeError(
                f"{cls.__name__} defining __aexit__ or __exit__ must also define __aenter__ or __enter__"
            )
        if original_aenter and original_enter:
            raise TypeError(f"{cls.__name__} cannot define both __aenter__ and __enter__")
        if original_aexit and original_exit:
            raise TypeError(f"{cls.__name__} cannot define both __aexit__ and __exit__")

        @functools.wraps(original_aenter or original_enter)  # type: ignore[arg-type]
        async def aenter_wrapper(self: "Toolset", *args: t.Any, **kwargs: t.Any) -> t.Any:
            async with self._entry_lock:
                if self._entry_ref_count == 0:
                    handle = None
                    if original_aenter:
                        handle = await original_aenter(self, *args, **kwargs)
                    elif original_enter:
                        handle = original_enter(self, *args, **kwargs)
                    self._context_handle = handle if handle is not None else self
                self._entry_ref_count += 1
                return self._context_handle

        cls.__aenter__ = aenter_wrapper  # type: ignore[attr-defined]

        @functools.wraps(original_aexit or original_exit)  # type: ignore[arg-type]
        async def aexit_wrapper(self: "Toolset", *args: t.Any, **kwargs: t.Any) -> t.Any:
            async with self._entry_lock:
                self._entry_ref_count -= 1
                if self._entry_ref_count == 0:
                    if original_aexit:
                        await original_aexit(self, *args, **kwargs)
                    elif original_exit:
                        original_exit(self, *args, **kwargs)
                    self._context_handle = None

        cls.__aexit__ = aexit_wrapper  # type: ignore[attr-defined]

    def get_tools(self, *, variant: str | None = None) -> list[AnyTool]:
        variant = variant or self.variant

        tools: list[AnyTool] = []
        seen_names: set[str] = set()

        for cls in self.__class__.__mro__:
            for name, class_member in cls.__dict__.items():
                if name in seen_names or not isinstance(class_member, RiggingToolMethod):
                    continue

                variants = getattr(class_member, TOOL_VARIANTS_ATTR, [])
                if not variant or not variants or variant in variants:
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
