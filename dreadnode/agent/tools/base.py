import inspect
import typing as t

from pydantic import BaseModel, ConfigDict
from rigging import tools
from rigging.tools.base import Tool, ToolMethod

from dreadnode.agent.configurable import CONFIGURABLE_ATTR, configurable

Tool = tools.Tool
tool = tools.tool

AnyTool = Tool[t.Any, t.Any]

P = t.ParamSpec("P")
R = t.TypeVar("R")

TOOL_VARIANTS_ATTR = "_tool_variants"


def tool_method(
    *,
    variants: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], ToolMethod[P, R]]:
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
        catch: A set of exceptions to catch and return as an error message.
        truncate: The maximum number of characters for the tool's output.
    """

    def decorator(func: t.Callable[P, R]) -> ToolMethod[P, R]:
        tool_method_descriptor: ToolMethod[P, R] = tools.tool_method(
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )(func)

        setattr(tool_method_descriptor, TOOL_VARIANTS_ATTR, variants or ["all"])

        return tool_method_descriptor

    return decorator


class Toolset(BaseModel):
    """
    A Pydantic-based class for creating a collection of related, stateful tools.

    Inheriting from this class provides:
    - Pydantic's declarative syntax for defining state (fields).
    - Automatic application of the `@configurable` decorator.
    - A `get_tools` method for discovering methods decorated with `@dreadnode.tool_method`.
    """

    variant: str = "all"
    """The variant for filtering tools available in this toolset."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """
        This method automatically applies the @configurable decorator to any
        subclass of Toolset, making them configurable by default.
        """
        super().__init_subclass__(**kwargs)

        # Apply the @configurable decorator if it hasn't been yet
        if not getattr(cls, CONFIGURABLE_ATTR, False):
            configurable(cls)

    @property
    def name(self) -> str:
        """The name of the toolset, derived from the class name."""
        return self.__class__.__name__

    def get_tools(self, *, variant: str | None = None) -> list[AnyTool]:
        variant = variant or self.variant
        tools: list[AnyTool] = []

        # The loop
        for name, raw in inspect.getmembers_static(self.__class__):
            if isinstance(raw, ToolMethod):
                variants = getattr(raw, TOOL_VARIANTS_ATTR, [])
                if variant in variants:
                    bound_tool = t.cast("AnyTool", getattr(self, name))
                    tools.append(bound_tool)
        return tools
