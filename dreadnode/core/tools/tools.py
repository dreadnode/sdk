import asyncio
import contextlib
import functools
import inspect
import json
import re
import typing as t

import typing_extensions as te
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_xml import attr
from rigging.error import Stop, ToolDefinitionError

from dreadnode.core.generators.models import (
    ErrorModel,
    XMLModel,
    make_from_schema,
    make_from_signature,
)
from dreadnode.core.generators.utils import deref_json, shorten_string
from dreadnode.core.meta import Component, Model

if t.TYPE_CHECKING:
    from rigging.message import Message


P = t.ParamSpec("P")
R = t.TypeVar("R")


@contextlib.asynccontextmanager
async def _tool_span(tool_name: str, tool_call: "ToolCall") -> t.AsyncIterator[t.Any]:
    """
    Create an async-aware span for tool execution.

    Uses a basic Span with proper async context handling. The span is
    created and ended in the same async context to avoid context token issues.
    """
    span = None
    try:
        from contextvars import copy_context
        from dreadnode.core.tracing.span import Span, current_run_span, get_default_tracer

        # Capture current context
        ctx = copy_context()
        run = current_run_span.get()
        tracer = get_default_tracer()

        span = Span(
            name=f"tool:{tool_name}",
            tracer=tracer,
            type="tool",
            label=tool_name,
            attributes={
                "dreadnode.run.id": run.run_id if run else "",
                "tool.name": tool_name,
                "tool.call_id": tool_call.id,
                "tool.arguments": tool_call.function.arguments,
            },
        )
        span.__enter__()
    except ImportError:
        pass

    try:
        yield span
    finally:
        if span is not None:
            try:
                span.__exit__(None, None, None)
            except ValueError:
                # Context token issue in async cleanup - span data was already recorded
                pass


TOOL_VARIANTS_ATTR = "_tool_variants"
TOOL_STOP_TAG = "rg-stop"
DEFAULT_CATCH_EXCEPTIONS: set[type[Exception]] = {json.JSONDecodeError, ValidationError}

ToolChoice = str | dict[str, t.Any]
ToolMode = t.Literal["auto", "api", "xml", "json", "json-in-xml", "json-with-tag", "pythonic"]

"""
How tool calls are handled.

- `auto`: The method is chosen based on support (api w/ fallback to json-in-xml).
- `api`: Tool calls are delegated to api-provided function calling.
- `xml`: Tool calls are parsed in a nested XML format which is native to Rigging.
- `json`: Tool calls are parsed as raw name/arg JSON anywhere in assistant message content.
- `json-in-xml`: Tool calls are parsed using JSON for arguments, and XML for everything else.
- `json-with-tag`: Tool calls are parsed as name/arg JSON structures inside an XML tag to identify it.
- `pythonic`: Tool calls are parsed as pythonic function call syntax.
"""


class NamedFunction(BaseModel):
    name: str


class ToolChoiceDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: NamedFunction


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, t.Any] | None = None

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: t.Any) -> t.Any:
        if not isinstance(value, dict):
            return value

        if value.get("type") == "object" and value.get("properties") == {}:
            return None

        return value


class ToolDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(json_schema_extra={"dreadnode.type": "tool_call"})

    id: str
    type: t.Literal["function"] = "function"
    function: FunctionCall

    def __str__(self) -> str:
        arguments = shorten_string(self.function.arguments, max_length=50)
        return f"ToolCall({self.function.name}({arguments}), id='{self.id}')"

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def arguments(self) -> str:
        return self.function.arguments


class ToolResponse(XMLModel):
    id: str = attr(default="")
    result: str


class Tool(BaseModel, t.Generic[P, R]):
    """Base class for representing a tool to a generator."""

    name: str
    """The name of the tool."""
    description: str
    """A description of the tool."""
    parameters_schema: dict[str, t.Any]
    """The JSON schema for the tool's parameters."""
    fn: t.Callable[P, R] = Field(  # type: ignore [assignment]
        default_factory=lambda: lambda *args, **kwargs: None,  # noqa: ARG005
        exclude=True,
    )
    """The function to call."""
    catch: bool | set[type[Exception]] = set(DEFAULT_CATCH_EXCEPTIONS)
    """
    Whether to catch exceptions and return them as messages.

    - `False`: Do not catch exceptions.
    - `True`: Catch all exceptions.
    - `set[type[Exception]]`: Catch only the specified exceptions.

    By default, catches `json.JSONDecodeError` and `ValidationError`.
    """
    truncate: int | None = None
    """If set, the maximum number of characters to truncate any tool output to."""

    _signature: inspect.Signature | None = PrivateAttr(default=None, init=False)
    _type_adapter: TypeAdapter[t.Any] | None = PrivateAttr(default=None, init=False)
    _model: type[XMLModel] | None = PrivateAttr(default=None, init=False)

    @classmethod
    def from_callable(
        cls,
        fn: t.Callable[P, R],
        *,
        name: str | None = None,
        description: str | None = None,
        catch: bool | t.Iterable[type[Exception]] | None = None,
        truncate: int | None = None,
    ) -> te.Self:
        from rigging.prompt import Prompt

        fn_for_signature = fn

        if isinstance(fn, Prompt):
            fn_for_signature = fn.func  # type: ignore [assignment]
            fn = fn.run  # type: ignore [assignment]

        elif hasattr(fn, "__dn_prompt__") and isinstance(fn.__dn_prompt__, Prompt):
            if fn.__name__ in ["run_many", "run_over"]:
                raise ValueError(
                    "Only the singular Prompt.run (Prompt.bind) is supported when using prompt objects inside API tools",
                )
            fn_for_signature = fn.__dn_prompt__.func  # type: ignore [assignment]

        signature = inspect.signature(fn_for_signature, eval_str=True)

        @functools.wraps(fn_for_signature)
        def empty_func(*args, **kwargs):  # type: ignore [no-untyped-def] # noqa: ARG001
            return kwargs

        annotations: dict[str, t.Any] = {}
        for param_name, param in signature.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                annotations[param_name] = param.annotation

        if signature.return_annotation is not inspect.Parameter.empty:
            annotations["return"] = signature.return_annotation

        empty_func.__annotations__ = annotations

        type_adapter: TypeAdapter[t.Any] = TypeAdapter(empty_func)

        schema = type_adapter.json_schema()

        # Maintain the behavior of Annotated[<type>, "<description>"] by walking
        # adjusting the schema manually using signature annotations.

        for prop_name in schema.get("properties", {}):
            if prop_name is None or prop_name not in signature.parameters:
                continue

            param = signature.parameters[prop_name]
            if t.get_origin(param.annotation) != t.Annotated:
                continue

            annotation_args = t.get_args(param.annotation)
            if len(annotation_args) != 2 or not isinstance(annotation_args[1], str):
                continue

            schema["properties"][prop_name]["description"] = annotation_args[1]

        # Deref and flatten the schema for consistency

        schema = deref_json(schema, is_json_schema=True)

        description = inspect.cleandoc(description or fn_for_signature.__doc__ or "")
        description = re.sub(r"(?![\r\n])(\b\s+)", " ", description)

        self = cls(
            name=name or fn_for_signature.__name__,
            description=description,
            parameters_schema=schema,
            fn=fn,
            catch=catch or DEFAULT_CATCH_EXCEPTIONS,
            truncate=truncate,
        )

        self._signature = signature
        self.__signature__ = signature  # type: ignore [misc]
        self.__name__ = self.name  # type: ignore [attr-defined]
        self.__doc__ = self.description

        # For handling API calls, we'll use the type adapter to validate
        # the arguments before calling the function

        self._type_adapter = type_adapter

        return self

    @property
    def definition(self) -> ToolDefinition:
        """
        Returns the tool definition for this tool.
        This is used for API calls and should be used
        to construct the tool call in the generator.
        """
        return ToolDefinition(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters_schema,
            ),
        )

    @property
    def api_definition(self) -> ToolDefinition:
        return self.definition

    @property
    def model(self) -> type[XMLModel]:
        # Usually, we only dynamically construct a model when we are
        # using `xml` tool calls (noted above). We'll do this lazily
        # to avoid overhead and exceptions.

        # We use the signature if we have it as it's more accurate,
        # but fallback to using just the schema if we don't.

        if self._model is None:
            try:
                self._model = (
                    make_from_signature(self._signature, "params")
                    if self._signature
                    else make_from_schema(self.parameters_schema, "params")
                )
            except Exception as e:
                raise ToolDefinitionError(
                    f"Failed to create model for tool '{self.name}'. "
                    "This is likely due to constraints on arguments when the `xml` tool mode is used.",
                ) from e
        return self._model

    async def handle_tool_call(  # noqa: PLR0912
        self,
        tool_call: ToolCall,
    ) -> tuple["Message", bool]:
        """
        Handle an incoming tool call from a generator.

        Args:
            tool_call: The tool call to handle.

        Returns:
            A tuple containing the message to send back to the generator and a
            boolean indicating whether tool calling should stop.
        """
        from dreadnode.core.generators.message import ContentText, ContentTypes, Message

        result: t.Any
        stop = False

        try:
            # Load + Validate args
            kwargs = json.loads(tool_call.function.arguments)
            if self._type_adapter is not None:
                kwargs = self._type_adapter.validate_python(kwargs)
            kwargs = kwargs or {}

            # Call the function
            result = self.fn(**kwargs)  # type: ignore [call-arg]
            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, Stop):
                raise result  # noqa: TRY301
        except Stop as e:
            result = f"<{TOOL_STOP_TAG}>{e.message}</{TOOL_STOP_TAG}>"
            stop = True
        except Exception as e:
            if self.catch is True or (
                not isinstance(self.catch, bool) and isinstance(e, tuple(self.catch))
            ):
                result = ErrorModel.from_exception(e)
            else:
                raise

        message = Message(role="tool", tool_call_id=tool_call.id)

        # If this is being gracefully handled as an ErrorModel,
        # we will construct it explicitly so it can attach
        # metadata about the failure.

        if isinstance(result, ErrorModel):
            message = Message.from_model(
                result,
                role="tool",
                tool_call_id=tool_call.id,
            )

        # If the tool gave us back anything that looks like a message, we'll
        # just pass it along. Otherwise we need to box up the result.

        if isinstance(result, Message):
            message.content_parts = result.content_parts
        elif isinstance(result, ContentTypes):
            message.content_parts = [result]
        elif (
            isinstance(result, list)
            and result
            and all(isinstance(item, ContentTypes) for item in result)
        ):
            message.content_parts = result
        elif isinstance(result, XMLModel):
            message.content_parts = [ContentText(text=result.to_pretty_xml())]
        else:
            with contextlib.suppress(Exception):
                if type(result) not in [str, int, float, bool]:
                    result = TypeAdapter(t.Any).dump_json(result).decode(errors="replace")
            message.content_parts = [ContentText(text=str(result))]

        if self.truncate:
            # Use shorten instead of truncate to try and preserve
            # the most context possible.
            message = message.shorten(self.truncate)

        return message, stop

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    def clone(self) -> "Tool[P, R]":
        """
        Create a clone of this tool with the same parameters.
        Useful for creating tools with the same signature but different names.
        """
        new = Tool[P, R](
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
            fn=self.fn,
            catch=self.catch,
            truncate=self.truncate,
        )

        new._signature = self._signature
        new.__signature__ = self.__signature__  # type: ignore [misc]
        new._type_adapter = self._type_adapter
        new.__name__ = self.name  # type: ignore [attr-defined]
        new.__doc__ = self.description

        return new

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        catch: bool | t.Iterable[type[Exception]] | None = None,
        truncate: int | None = None,
    ) -> "Tool[P, R]":
        """
        Create a new tool with updated parameters.
        Useful for creating tools with the same signature but different names or descriptions.

        Args:
            name: The name of the tool.
            description: The description of the tool.
            catch: Whether to catch exceptions and return them as messages.
                - `False`: Do not catch exceptions.
                - `True`: Catch all exceptions.
                - `list[type[Exception]]`: Catch only the specified exceptions.
                - `None`: By default, catches `json.JSONDecodeError` and `ValidationError
            truncate: If set, the maximum number of characters to truncate any tool output to.

        Returns:
            A new tool with the updated parameters.
        """
        new = self.clone()
        new.name = name or self.name
        new.description = description or self.description
        new.catch = (
            catch if isinstance(catch, bool) else self.catch if catch is None else set(catch)
        )
        new.truncate = truncate if truncate is not None else self.truncate
        return new


class ToolMethod(property, t.Generic[P, R]):
    """
    A descriptor that acts as a factory for creating bound Tool instances.

    It inherits from `property` to be ignored by pydantic's `ModelMetaclass`
    during field inspection. This prevents validation errors which would
    otherwise treat the descriptor as a field and stop tool_method decorators
    from being applied in BaseModel classes.
    """

    def __init__(
        self,
        fget: t.Callable[..., t.Any],
        name: str,
        description: str,
        *,
        catch: bool | t.Iterable[type[Exception]] | None,
        parameters_schema: dict[str, t.Any],
        truncate: int | None,
        signature: inspect.Signature,
        type_adapter: TypeAdapter[t.Any],
    ):
        super().__init__(fget)
        self.tool_name = name
        self.tool_description = description
        self.tool_parameters_schema = parameters_schema
        self.tool_catch = catch
        self.tool_truncate = truncate
        self._tool_signature = signature
        self._tool_type_adapter = type_adapter

    @te.overload  # type: ignore [override, unused-ignore]
    def __get__(self, instance: None, owner: type[object] | None = None) -> Tool[P, R]: ...

    @te.overload
    def __get__(self, instance: object, owner: type[object] | None = None) -> Tool[P, R]: ...

    @te.override
    def __get__(self, instance: object | None, owner: type[object] | None = None) -> Tool[P, R]:
        if self.fget is None:
            raise AttributeError(
                f"Tool '{self.tool_name}' is not defined on instance of {owner.__name__ if owner else 'unknown'}.",
            )

        # Class access: return an unbound Tool for inspection.
        if instance is None:
            tool = Tool[P, R](
                fn=self.fget,
                name=self.tool_name,
                description=self.tool_description,
                parameters_schema=self.tool_parameters_schema,
                catch=self.tool_catch or DEFAULT_CATCH_EXCEPTIONS,
                truncate=self.tool_truncate,
            )
            tool._signature = self._tool_signature  # noqa: SLF001
            tool._type_adapter = self._tool_type_adapter  # noqa: SLF001
            return tool

        # Instance access: return a new Tool bound to the instance.
        bound_method = self.fget.__get__(instance, owner)
        bound_tool = Tool[P, R](
            fn=bound_method,
            name=self.tool_name,
            description=self.tool_description,
            parameters_schema=self.tool_parameters_schema,
            catch=self.tool_catch or DEFAULT_CATCH_EXCEPTIONS,
            truncate=self.tool_truncate,
        )
        bound_tool._signature = self._tool_signature  # noqa: SLF001
        bound_tool._type_adapter = self._tool_type_adapter  # noqa: SLF001
        return bound_tool


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

    def get_tools(self, *, variant: str | None = None) -> list[Tool]:
        variant = variant or self.variant

        tools: list[Tool] = []
        seen_names: set[str] = set()

        for cls in self.__class__.__mro__:
            for name, class_member in cls.__dict__.items():
                if name in seen_names or not isinstance(class_member, ToolMethod):
                    continue

                variants = getattr(class_member, TOOL_VARIANTS_ATTR, [])
                if not variant or not variants or variant in variants:
                    bound_tool = t.cast("Tool", getattr(self, name))
                    tools.append(bound_tool)
                    seen_names.add(name)

        return tools


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
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], ToolMethod[P, R]]: ...


@t.overload
def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R],
    /,
) -> ToolMethod[P, R]: ...


def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    /,
    *,
    variants: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], ToolMethod[P, R]] | ToolMethod[P, R]:
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
    ) -> ToolMethod[P, R]:
        tool_method_descriptor: ToolMethod[P, R] = tool_method(
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )(func)

        setattr(tool_method_descriptor, TOOL_VARIANTS_ATTR, variants or ["all"])

        return tool_method_descriptor

    return make_tool_method(func) if func is not None else make_tool_method


def discover_tools_on_obj(obj: t.Any) -> list[Tool]:
    tools: list[Tool] = []

    if not hasattr(obj, "__class__"):
        return tools

    if isinstance(obj, Toolset):
        return obj.get_tools()

    seen_names: set[str] = set()

    for cls in getattr(obj.__class__, "__mro__", []):
        for name, class_member in getattr(cls, "__dict__", {}).items():
            if name in seen_names or not isinstance(class_member, ToolMethod):
                continue

            bound_tool = t.cast("Tool", getattr(obj, name))
            tools.append(bound_tool)
            seen_names.add(name)

    return tools
