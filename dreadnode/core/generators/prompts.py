"""
Simple prompt decorator for structured LLM interactions.

This module provides a decorator to convert functions with XMLModel
input/output types into callable prompts that use generators.
"""

from __future__ import annotations

import functools
import inspect
import typing as t

from dreadnode.core.generators.models import XMLModel

if t.TYPE_CHECKING:
    from dreadnode.core.generators.generator import Generator

P = t.ParamSpec("P")
R = t.TypeVar("R", bound=XMLModel)
InputT = t.TypeVar("InputT", bound=XMLModel)


class Prompt(t.Generic[P, R]):
    """
    A prompt that wraps a function and uses a generator to produce structured outputs.

    The decorated function's docstring becomes the system prompt, and the input
    XMLModel is serialized to XML for the user message. The response is parsed
    into the return type XMLModel.
    """

    def __init__(
        self,
        func: t.Callable[P, R],
        *,
        generator: Generator | None = None,
    ) -> None:
        self.func = func
        self.generator = generator
        self._system_prompt = inspect.getdoc(func) or ""

        # Extract input and output types from signature
        sig = inspect.signature(func)
        hints = t.get_type_hints(func)

        self._input_params: dict[str, type[XMLModel]] = {}
        for param_name, param in sig.parameters.items():
            if param_name in hints:
                param_type = hints[param_name]
                if isinstance(param_type, type) and issubclass(param_type, XMLModel):
                    self._input_params[param_name] = param_type

        self._output_type: type[R] | None = None
        if "return" in hints:
            return_type = hints["return"]
            if isinstance(return_type, type) and issubclass(return_type, XMLModel):
                self._output_type = return_type

        functools.update_wrapper(self, func)

    def bind(self, generator: Generator) -> BoundPrompt[P, R]:
        """Bind this prompt to a specific generator."""
        return BoundPrompt(self, generator)

    def with_generator(self, generator: Generator) -> Prompt[P, R]:
        """Create a new prompt with the specified generator."""
        new_prompt = Prompt(self.func, generator=generator)
        return new_prompt

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the prompt with the bound generator."""
        if self.generator is None:
            raise RuntimeError("Prompt has no generator bound. Use .bind(generator) first.")
        return await BoundPrompt(self, self.generator)(*args, **kwargs)


class BoundPrompt(t.Generic[P, R]):
    """A prompt bound to a specific generator."""

    def __init__(self, prompt: Prompt[P, R], generator: Generator) -> None:
        self.prompt = prompt
        self.generator = generator

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the prompt and return the parsed result."""
        from dreadnode.core.generators.message import Message

        if self.prompt._output_type is None:
            raise RuntimeError("Prompt has no XMLModel return type defined.")

        # Build the user message from input arguments
        sig = inspect.signature(self.prompt.func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        user_content_parts: list[str] = []
        for param_name, value in bound.arguments.items():
            if isinstance(value, XMLModel):
                user_content_parts.append(value.to_pretty_xml())
            elif value is not None:
                user_content_parts.append(str(value))

        user_content = "\n\n".join(user_content_parts)

        # Build expected output format hint
        output_example = self.prompt._output_type.xml_example()
        format_hint = f"\n\nRespond with the following XML structure:\n{output_example}"

        # Build messages
        messages = [
            Message(role="system", content=self.prompt._system_prompt + format_hint),
            Message(role="user", content=user_content),
        ]

        # Generate response
        from dreadnode.core.generators.generator import GenerateParams

        results = await self.generator.generate_messages(
            [messages],
            [GenerateParams()],
        )

        if not results:
            raise RuntimeError("Generator returned no results")

        result = results[0]
        if isinstance(result, BaseException):
            raise result

        # Parse the response
        response_content = result.message.content
        parsed, _ = self.prompt._output_type.one_from_text(response_content)
        return parsed


@t.overload
def prompt(
    func: None = None,
    /,
    *,
    generator: Generator | None = None,
) -> t.Callable[[t.Callable[P, R]], Prompt[P, R]]: ...


@t.overload
def prompt(
    func: t.Callable[P, R],
    /,
) -> Prompt[P, R]: ...


def prompt(
    func: t.Callable[P, R] | None = None,
    /,
    *,
    generator: Generator | None = None,
) -> t.Callable[[t.Callable[P, R]], Prompt[P, R]] | Prompt[P, R]:
    """
    Decorator to convert a function into a structured prompt.

    The function should have:
    - XMLModel types for input parameters
    - An XMLModel return type
    - A docstring that serves as the system prompt

    Example:
        ```python
        class Input(XMLModel):
            question: str = element()

        class Output(XMLModel):
            answer: str = element()

        @prompt()
        def ask(input: Input) -> Output:
            '''Answer the user's question.'''

        # Use with a generator
        result = await ask.bind(generator)(Input(question="What is 2+2?"))
        ```

    Args:
        func: The function to convert.
        generator: Optional default generator to use.

    Returns:
        A Prompt object that can be bound to a generator and called.
    """

    def decorator(f: t.Callable[P, R]) -> Prompt[P, R]:
        return Prompt(f, generator=generator)

    if func is not None:
        return decorator(func)
    return decorator
