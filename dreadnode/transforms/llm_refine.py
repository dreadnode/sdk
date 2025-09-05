import typing as t
from textwrap import dedent

import rigging as rg

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform
from dreadnode.types import AnyDict

if t.TYPE_CHECKING:
    from dreadnode.optimization.trial import Trials

T = t.TypeVar("T")


class Input(rg.Model):
    guidance: str = rg.element()
    context: str = rg.element()


class Refinement(rg.Model):
    reasoning: str = rg.element()
    prompt: str = rg.element()


@rg.prompt
def refine(input: Input) -> Refinement:  # type: ignore [empty-body]
    """
    You will improve, refine, and create an updated prompt based on context and guidance.
    """


def llm_refine(
    model: str | rg.Generator,
    guidance: str,
    *,
    model_params: AnyDict | None = None,
    name: str = "llm_refine",
) -> Transform[t.Any, str]:
    """
    A generic transform that uses an LLM to refine a candidate.

    Args:
        model: The model to use for refining the candidate.
        guidance: The guidance to use for refining the candidate. Can be a string or a Lookup that resolves to a string.
        model_params: Optional model parameters (e.g. temperature, max_tokens)
        name: The name of the transform.
    """

    async def transform(
        object: t.Any,
        *,
        model: str | rg.Generator = Config(model, help="The model to use", expose_as=str),  # noqa: B008
        guidance: str = guidance,
        model_params: AnyDict | None = model_params,
    ) -> str:
        generator: rg.Generator
        if isinstance(model, str):
            generator = rg.get_generator(
                model,
                params=rg.GenerateParams.model_validate(model_params) if model_params else None,
            )
        elif isinstance(model, rg.Generator):
            generator = model
        else:
            raise TypeError("Model must be a string identifier or a Generator instance.")

        refiner_input = Input(context=str(object), guidance=guidance)
        refinement = await refine.bind(generator)(refiner_input)
        return refinement.prompt

    return Transform(transform, name=name)


def prompt_trials_adapter(trials: "Trials[str]") -> str:
    """
    Adapter which can be used to create attempt context from a set of prompt/response trials.

    Trials are assumed to be a str candidate holding the prompt, and an output object
    that is (or includes) the model's response to the prompt.

    The list is assumed to be ordered by relevancy, and is reversed when
    formatting so the context is presented in ascending order of relevancy to the model.
    """
    context_parts = [
        dedent(f"""
        <attempt score={trial.score:.2f}>
            <prompt>{trial.candidate}</prompt>
            <response>{trial.output}</response>
        </attempt>
        """)
        for trial in reversed(trials)
    ]
    return "\n".join(context_parts)
