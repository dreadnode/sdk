import rigging as rg

from dreadnode.meta import Config
from dreadnode.transforms.base import Transform
from dreadnode.types import AnyDict


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
    model: str,
    guidance: str,
    *,
    model_params: AnyDict | None = None,
    name: str = "llm_refine",
) -> Transform[str]:
    """
    A generic transform that uses an LLM to refine a candidate.

    Args:
        model: The model to use for refining the candidate.
        guidance: The guidance to use for refining the candidate. Can be a string or a Lookup that resolves to a string.
        model_params: Optional model parameters (e.g. temperature, max_tokens)
        name: The name of the transform.
    """

    async def transform(object: str, *, model: str = Config(model, help="The model to use")) -> str:
        nonlocal guidance

        generator: rg.Generator
        if isinstance(model, str):
            generator = rg.get_generator(
                model, params=rg.GenerateParams.model_validate(model_params) or rg.GenerateParams()
            )
        elif isinstance(model, rg.Generator):
            generator = model
        else:
            raise TypeError("Model must be a string identifier or a Generator instance.")

        refiner_input = Input(context=str(object), guidance=guidance)
        refinement = await refine.bind(generator)(refiner_input)
        return refinement.prompt

    return Transform(transform, name=name)
