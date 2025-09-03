import typing as t

import rigging as rg

import dreadnode as dn
from dreadnode.optimization.trial import Trial
from dreadnode.transforms import Transform

AnyDict = dict[str, t.Any]


def default_trial_formatter(trial: Trial[str]) -> str:
    """A default formatter that converts a trial into a human-readable summary string."""
    response = "[unknown]"
    if trial.eval_result and trial.eval_result.samples:
        sample_output = trial.eval_result.samples[0].output
        response = str(sample_output)
        if isinstance(sample_output, dict):
            response = sample_output.get("output", response)
        if isinstance(sample_output, rg.Chat):
            response = sample_output.last.content

    return f"""
    <attempt score={trial.score:.2f}>
        <prompt>{trial.candidate}</prompt>
        <response>{response}</response>
    </attempt>
    """


def refine_prompt(
    model: str | rg.Generator,
    guidance: str,
    *,
    context_formatter: t.Callable[[Trial[str]], str] = default_trial_formatter,
) -> Transform[list[Trial[str]], str]:
    """Creates a transform that uses an LLM to reflect on trial history and generate a new prompt."""

    refiner = dn.transforms.llm_refine(model=model, guidance=guidance)

    async def refine_with_context(context: list[Trial[str]]) -> str:
        context_parts = [context_formatter(trial) for trial in context]
        context = "\n---\n".join(context_parts)
        return await refiner(context)

    return Transform(refine_with_context, name="refine_prompt")
