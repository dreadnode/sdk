import typing as t

import dreadnode as dn
from dreadnode.optimization import Study, Trial
from dreadnode.optimization.search import BeamSearch
from dreadnode.transforms import Transform


def generative_attack(
    initial_prompt: str,
    target_task: dn.Task,
    objective_scorer: dn.Scorer,
    refinement_transform: Transform[list[Trial[str]], str],
    *,
    prompt_param_name: str,
    beam_width: int = 3,
    branching_factor: int = 2,
    max_steps: int = 10,
) -> Study[str]:
    """
    Configures a complete generative red teaming study from its core components.

    Args:
        initial_prompt: The starting prompt for the search.
        target_task: The dn.Task to execute with each generated prompt.
        objective_scorer: A dn.Scorer that evaluates the output of the target_task.
        refinement_transform: A dn.Transform that takes a trial history (list[Trial[str]])
                              and returns a new prompt string.
        prompt_param_name: The name of the argument in `target_task` that accepts the prompt.
        beam_width: The width of the beam search.
        branching_factor: How many new candidates to generate from each beam.
        max_steps: The maximum number of optimization steps.
    """

    search_strategy = BeamSearch[str](
        transform=refinement_transform,
        initial_candidate=initial_prompt,
        beam_width=beam_width,
        branching_factor=branching_factor,
    )

    # This function creates a runnable task for a given candidate prompt.
    # It uses `.configure` to inject the prompt into the user's target task.
    def apply_candidate(prompt: str) -> dn.Task:
        return target_task.configure(**{prompt_param_name: prompt})

    from dreadnode.optimization import rebuild_event_models
    from dreadnode.optimization.search import Search  # noqa: F401
    from dreadnode.tracing.span import TaskSpan  # noqa: F401

    rebuild_event_models()
    Study.model_rebuild()

    return Study[str](
        strategy=search_strategy,
        apply_candidate_fn=apply_candidate,
        objective=objective_scorer,
        dataset=[{}],  # This attack is dataset-agnostic.
        max_steps=max_steps,
        direction="maximize",
        target_score=1.0,
    )


def default_trial_formatter(trial: Trial[str]) -> str:
    """
    A default formatter that converts a trial into a human-readable summary string.
    """
    # Safely access the results from the trial's evaluation
    output_dict = trial.eval_result.samples[0].output if trial.eval_result else {}
    response_text = output_dict.get("output", "Evaluation failed or is pending.")

    return (
        f"ATTEMPT (Score: {trial.score:.2f}):\n"
        f"  - Prompt: {trial.candidate}\n"
        f"  - Response: {response_text}"
    )


def iterative_prompt_refiner(
    model: str,
    guidance: str,
    *,
    context_formatter: t.Callable[[Trial[str]], str] = default_trial_formatter,
    history_lookback: int = 3,
    name: str = "llm_prompt_refiner",
) -> Transform:
    """
    Creates a refinement transform that uses an LLM to reflect on trial history.

    This is a high-level helper that abstracts away the boilerplate of formatting
    the trial path and calling a refinement model.

    Args:
        model: The generator model to use for refinement (e.g., "gpt-4-turbo").
        guidance: The core instruction for the refiner LLM.
        context_formatter: A function to format each trial into a string for context.
                           Defaults to a standard summary.
        history_lookback: The number of recent trials to include in the context.
        name: The name of the resulting transform.
    """

    async def refine_from_history(path: list[Trial[str]]) -> str:
        """
        Analyzes the trial history and generates a new, improved prompt.
        This function is generated and configured by create_prompt_refiner.
        """
        recent_history = path[-history_lookback:]
        context_parts = [context_formatter(trial) for trial in recent_history]
        context = "\n---\n".join(context_parts)

        refiner = dn.transforms.llm_refine(model=model, guidance=guidance)
        return await refiner(context)

    return Transform(refine_from_history, name=name)
