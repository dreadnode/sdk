import typing as t

import typing_extensions as te
from pydantic import ConfigDict, FilePath

import dreadnode as dn
from dreadnode.meta import Model
from dreadnode.meta.types import Config
from dreadnode.optimization import Study, Trial
from dreadnode.optimization.search.beam import BeamSearch
from dreadnode.transforms import Transform
from dreadnode.types import AnyDict

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

InputDataset = list[In]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]


class Attack(Model, t.Generic[In, Out]):
    """
    Prepared evaluation of a task with an associated dataset and configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)
    """A generative red teaming attack configuration.
    """

    dataset: t.Annotated[InputDataset[In] | list[AnyDict] | FilePath, Config(expose_as=FilePath)]
    """The initial prompt to start the attack from."""
    search_strategy: dn.Search[str] = Config(
        description="The search strategy to use for generating new prompts."
    )
    objective: dn.Scorer = Config(description="The objective scorer to optimize.")
    transforms: Transform[list[Trial[str]], str] = Config(
        description="A transform that generates new prompt candidates from trial history."
    )
    prompt_param_name: str = Config(
        default="prompt",
        description="The name of the argument in `target_task` that accepts the prompt.",
    )
    beam_width: int = Config(default=3, description="The width of the beam search.")
    branching_factor: int = Config(
        default=3, description="How many new candidates to generate from each beam."
    )
    max_steps: int = Config(default=10, description="The maximum number of optimization steps.")

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

    # make_search_strategy?
    search_strategy = BeamSearch[str](
        transform=self.transforms,
        initial_candidate=initial_prompt,
        beam_width=beam_width,
        branching_factor=branching_factor,
    )

    # This function creates a runnable task for a given candidate prompt.
    # It uses `.configure` to inject the prompt into the user's target task.
    def make_attack(prompt: str) -> dn.Task:
        return target_task.configure(**{prompt_param_name: prompt})

        return Study[str](
            strategy=search_strategy,
            apply_candidate_fn=apply_candidate,
            objective=objective,
            dataset=[{}],  # This attack is dataset-agnostic.
            max_steps=max_steps,
            direction="maximize",
            target_score=1.0,
        )

    def stream(self) -> t.Iterator[dn.optimization.events.StudyEvent]:
        """
        Execute the attack study and yield events as they occur.
        """
        study = self.make_study()
        yield from study.stream()

    def run(self) -> dn.optimization.Study[str]:
        """
        Execute the attack study and return the completed study object.
        """
        study = self.make_study()
        study.run()
        return study

    def console(self) -> None:
        """
        Run the attack and display a live console dashboard of progress.
        """
        study = self.make_study()
        study.console()

    # Move to Transforms?

    # def iterative_prompt_refiner(
    #     model: str,
    #     guidance: str,
    #     *,
    #     context_formatter: t.Callable[[Trial[str]], str] = default_trial_formatter,
    #     history_lookback: int = 3,
    #     name: str = "llm_prompt_refiner",
    # ) -> Transform:
    #     """
    #     Creates a refinement transform that uses an LLM to reflect on trial history.

    #     This is a high-level helper that abstracts away the boilerplate of formatting
    #     the trial path and calling a refinement model.

    #     Args:
    #         model: The generator model to use for refinement (e.g., "gpt-4-turbo").
    #         guidance: The core instruction for the refiner LLM.
    #         context_formatter: A function to format each trial into a string for context.
    #                         Defaults to a standard summary.
    #         history_lookback: The number of recent trials to include in the context.
    #         name: The name of the resulting transform.
    #     """

    #     async def refine_from_history(path: list[Trial[str]]) -> str:
    #         """
    #         Analyzes the trial history and generates a new, improved prompt.
    #         This function is generated and configured by create_prompt_refiner.
    #         """
    #         recent_history = path[-history_lookback:]
    #         context_parts = [context_formatter(trial) for trial in recent_history]
    #         context = "\n---\n".join(context_parts)

    #         refiner = dn.transforms.llm_refine(model=model, guidance=guidance)
    #         return await refiner(context)

    #     return Transform(refine_from_history, name=name)
