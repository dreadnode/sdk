import typing as t

import rigging as rg
from loguru import logger

from dreadnode.common_types import AnyDict
from dreadnode.meta import Config
from dreadnode.metric import Metric
from dreadnode.scorers import Scorer


class JudgeInput(rg.Model):
    input: str | None = rg.element(default=None)
    expected_output: str | None = rg.element(default=None)
    output: str = rg.element()
    rubric: str = rg.element()


class Judgement(rg.Model):
    reason: str = rg.element(description="The reason for the judgement.")
    passing: bool = rg.element(
        description="Whether the output passed based on the rubric (true/false)."
    )
    score: float = rg.element(
        description="The float score assigned to the output based on the rubric."
    )


@rg.prompt()
def judge(input: JudgeInput) -> Judgement:  # type: ignore [empty-body]
    """
    You are grading output according to a user-specified rubric.

    If the statement in the rubric is true for the provided input and output, then the output passes the test.
    Assign a score based on the rubric, where applicable, otherwise 1.0 for passing and 0.0 for failing.
    """


def llm_judge(
    model: str | rg.Generator,
    rubric: str,
    *,
    input: t.Any | None = None,
    expected_output: t.Any | None = None,
    model_params: rg.GenerateParams | AnyDict | None = None,
    fallback_model: str | rg.Generator | None = None,
    passing: t.Callable[[float], bool] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    name: str = "llm_judge",
    system_prompt: str | None = None,
) -> "Scorer[t.Any]":
    """
    Score the output of a task using an LLM to judge it against a rubric.

    Args:
        model: The model to use for judging.
        rubric: The rubric to use for judging.
        input: The input which produced the output for context, if applicable.
        expected_output: The expected output to compare against, if applicable.
        model_params: Optional parameters for the model.
        fallback_model: Optional fallback model to use if the primary model fails.
        passing: Optional callback to determine if the score is passing based on the score value - overrides any model-specified value.
        min_score: Optional minimum score for the judgement - if provided, the score will be clamped to this value.
        max_score: Optional maximum score for the judgement - if provided, the score will be clamped to this value.
        name: The name of the scorer.
        system_prompt: Optional custom system prompt for the judge. If None, uses default.
    """

    async def evaluate(
        data: t.Any,
        *,
        model: str | rg.Generator = Config(  # noqa: B008
            model, help="The model to use for judging.", expose_as=str
        ),
        rubric: str = rubric,
        input: t.Any | None = input,
        expected_output: t.Any | None = expected_output,
        model_params: rg.GenerateParams | AnyDict | None = model_params,
        fallback_model: str | rg.Generator | None = fallback_model,
        min_score: float | None = min_score,
        max_score: float | None = max_score,
        system_prompt: str | None = system_prompt,
    ) -> list[Metric]:
        def _create_generator(
            model: str | rg.Generator,
            params: rg.GenerateParams | AnyDict | None,
        ) -> rg.Generator:
            """Create a Generator from a model identifier or return the Generator instance."""
            if isinstance(model, str):
                return rg.get_generator(
                    model,
                    params=params
                    if isinstance(params, rg.GenerateParams)
                    else rg.GenerateParams.model_validate(params)
                    if params
                    else None,
                )
            if isinstance(model, rg.Generator):
                return model
            raise TypeError("Model must be a string identifier or a Generator instance.")

        generator = _create_generator(model, model_params)

        input_data = JudgeInput(
            input=str(input) if input is not None else None,
            expected_output=str(expected_output) if expected_output is not None else None,
            output=str(data),
            rubric=rubric,
        )

        # Track fallback usage for observability
        used_fallback = False
        primary_error: str | None = None

        # Try primary model, fallback if needed
        try:
            pipeline = generator.chat([])
            if system_prompt:
                pipeline.chat.inject_system_content(system_prompt)
            judgement = await judge.bind(pipeline)(input_data)
        except Exception as e:
            if fallback_model is None:
                raise
            # Log primary model failure and fallback usage
            used_fallback = True
            primary_error = f"{type(e).__name__}: {e}"
            primary_model_name = model if isinstance(model, str) else type(model).__name__
            fallback_model_name = (
                fallback_model if isinstance(fallback_model, str) else type(fallback_model).__name__
            )
            logger.warning(
                f"Primary model '{primary_model_name}' failed with {primary_error}. "
                f"Using fallback model '{fallback_model_name}'."
            )
            # Use fallback model
            generator = _create_generator(fallback_model, model_params)
            pipeline = generator.chat([])
            if system_prompt:
                pipeline.chat.inject_system_content(system_prompt)
            judgement = await judge.bind(pipeline)(input_data)

        if min_score is not None:
            judgement.score = max(min_score, judgement.score)
        if max_score is not None:
            judgement.score = min(max_score, judgement.score)

        if passing is not None:
            judgement.passing = passing(judgement.score)

        score_metric = Metric(
            value=judgement.score,
            attributes={
                "reason": judgement.reason,
                "used_fallback": used_fallback,
                "fallback_model": (
                    str(fallback_model)
                    if isinstance(fallback_model, str)
                    else type(fallback_model).__name__
                )
                if used_fallback
                else None,
                "primary_error": primary_error,
            },
        )
        pass_metric = Metric(value=float(judgement.passing))
        pass_metric._scorer_name = f"{name}_pass"  # type: ignore[attr-defined] # noqa: SLF001

        return [score_metric, pass_metric]

    return Scorer(evaluate, name=name)
