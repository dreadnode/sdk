import typing as t

import rigging as rg

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
    passing: t.Callable[[float], bool] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    name: str = "llm_judge",
) -> "Scorer[t.Any]":
    """
    Score the output of a task using an LLM to judge it against a rubric.

    Args:
        model: The model to use for judging.
        rubric: The rubric to use for judging.
        input: The input which produced the output for context, if applicable.
        expected_output: The expected output to compare against, if applicable.
        model_params: Optional parameters for the model.
        passing: Optional callback to determine if the score is passing based on the score value - overrides any model-specified value.
        min_score: Optional minimum score for the judgement - if provided, the score will be clamped to this value.
        max_score: Optional maximum score for the judgement - if provided, the score will be clamped to this value.
        name: The name of the scorer.
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
        min_score: float | None = min_score,
        max_score: float | None = max_score,
    ) -> list[Metric]:
        generator: rg.Generator
        if isinstance(model, str):
            generator = rg.get_generator(
                model,
                params=model_params
                if isinstance(model_params, rg.GenerateParams)
                else rg.GenerateParams.model_validate(model_params)
                if model_params
                else None,
            )
        elif isinstance(model, rg.Generator):
            generator = model
        else:
            raise TypeError("Model must be a string identifier or a Generator instance.")

        input_data = JudgeInput(
            input=str(input) if input is not None else None,
            expected_output=str(expected_output) if expected_output is not None else None,
            output=str(data),
            rubric=rubric,
        )

        judgement = await judge.bind(generator)(input_data)

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
            },
        )
        pass_metric = Metric(value=float(judgement.passing))
        pass_metric._scorer_name = f"{name}_pass"  # type: ignore[attr-defined] # noqa: SLF001

        return [score_metric, pass_metric]

    return Scorer(evaluate, name=name)
