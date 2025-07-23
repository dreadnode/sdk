import typing as t

from rigging import GenerateParams, get_generator
from rigging.generator import Generator
from rigging.model import Model, element
from rigging.prompt import prompt

from dreadnode.metric import Metric, Scorer
from dreadnode.task import TaskInput


class JudgeInput(Model):
    input: str | None = element(default=None)
    expected_output: str | None = element(default=None)
    output: str = element()
    rubric: str = element()


class Judgement(Model):
    reason: str = element()
    pass_: bool = element(alias="pass")
    score: float = element()


@prompt()
def judge(input: JudgeInput) -> Judgement:  # type: ignore [empty-body]
    """
    You are grading output according to a user-specified rubric. \
    If the statement in the rubric is true for the provided input and output, then the output passes the test.
    Assign a score based on the rubric, where applicable, otherwise 1.0 for passing and 0.0 for failing.
    """


def llm_judge(
    model: "str | Generator | TaskInput",
    rubric: str | TaskInput,
    *,
    expected_output: str | TaskInput | None = None,
    params: "GenerateParams | None" = None,
    passing: t.Callable[[float], bool] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    name: str = "llm_judge",
) -> "Scorer[t.Any]":
    """
    Score the output of a task using an LLM to judge it against a rubric.

    Args:
        model: The model to use for judging. Can be a string identifier (rigging), a Generator instance
            or a TaskInput that resolves to a string identifier.
        rubric: The rubric to use for judging. Can be a string or a TaskInput that resolves to a string.
        expected_output: The expected output to compare against, if applicable. Can be a string or a TaskInput that resolves to a string.
        params: Optional parameters for the generator.
        passing: Optional callback to determine if the score is passing based on the score value - overrides any model-specified value.
        min_score: Optional minimum score for the judgement - if provided, the score will be clamped to this value.
        max_score: Optional maximum score for the judgement - if provided, the score will be clamped to this value.
        name: The name of the scorer.
    """

    async def evaluate(data: t.Any) -> Metric:
        _model = model.resolve() if isinstance(model, TaskInput) else model
        _rubric = rubric.resolve(cast_as=str) if isinstance(rubric, TaskInput) else rubric
        _expected_output = (
            expected_output.resolve(cast_as=str)
            if isinstance(expected_output, TaskInput)
            else expected_output
        )

        generator: Generator
        if isinstance(_model, str):
            generator = get_generator(_model, params=params or GenerateParams())
        elif isinstance(_model, Generator):
            generator = _model
        else:
            raise TypeError("Model must be a string identifier or a Generator instance.")

        input_data = JudgeInput(
            input=str(data),
            expected_output=_expected_output,
            output=str(data),
            rubric=_rubric,
        )

        judgement = await judge.bind(generator)(input_data)

        if min_score is not None:
            judgement.score = max(min_score, judgement.score)
        if max_score is not None:
            judgement.score = min(max_score, judgement.score)

        if passing is not None:
            judgement.pass_ = passing(judgement.score)

        return Metric(
            value=judgement.score,
            attributes={
                "reason": judgement.reason,
                "pass": judgement.pass_,
            },
        )

    return Scorer.from_callable(evaluate, name=name, catch=True)
