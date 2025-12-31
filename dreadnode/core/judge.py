import typing as t
from abc import abstractmethod
from dataclasses import dataclass

from dreadnode.core.agents.result import AgentResult
from dreadnode.core.metric import Metric

R_Rubric = t.TypeVar("R_Rubric", bound="Rubric")


@dataclass
class Rubric:
    """
    The base configuration for any Judge. It is a simple data container
    that defines WHAT a Judge should evaluate. The only required field is a name
    for the final metric.
    """

    name: str


class Judge(t.Generic[R_Rubric]):
    """
    The generic Judge base class. It is an abstract component that knows
    HOW to perform an evaluation.
    It is typed to a specific Rubric subclass to ensure that a Judge
    always receives the correct configuration.
    """

    def __init__(self, rubric: R_Rubric):
        self.rubric = rubric

    @abstractmethod
    def evaluate(self, result: AgentResult) -> Metric:
        """
        The core method of a Judge. It takes the entire result of an agent run
        and produces a single, final Metric based on its Rubric.
        This method MUST be implemented by all subclasses.
        """
        raise NotImplementedError
