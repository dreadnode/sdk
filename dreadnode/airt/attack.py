import typing as t

import rigging as rg
from rigging import Generator
from rigging.transform import Transform

from dreadnode.airt.constraints.base import Constraint
from dreadnode.metric import Scorer

if t.TYPE_CHECKING:
    from dreadnode.airt.search.base import Search


class AttackConfig:
    """
    A pipeline for attacking a model where the outputs are unknown.
    """

    def __init__(
        self,
        generator: str | Generator,
        prompts: list[str],
        transforms: list[Transform] | None = None,
        scorers: list[Scorer] | None = None,
        constraints: list[Constraint] | None = None,
    ) -> None:
        self._generator = rg.get_generator(generator) if isinstance(generator, str) else generator
        self.prompts = prompts
        self.transforms = transforms or []
        self.scorers = scorers or []
        self.constraints = constraints or []
        self.results: t.Any = None

    def build_pipeline(self, input) -> rg.ChatPipeline:
        _pipeline = self._generator.chat(input)

        if self.transforms:
            _pipeline = _pipeline.transform(self.transforms)
        if self.scorers:
            _pipeline = _pipeline.score(self.scorers)

        return _pipeline

    def run(self, search_func: "Search") -> t.Any:
        """
        Run the attack using the specified search function.

        Args:
            search_func: The search function to use (e.g., beam_search, random_search).

        Returns:
            The results of the attack.
        """
        self.results = search_func(self)
        return self.results
