import math
import random
import typing as t

from dreadnode.common_types import AnyDict
from dreadnode.optimization.search.base import (
    Categorical,
    Float,
    Int,
    OptimizationContext,
    Search,
    SearchSpace,
)
from dreadnode.optimization.trial import Trial


def _sample_from_space(search_space: SearchSpace, random: random.Random) -> AnyDict:  # noqa: PLR0912
    """
    Generate a random candidate from the search space.
    """
    candidate: AnyDict = {}
    for name, dist in search_space.items():
        if isinstance(dist, Float):
            if dist.log:
                if dist.low <= 0:
                    raise ValueError("Log scale requires low > 0.")
                log_low = math.log(dist.low)
                log_high = math.log(dist.high)
                value = math.exp(random.uniform(log_low, log_high))  # nosec
            elif dist.step:
                num_steps = int((dist.high - dist.low) / dist.step)
                random_step = random.randint(0, num_steps)  # nosec
                value = dist.low + random_step * dist.step
            else:
                value = random.uniform(dist.low, dist.high)  # nosec
            candidate[name] = value

        elif isinstance(dist, Int):
            if dist.log:
                if dist.low <= 0:
                    raise ValueError("Log scale requires low > 0.")
                log_low = math.log(dist.low)
                log_high = math.log(dist.high)
                value = round(math.exp(random.uniform(log_low, log_high)))  # nosec
            elif dist.step > 1:
                num_steps = (dist.high - dist.low) // dist.step
                random_step = random.randint(0, num_steps)  # nosec
                value = dist.low + random_step * dist.step
            else:
                value = random.randint(dist.low, dist.high)  # nosec
            candidate[name] = max(dist.low, min(dist.high, value))  # check bounds after rounding

        elif isinstance(dist, Categorical):
            candidate[name] = random.choice(dist.choices)  # nosec
        elif isinstance(dist, list):
            candidate[name] = random.choice(dist)  # nosec
        else:
            raise TypeError(f"Unsupported distribution type: {type(dist)}")

    return candidate


class RandomSearch(Search[AnyDict]):
    """
    A search strategy that suggests candidates by sampling uniformly and
    independently from the search space at each step.

    This strategy is "memoryless" and does not learn from the results of
    past trials. It is primarily useful as a simple baseline for comparing
    the performance of more sophisticated optimization algorithms.
    """

    def __init__(
        self, search_space: SearchSpace, *, trials_per_step: int = 1, seed: float | None = None
    ):
        """
        Initializes the RandomSearch strategy.

        Args:
            search_space: The search space to explore.
            trials_per_step: The number of trials to suggest at each step.
        """
        self.search_space = search_space
        self.trials_per_step = trials_per_step
        self.seed = seed
        self.random = random.Random(seed)  # noqa: S311 # nosec

    def reset(self, _: OptimizationContext) -> None:
        self.random = random.Random(self.seed)  # noqa: S311 # nosec

    async def suggest(self, step: int) -> t.AsyncIterator[Trial[AnyDict]]:
        """Suggests the next batch of random candidates."""
        for _ in range(self.trials_per_step):
            candidate = _sample_from_space(self.search_space, self.random)
            yield Trial(candidate=candidate, step=step)

    async def observe(self, trials: list[Trial[AnyDict]]) -> None:
        """Informs the strategy of recent trial results. This is a no-op for RandomSearch."""
