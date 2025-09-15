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


def random_search(search_space: SearchSpace, *, seed: float | None = None) -> Search[AnyDict]:
    """
    Create a search strategy that suggests candidates by sampling uniformly and
    independently from the search space at each step.

    This strategy is "memoryless" and does not learn from the results of
    past trials. It is primarily useful as a simple baseline for comparing
    the performance of more sophisticated optimization algorithms.

    Args:
        search_space: The search space to explore.
        seed: The random seed to use for reproducibility.
    """

    async def search(
        _: OptimizationContext, *, seed: float | None = seed
    ) -> t.AsyncGenerator[Trial[AnyDict], None]:
        _random = random.Random(seed)  # noqa: S311 # nosec
        while True:
            yield Trial(candidate=_sample_from_space(search_space, _random))

    return Search(search, name="random_search")
