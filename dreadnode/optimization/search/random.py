import math
import random
import typing as t

import numpy as np

from dreadnode.common_types import AnyDict
from dreadnode.data_types import Image
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

    return Search(search, name="random")


def random_image_search(shape: tuple[int, ...], *, seed: int | None = None) -> Search[Image]:
    """
    A simple search strategy that generates a fixed number of random noise images.

    Args:
        shape: The shape of the images to generate (e.g., (224, 224, 3)).
    """

    np_random = np.random.default_rng(seed)

    async def search(_: OptimizationContext) -> t.AsyncGenerator[Trial[Image], None]:
        while True:
            random_array = np_random.integers(0, 256, size=shape, dtype=np.uint8)
            yield Trial(candidate=Image(random_array))

    return Search(search, name="random_image")
