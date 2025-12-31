import itertools
import typing as t

from loguru import logger

from dreadnode.core.optimization.trial import Trial
from dreadnode.core.search import OptimizationContext, Search
from dreadnode.core.types.common import AnyDict


def grid_search(
    search_space: dict[str, list[t.Any]],
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> Search[AnyDict]:
    """
    Creates a search strategy that exhaustively iterates over all combinations
    of the provided parameter values.

    This is useful for small, discrete search spaces where you want to evaluate
    every possible configuration.

    Args:
        search_space: A dictionary mapping parameter names to lists of values.
        shuffle: If True, randomize the order of combinations.
        seed: Random seed for shuffling (only used if shuffle=True).

    Example:
        search = grid_search({
            "model": ["claude-3-haiku", "claude-3-sonnet"],
            "temperature": [0.3, 0.7, 1.0],
            "max_steps": [10, 20],
        })
        # Yields 2 * 3 * 2 = 12 candidates
    """

    async def search(
        context: OptimizationContext,  # noqa: ARG001
        *,
        search_space: dict[str, list[t.Any]] = search_space,
        shuffle: bool = shuffle,
        seed: int | None = seed,
    ) -> t.AsyncGenerator[Trial[AnyDict], None]:
        if not search_space:
            logger.warning("Empty search space provided to grid_search")
            return

        keys = list(search_space.keys())
        value_lists = [search_space[k] for k in keys]

        combinations = list(itertools.product(*value_lists))
        total = len(combinations)

        logger.info(
            f"Starting grid search: "
            f"parameters={keys}, "
            f"total_combinations={total}, "
            f"shuffle={shuffle}"
        )

        if shuffle:
            import random

            rng = random.Random(seed)
            rng.shuffle(combinations)

        for i, values in enumerate(combinations):
            candidate = dict(zip(keys, values))
            logger.debug(f"Grid search [{i + 1}/{total}]: {candidate}")
            yield Trial(candidate=candidate)

        logger.info(f"Grid search exhausted all {total} combinations")

    return Search(search, name="grid")
