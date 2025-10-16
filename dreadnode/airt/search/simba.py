import typing as t

import numpy as np
from loguru import logger

from dreadnode.airt.search.image_utils import clip, get_random
from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.image import Norm


def simba_search(
    original: Image,
    *,
    theta: float = 0.1,
    num_masks: int = 500,
    objective: str | None = None,
    norm: Norm = "l2",
    max_iterations: int = 10_000,
    seed: int | None = None,
) -> Search[Image]:
    """
    Implements the SimBA (Simple Black-box Attack) algorithm for generating
    adversarial examples in a black-box setting.

    This method iteratively perturbs the original image using random noise
    masks and retains perturbations that improve the adversarial objective.

    Args:
        original: The original, non-adversarial image.
        theta: The magnitude of each perturbation step.
        num_masks: The number of random noise masks to generate and use.
        objective: The name of the objective to use for scoring candidates.
        norm: The distance metric to use for generating noise masks.
        max_iterations: The maximum number of iterations to perform.
        seed: Optional random seed for reproducibility.

    Returns:
        A Search that yields Trials with perturbed images.
    """

    random_generator = np.random.default_rng(seed)  # nosec

    async def search(
        _: OptimizationContext,
        *,
        theta: float = theta,
        num_masks: int = num_masks,
        objective: str | None = objective,
        norm: Norm = norm,
        max_iterations: int = max_iterations,
    ) -> t.AsyncGenerator[Trial[Image], None]:
        logger.info(
            "Starting SimBA search: "
            f"theta={theta}, "
            f"num_masks={num_masks}, "
            f"objective='{objective}', "
            f"norm='{norm}', "
            f"max_iterations={max_iterations}"
        )

        start_trial = Trial(candidate=original)
        yield start_trial
        await start_trial

        best_score = start_trial.get_directional_score(objective)
        original_array = original.to_numpy()

        mask_shape = (num_masks, *list(original.shape))
        logger.info(f"Generating {num_masks} random masks with shape {mask_shape}")
        mask_collection = get_random(mask_shape, norm, seed=seed) * theta
        current_mask = np.zeros_like(original_array)

        for iteration in range(1, max_iterations + 1):
            mask_idx = random_generator.choice(mask_collection.shape[0])
            new_mask = mask_collection[mask_idx]
            logger.trace(f"[{iteration}] Mask index: {mask_idx}")
            masked_array = clip(original_array + current_mask + new_mask, 0, 1)

            trial = Trial(candidate=Image(masked_array))
            yield trial
            await trial

            new_score = trial.get_directional_score(objective)
            is_better = new_score > best_score

            logger.info(
                f"[{iteration}] Trial: {trial} (better: {is_better}, best so far: {best_score:.5f})"
            )

            if not is_better:
                continue

            best_score = new_score
            current_mask = current_mask + new_mask

    return Search(search, name="simba")
