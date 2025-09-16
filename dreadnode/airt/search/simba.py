import typing as t

import numpy as np

from dreadnode.airt.search.image_utils import get_random
from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.image import DistanceMethod


def simba_search(
    original: Image,
    *,
    theta: float = 0.1,
    num_masks: int = 500,
    objective: str | None = None,
    distance_method: DistanceMethod = "l2",
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
        distance_method: The distance metric to use for generating noise masks.
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
    ) -> t.AsyncGenerator[Trial[Image], None]:
        start_trial = Trial(candidate=original)
        yield start_trial
        await start_trial

        best_score = start_trial.get_directional_score(objective)

        original_array = original.to_numpy()

        mask_shape = (num_masks, *list(original.shape))
        mask_collection = get_random(mask_shape, distance_method, seed=seed) * theta
        current_mask = np.zeros_like(original_array)

        while True:
            mask_idx = random_generator.choice(mask_collection.shape[0])
            new_mask = mask_collection[mask_idx]
            masked_array = np.clip(original_array + current_mask + new_mask, 0, 1)

            trial = Trial(candidate=Image(masked_array))
            yield trial
            await trial

            new_score = trial.get_directional_score(objective)
            if new_score <= best_score:
                continue

            best_score = new_score
            current_mask = current_mask + new_mask

    return Search(search, name="simba")
