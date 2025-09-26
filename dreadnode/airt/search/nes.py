import typing as t

import numpy as np
from loguru import logger

from dreadnode.airt.search.image_utils import clip
from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.image import image_distance


def nes_search(
    original: Image,
    *,
    objective: str | None = None,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
    num_samples: int = 64,
    sigma: float = 0.001,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    seed: int | None = None,
) -> Search[Image]:
    """
    Implements a Natural Evolution Strategies (NES) based search for black-box attacks.

    This method estimates the full, dense gradient of the objective function by
    querying the model along multiple random, high-dimensional directions. It offers an
    excellent balance of query efficiency and gradient accuracy, making it a powerful
    technique for black-box optimization.

    Args:
        original: The original, non-adversarial image.
        objective: The name of the objective to use for scoring candidates.
        max_iterations: The number of main optimization iterations to perform.
        learning_rate: The step size for updating the image based on the
                       estimated gradient.
        num_samples: The number of random direction vectors to sample for each
                     gradient estimate. Total queries per iteration will be
                     (2 * num_samples) + 1.
        sigma: The exploration variance (magnitude of the random perturbations).
        adam_beta1: The beta1 parameter for the Adam optimizer.
        adam_beta2: The beta2 parameter for the Adam optimizer.
        adam_epsilon: The epsilon parameter for the Adam optimizer.
        seed: Optional random seed for reproducibility.

    Returns:
        A Search that yields Trials with perturbed images.
    """

    random_generator = np.random.default_rng(seed)

    async def search(
        _: OptimizationContext,
        *,
        objective: str | None = objective,
        max_iterations: int = max_iterations,
        learning_rate: float = learning_rate,
        num_samples: int = num_samples,
        sigma: float = sigma,
        adam_beta1: float = adam_beta1,
        adam_beta2: float = adam_beta2,
        adam_epsilon: float = adam_epsilon,
    ) -> t.AsyncGenerator[Trial[Image], None]:
        logger.info(
            "Starting NES search: "
            f"objective='{objective}', "
            f"max_iterations={max_iterations}, "
            f"learning_rate={learning_rate}, "
            f"num_samples={num_samples}, "
            f"sigma={sigma}"
        )

        # Start with the original image
        start_trial = Trial(candidate=original)
        yield start_trial
        await start_trial

        best_score = start_trial.get_directional_score(objective)
        current_best_array = original.to_numpy()
        original_array = current_best_array.copy()

        adam_m = np.zeros_like(original_array, dtype=np.float32)  # (momentum)
        adam_v = np.zeros_like(original_array, dtype=np.float32)  # (adaptive scaling)

        for iteration in range(1, max_iterations + 1):
            # 1. Generate N random perturbation vectors - positive and negative probes

            perturbation_vectors = random_generator.standard_normal((num_samples, *original.shape))

            probe_trials_p: list[Trial[Image]] = []
            probe_trials_n: list[Trial[Image]] = []

            for p_vec in perturbation_vectors:
                perturbed_p = current_best_array + sigma * p_vec
                probe_trials_p.append(
                    Trial(candidate=Image(clip(perturbed_p, 0, 1)), is_probe=True)
                )

                perturbed_n = current_best_array - sigma * p_vec
                probe_trials_n.append(
                    Trial(candidate=Image(clip(perturbed_n, 0, 1)), is_probe=True)
                )

            all_probes = probe_trials_p + probe_trials_n
            for trial in all_probes:
                yield trial

            await Trial.wait_for(*all_probes)

            # 2. Collect scores and use them to weight the perturbation vectors

            scores_p = np.array([t.get_directional_score(objective) for t in probe_trials_p])
            scores_n = np.array([t.get_directional_score(objective) for t in probe_trials_n])
            score_diffs = scores_p - scores_n

            logger.info(
                f"[{iteration}] Score Diffs: "
                f"mean={np.mean(score_diffs):.5f}, "
                f"max={np.max(score_diffs):.5f}, "
                f"min={np.min(score_diffs):.5f}, "
                f"std={np.std(score_diffs):.5f}"
            )

            # 3. Estimate the full gradient as the weighted average of the random directions
            #
            # We reshape score_diffs to (num_samples, 1, 1, 1) to enable broadcasting
            # against the perturbation_vectors array of shape (num_samples, C, H, W).

            weights = score_diffs.reshape(num_samples, *([1] * original.to_numpy().ndim))
            estimated_gradient = np.mean(weights * perturbation_vectors, axis=0)
            gradient_norm = np.linalg.norm(estimated_gradient)

            logger.info(
                f"[{iteration}] Estimated gradient: "
                f"mean={np.mean(estimated_gradient):.5f}, "
                f"max={np.max(estimated_gradient):.5f}, "
                f"min={np.min(estimated_gradient):.5f}, "
                f"norm={gradient_norm:.5f}"
            )

            # 4. Apply a gradient update step using Adam and our learning rate

            adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * estimated_gradient
            adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * (estimated_gradient**2)
            m_hat = adam_m / (1 - adam_beta1**iteration)
            v_hat = adam_v / (1 - adam_beta2**iteration)

            update_step = learning_rate * m_hat / (np.sqrt(v_hat) + adam_epsilon)
            new_array = current_best_array + update_step
            perturbation = new_array - original_array
            final_array = clip(original_array + perturbation, 0, 1)

            trial = Trial(candidate=Image(final_array))
            yield trial
            await trial

            new_score = trial.get_directional_score(objective)
            is_better = new_score > best_score

            distance = (await image_distance(original)(Image(final_array))).value

            logger.info(
                f"[{iteration}] Update Step: norm={np.linalg.norm(update_step):.5f}, score={new_score:.5f}, distance={distance:.5f}, is_better={is_better}"
            )

            if is_better:
                best_score = new_score
                current_best_array = final_array

    return Search(search, name="nes")
