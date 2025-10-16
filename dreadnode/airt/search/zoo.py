import math
import typing as t
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from dreadnode.airt.search.image_utils import clip, create_resizer
from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial

SAMPLING_PROB_MIN_FEATURES = 12288  # 64 * 64 * 3


@dataclass
class ScalingStage:
    start_iteration: int
    shape: tuple[int, ...]
    transition_upscaler: t.Callable[[NDArray[t.Any]], NDArray[t.Any]] | None
    final_upscaler: t.Callable[[NDArray[t.Any]], NDArray[t.Any]]

    @cached_property
    def num_features(self) -> int:
        return int(np.prod(self.shape))


def _create_scaling_plan(
    schedule: list[tuple[int, float]] | None,
    original_shape: tuple[int, ...],
    max_iterations: int,
) -> list[ScalingStage]:
    """Processes the schedule to create a concrete plan for the search loop."""

    plan: list[ScalingStage] = []
    last_shape = None
    cumulative_iterations = 1

    if not schedule:
        schedule = [(max_iterations, 1.0)]

    if len(original_shape) == 3:
        h, w, c = original_shape  # H, W, C format for color images
    else:
        h, w = original_shape
        c = 1  # Grayscale has one channel conceptually

    for iterations, ratio in schedule:
        # Calculate new height and width that preserve aspect ratio
        total_pixels = h * w
        new_total_pixels = total_pixels * ratio
        aspect_ratio = w / h if h > 0 else 1.0
        new_h = int(math.sqrt(new_total_pixels / aspect_ratio))
        new_w = int(new_h * aspect_ratio)

        shape = (
            (max(1, new_h), max(1, new_w), c)
            if len(original_shape) == 3
            else (max(1, new_h), max(1, new_w))
        )

        plan.append(
            ScalingStage(
                start_iteration=cumulative_iterations,
                shape=shape,
                transition_upscaler=create_resizer(last_shape, shape) if last_shape else None,
                final_upscaler=create_resizer(shape, original_shape),
            )
        )
        last_shape = shape
        cumulative_iterations += iterations

    return plan


def zoo_search(  # noqa: PLR0915
    original: Image,
    *,
    objective: str | None = None,
    max_iterations: int = 10_000,
    learning_rate: float = 0.01,
    num_samples: int = 64,
    epsilon: float = 0.01,
    scaling_schedule: list[tuple[int, float]] | None = None,
    importance_sampling_freq: int = 10,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    seed: int | None = None,
) -> Search[Image]:
    """
    Implements a Zeroth-Order Optimization (ZOO) search for black-box settings.

    See: ZOO - https://arxiv.org/abs/1708.03999

    Args:
        original: The original, non-adversarial image.
        objective: The name of the objective to use for scoring candidates.
        max_iterations: The number of optimization iterations to perform.
        learning_rate: The step size for updating the perturbation.
        num_samples: The number of random pixels to sample at each iteration
            to estimate the gradient. A higher number is more accurate
            but requires more model queries.
        epsilon: The small perturbation value used for finite difference
            gradient estimation.
        scaling_schedule: A list of tuples `(iterations, ratio)` to define a hierarchical attack.
            - `iterations`: The number of iterations to run for this stage.
            - `ratio`: A float (0.0 to 1.0) for the attack space size relative
              to the original image's total pixels.
        importance_sampling_freq: If provided, biases coordinate selection based on historical
            gradient magnitudes and defines how often (in iterations) to update the probabilities.
            Set to `0` to disable importance sampling.
        adam_beta1: The beta1 parameter for the Adam optimizer.
        adam_beta2: The beta2 parameter for the Adam optimizer.
        adam_epsilon: The epsilon parameter for the Adam optimizer.
        seed: Optional random seed for reproducibility.

    Returns:
        A Search that yields Trials with perturbed images.
    """

    random_generator = np.random.default_rng(seed)

    async def search(  # noqa: PLR0915
        _: OptimizationContext,
        *,
        objective: str | None = objective,
        max_iterations: int = max_iterations,
        learning_rate: float = learning_rate,
        num_samples: int = num_samples,
        epsilon: float = epsilon,
        scaling_schedule: list[tuple[int, float]] | None = scaling_schedule,
        importance_sampling_freq: int = importance_sampling_freq,
        adam_beta1: float = adam_beta1,
        adam_beta2: float = adam_beta2,
        adam_epsilon: float = adam_epsilon,
    ) -> t.AsyncGenerator[Trial[Image], None]:
        # 1. Initialization

        original_array = original.to_numpy()
        scaling_plan = _create_scaling_plan(scaling_schedule, original.shape, max_iterations)
        logger.debug(f"Resolved Scaling Plan: {scaling_plan}")

        start_trial = Trial(candidate=original)
        yield start_trial
        await start_trial
        best_score = start_trial.get_directional_score(objective)

        next_stage_idx = 1
        scaling_step = scaling_plan[0]

        current_perturbation = np.zeros(scaling_step.shape, dtype=np.float32)
        current_best_perturbation = current_perturbation.copy()
        adam_m = np.zeros_like(current_perturbation)
        adam_v = np.zeros_like(current_perturbation)

        sampling_probs = (
            np.ones(scaling_step.num_features, dtype=np.float32) / scaling_step.num_features
            if importance_sampling_freq is not None
            else None
        )

        logger.info(
            f"Starting ZOO search: "
            f"objective='{objective}', "
            f"max_iterations={max_iterations}, "
            f"learning_rate={learning_rate}, "
            f"num_samples={num_samples}, "
            f"epsilon={epsilon}, "
            f"shape={scaling_step.shape}, "
            f"features={scaling_step.num_features}, "
            f"scaling_schedule={scaling_schedule}, "
            f"importance_sampling_freq={importance_sampling_freq}, "
            f"adam_beta1={adam_beta1}, "
            f"adam_beta2={adam_beta2}, "
            f"adam_epsilon={adam_epsilon}, "
            f"seed={seed}"
        )

        # 2. Main Optimization Loop

        for iteration in range(1, max_iterations + 1):
            # 2a. Check for scaling transition
            if (
                next_stage_idx < len(scaling_plan)
                and iteration == scaling_plan[next_stage_idx].start_iteration
            ):
                scaling_step = scaling_plan[next_stage_idx]
                next_stage_idx += 1

                logger.info(
                    f"[{iteration}] Rescaling: "
                    f"shape={scaling_step.shape}, "
                    f"num_features={scaling_step.num_features}"
                )

                # Upscale the perturbation and ADAM state
                upscaler = scaling_step.transition_upscaler
                if upscaler:
                    current_perturbation = upscaler(current_perturbation)
                    current_best_perturbation = upscaler(current_best_perturbation)
                    adam_m = upscaler(adam_m)
                    adam_v = upscaler(adam_v)

                # Reset importance sampling for the new stage
                sampling_probs = np.ones(scaling_step.num_features) / scaling_step.num_features

            def make_image(array: NDArray[t.Any]) -> Image:
                full_p = scaling_step.final_upscaler(array)  # noqa: B023
                final_array = clip(original_array + full_p, 0, 1)
                return Image(final_array)

            # 2b. Sample coordinates, create probes, and estimate gradient

            sample_indices_flat = random_generator.choice(
                scaling_step.num_features, num_samples, replace=False, p=sampling_probs
            )
            sample_indices_nd = [
                np.unravel_index(idx, scaling_step.shape) for idx in sample_indices_flat
            ]

            probes_p: list[Trial[Image]] = []
            probes_n: list[Trial[Image]] = []
            for idx_nd in sample_indices_nd:
                basis_vector = np.zeros(scaling_step.shape, dtype=np.float32)
                basis_vector[idx_nd] = 1.0
                probes_p.append(
                    Trial(
                        candidate=make_image(current_perturbation + epsilon * basis_vector),
                        is_probe=True,
                    )
                )
                probes_n.append(
                    Trial(
                        candidate=make_image(current_perturbation - epsilon * basis_vector),
                        is_probe=True,
                    )
                )

            all_probes = probes_p + probes_n
            for trial in all_probes:
                yield trial

            await Trial.wait_for(*all_probes)

            scores_p = np.array([t.get_directional_score(objective) for t in probes_p])
            scores_n = np.array([t.get_directional_score(objective) for t in probes_n])
            score_diffs = scores_p - scores_n

            logger.info(
                f"[{iteration}] Score Diffs: "
                f"mean={np.mean(score_diffs):.5f}, "
                f"max={np.max(score_diffs):.5f}, "
                f"min={np.min(score_diffs):.5f}, "
                f"std={np.std(score_diffs):.5f}"
            )

            estimated_gradient = np.zeros_like(current_perturbation)
            for i, idx_nd in enumerate(sample_indices_nd):
                score_p = probes_p[i].get_directional_score(objective)
                score_n = probes_n[i].get_directional_score(objective)
                estimated_gradient[idx_nd] = (score_p - score_n) / (2 * epsilon)

            # 2c. Update with Adam

            adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * estimated_gradient
            adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * (estimated_gradient**2)
            m_hat = adam_m / (1 - adam_beta1**iteration)
            v_hat = adam_v / (1 - adam_beta2**iteration)
            update_step = learning_rate * m_hat / (np.sqrt(v_hat) + adam_epsilon)
            current_perturbation += update_step

            # 2d. Evaluate new candidate

            trial = Trial(candidate=make_image(current_perturbation))
            yield trial
            await trial

            new_score = trial.get_directional_score(objective)
            is_better = new_score > best_score

            distance = np.linalg.norm(current_best_perturbation)

            logger.info(
                f"[{iteration}] Update Step: "
                f"norm={np.linalg.norm(update_step):.5f}, "
                f"score={new_score:.5f}, "
                f"distance={distance:.5f}, "
                f"is_better={is_better}"
            )

            if is_better:
                best_score = new_score
                current_best_perturbation = current_perturbation.copy()

            # 2e. Update Importance Sampling
            #
            # Follow the paper and only activate this when we have a sufficiently large
            # attack space (>= 64x64x3 = 12288 features).

            if (
                scaling_step.num_features >= SAMPLING_PROB_MIN_FEATURES
                and importance_sampling_freq
                and iteration % importance_sampling_freq == 0
            ):
                importance = np.abs(m_hat).flatten()
                sum_importance = np.sum(importance)
                if sum_importance > 1e-8:
                    sampling_probs = importance / sum_importance

    return Search(search, name="zoo")
