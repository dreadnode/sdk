import typing as t

import numpy as np
from loguru import logger

from dreadnode.airt.search.image_utils import clip, get_random
from dreadnode.data_types import Image
from dreadnode.optimization.search import bisection_image_search, random_image_search
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.image import Norm, image_distance


def hop_skip_jump_search(  # noqa: PLR0915
    source: Image,
    target: Image | None = None,
    *,
    adversarial_objective: str | None = None,
    adversarial_threshold: float = 0.0,
    norm: Norm = "l2",
    theta: float = 0.01,
    boundary_tolerance: float | None = None,
    step_size: float | None = None,
    min_evaluations: int = 50,
    max_evaluations: int = 100,
    max_iterations: int = 1_000,
) -> Search[Image]:
    """
    Implements the HopSkipJump search for black-box image classifier settings.

    See: HopSkipJumpAttack - https://arxiv.org/abs/1904.02144

    Args:
        source: The original, unperturbed image.
        target: An initial adversarial example. If not provided, a random search will be performed
            to find one that satisfies the adversarial objective and threshold.
        adversarial_objective: The name of the objective to use for the adversarial decision.
        adversarial_threshold: The threshold value for the adversarial decision.
        norm: The distance metric to use. Options are 'l2' (Euclidean
            distance), 'l1' (Manhattan distance), or 'linf' (Chebyshev distance).
        theta: The relative size of the perturbation used for gradient estimation.
        boundary_tolerance: The maximum acceptable difference between the upper and lower alpha values
            when projecting onto the decision boundary. If not provided, defaults to `theta / 10`.
        step_size: The initial step size for the line search, as a ratio of the
            current distance to the source. If not provided, defaults to `theta`.
        min_evaluations: The minimum number of model evaluations to use for gradient estimation.
        max_evaluations: The maximum number of model evaluations to use for gradient estimation.
        max_iterations: The maximum number of main iterations to perform.
    """

    async def search(  # noqa: PLR0912, PLR0915
        context: OptimizationContext,
        *,
        source: Image = source,
        target: Image | None = target,
        decision_objective: str | None = adversarial_objective,
        decision_threshold: float = adversarial_threshold,
        distance_method: Norm = norm,
        theta: float = theta,
        boundary_tolerance: float | None = boundary_tolerance,
        step_size_ratio: float | None = step_size,
        min_evaluations: int = min_evaluations,
        max_evaluations: int = max_evaluations,
        max_iterations: int = max_iterations,
    ) -> t.AsyncGenerator[Trial[Image], None]:
        def is_adversarial(trial: Trial) -> bool:
            return trial.get_directional_score(decision_objective) > decision_threshold

        step_size_ratio = step_size_ratio or theta
        boundary_tolerance = boundary_tolerance or theta / 10

        logger.info(
            f"Starting HopSkipJump: "
            f"theta={theta}, "
            f"distance_method={distance_method}, "
            f"decision_objective={decision_objective}, "
            f"decision_threshold={decision_threshold}, "
            f"min_evaluations={min_evaluations}, "
            f"max_iterations={max_iterations})"
        )

        # 1 - Bootstrap (if needed)
        #
        # Annoying here and throughout that we don't have `yield from` in async generators

        if target is None:
            logger.info("No target provided, searching for an initial adversarial example.")
            random_search = random_image_search(shape=source.shape)
            async for trial in random_search(context):
                yield trial.as_probe()
                await trial
                if is_adversarial(trial):
                    target = trial.candidate
                    logger.success(f"Found initial adversarial example: {target}")
                    break

            if target is None:
                raise RuntimeError("Failed to find an initial adversarial example.")

        # 2 - Boundary search

        bisection = bisection_image_search(
            source,
            target,
            decision_objective=decision_objective,
            decision_threshold=decision_threshold,
            tolerance=boundary_tolerance,
        )

        async for trial in bisection(context):
            yield trial.as_probe()

        # 3 - Main loop

        current_best = trial.candidate
        yield Trial(candidate=current_best)

        for iteration in range(1, max_iterations + 1):
            # 3a - Gradient estimation

            distance = (await image_distance(source, norm=distance_method)(current_best)).value
            delta = theta * distance

            num_evals = min(int(min_evaluations * np.sqrt(iteration)), max_evaluations)
            noise_shape = (num_evals, *current_best.shape)

            random_noise = get_random(noise_shape, distance_method)
            noise_norms = np.linalg.norm(random_noise.reshape(num_evals, -1), axis=1).reshape(
                num_evals, *((1,) * (len(current_best.shape)))
            )
            random_noise /= noise_norms

            current_array = current_best.to_numpy()
            perturbation_arrays = clip(current_array + delta * random_noise, 0, 1)
            random_noise = (perturbation_arrays - current_array) / delta

            logger.info(
                f"[{iteration}] Estimating gradient: "
                f"num_evals={num_evals:.5f}, "
                f"distance={distance:.5f}), "
                f"delta={delta:.5f}"
            )

            perturbed = [Trial(candidate=Image(p), is_probe=True) for p in perturbation_arrays]
            for trial in perturbed:
                yield trial

            await Trial.wait_for(*perturbed)

            satisfied = np.array(
                [is_adversarial(probe) for probe in perturbed],
                dtype=np.float32,
            )

            f_val = 2 * satisfied - 1
            crossing_ratio = np.mean(satisfied)
            if crossing_ratio in [1.0, -1.0]:
                logger.warning(
                    f"[{iteration}] All perturbed samples are on the same side of the boundary, "
                    "gradient may be inaccurate. Consider adjusting theta or increasing min_evaluations."
                )
                gradient = np.mean(random_noise, axis=0) * (crossing_ratio)
            else:
                f_val -= f_val.mean()
                f_val_reshaped = f_val.reshape(num_evals, *((1,) * len(current_array.shape)))
                gradient = np.mean(f_val_reshaped * random_noise, axis=0)

            if distance_method == "l2":
                gradient /= np.linalg.norm(gradient)
            else:
                gradient = np.sign(gradient)

            source_direction = (source.to_numpy() - current_array).flatten()
            gradient_flat = gradient.flatten()
            dot_product = np.dot(source_direction, gradient_flat)

            logger.info(
                f"[{iteration}] Gradient: "
                f"ratio={crossing_ratio:.2%}, "
                f"alignment={dot_product:.5f}, "
                f"mean_abs={np.mean(np.abs(gradient)):.5f}, "
                f"max_abs={np.max(np.abs(gradient)):.5f}",
            )

            # 3c - Line search

            success = False
            sub_iteration = 0
            epsilon = (step_size_ratio * distance) / np.sqrt(iteration)

            while not success:
                sub_iteration += 1

                line_search_trial = Trial(
                    candidate=Image(clip(current_array + epsilon * gradient, 0, 1)), is_probe=True
                )
                yield line_search_trial
                await line_search_trial

                success = is_adversarial(line_search_trial)
                if success:
                    current_best = line_search_trial.candidate

                logger.info(
                    f"[{iteration}.{sub_iteration}] Line search: "
                    f"epsilon={epsilon:.5f}, "
                    f"is_adversarial={success}, "
                    f"trial={line_search_trial}"
                )

                epsilon /= 2.0

                # Safety break as we get into floating point noise
                if sub_iteration > 15:
                    logger.warning(
                        f"Line search failed to find an adversarial sample after {sub_iteration} attempts."
                    )
                    break

            # 3d - Projection

            distance_before_projection = (
                await image_distance(source, norm=distance_method)(current_best)
            ).value

            logger.info(f"[{iteration}] Projection: distance={distance_before_projection:.5f}")

            projector = bisection_image_search(
                source,
                current_best,
                decision_objective=decision_objective,
                decision_threshold=decision_threshold,
                tolerance=boundary_tolerance,
            )

            async for trial in projector(context):
                yield trial.as_probe()

            current_best = trial.candidate
            yield Trial(candidate=current_best)

    return Search(search, name="hop_skip_jump")
