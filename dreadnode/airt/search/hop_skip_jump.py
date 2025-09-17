import typing as t

import numpy as np
from loguru import logger

from dreadnode.airt.search.image_utils import get_random, normalize_for_shape
from dreadnode.data_types import Image
from dreadnode.optimization.search import bisection_image_search, random_image_search
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.image import DistanceMethod, image_distance


def hop_skip_jump_search(  # noqa: PLR0915
    source: Image,
    target: Image | None = None,
    *,
    decision_objective: str | None = None,
    decision_threshold: float = 0.0,
    distance_method: DistanceMethod = "l2",
    theta: float = 0.01,
    min_eval: int = 40,
    max_iters: int = 1_000,
) -> Search[Image]:
    """
    Implements the HopSkipJump attack for decision-based black-box settings.
    """

    async def search(  # noqa: PLR0912, PLR0915
        context: OptimizationContext,
        *,
        source: Image = source,
        target: Image | None = target,
        decision_objective: str | None = decision_objective,
        decision_threshold: float = decision_threshold,
        distance_method: DistanceMethod = distance_method,
        theta: float = theta,
        min_evaluations: int = min_eval,
        max_iterations: int = max_iters,
    ) -> t.AsyncGenerator[Trial[Image], None]:
        def is_adversarial(trial: Trial) -> bool:
            return trial.get_directional_score(decision_objective) > decision_threshold

        logger.info("Starting HopSkipJump search")

        # 1 - Bootstrap (if needed)

        if target is None:
            logger.info("No target provided, searching for an initial adversarial example.")
            random_search = random_image_search(shape=source.shape)
            async for trial in random_search(context):
                yield trial
                if is_adversarial(await trial):
                    target = trial.candidate
                    break

            if target is None:
                raise RuntimeError("Failed to find an initial adversarial example.")

        # 2 - Boundary search

        logger.info("Performing initial boundary search.")

        current_trial: Trial[Image] | None = None
        async for trial in bisection_image_search(
            source,
            target,
            decision_objective=decision_objective,
            decision_threshold=decision_threshold,
            tolerance=theta,
        )(context):
            yield trial
            current_trial = await trial

        if not current_trial or not is_adversarial(current_trial):
            raise RuntimeError("Failed to perform initial boundary search.")

        # 3 - Main loop

        theta = normalize_for_shape(theta, source.shape, distance_method)

        for iteration in range(1, max_iterations + 1):
            current = current_trial.candidate

            # 3a - Gradient estimation

            current_distance = (
                await image_distance(source, method=distance_method, normalize=False)(current)
            ).value

            image_size = np.prod(current.shape)
            if distance_method == "l2":
                delta = np.sqrt(image_size) * current_distance * theta
            else:
                delta = image_size * current_distance * theta

            # Special case from original
            if iteration == 1:
                delta = 1

            # override
            delta = 0.005

            num_evals = min(int(min_evaluations * np.sqrt(iteration)), max_iterations)
            noise_shape = (num_evals, *current.shape)
            random_noise = get_random(noise_shape, distance_method)
            noise_norms = np.linalg.norm(random_noise.reshape(num_evals, -1), axis=1).reshape(
                num_evals, *((1,) * (len(current.shape)))
            )
            random_noise /= noise_norms

            current_array = current.to_numpy()
            perturbation_arrays = np.clip(current_array + delta * random_noise, 0, 1)
            perturbations = (perturbation_arrays - current_array) / delta

            logger.info(
                f"[{iteration}] Estimating gradient with {num_evals} evaluations (delta={delta:.4f}, theta={theta:.4f}, current_distance={current_distance:.4f})."
            )

            perturbed_trials = [Trial(candidate=Image(p)) for p in perturbation_arrays]
            for trial in perturbed_trials:
                yield trial
            await Trial.wait_for(*perturbed_trials)

            satisfied = np.array(
                [is_adversarial(trial) for trial in perturbed_trials], dtype=np.float32
            )
            f_val = 2 * satisfied - 1
            if np.mean(f_val) in [1.0, -1.0]:
                gradient = np.mean(perturbations, axis=0) * (np.mean(f_val))
            else:
                f_val -= f_val.mean()
                f_val_reshaped = f_val.reshape(num_evals, *((1,) * len(current_array.shape)))
                gradient = np.mean(f_val_reshaped * perturbations, axis=0)

            if distance_method == "l2":
                gradient /= np.linalg.norm(gradient)
            else:
                gradient = np.sign(gradient)

            logger.info(
                f"[{iteration}] Estimated gradient norm {np.linalg.norm(gradient):.4f} ({satisfied.sum()} adversarial / {num_evals} total)."
            )

            # 3c - Line search

            # epsilon = current_distance / np.sqrt(iteration)
            epsilon = 2.0 * current_distance / np.sqrt(iteration)

            logger.info(f"[{iteration}] Performing line search with initial epsilon {epsilon:.4f}.")

            success = False
            while not success:
                potential = Image(np.clip(current_array + epsilon * gradient, 0, 1))
                potential_trial = Trial(candidate=potential)
                yield potential_trial
                await potential_trial

                if is_adversarial(potential_trial):
                    current = potential
                    success = True
                    break

                epsilon /= 2.0
                logger.info(f"[{iteration}] Trying epsilon {epsilon:.4f}.")

            logger.info(f"[{iteration}] Found adversarial example at epsilon {epsilon:.4f}.")

            # 3d - Projection

            projector = bisection_image_search(
                source,
                current,
                decision_objective=decision_objective,
                decision_threshold=decision_threshold,
                tolerance=theta,
            )

            logger.info(
                f"[{iteration}] Projecting back to boundary (current_distance={current_distance:.4f})."
            )

            async for trial in projector(context):
                yield trial
                current_trial = await trial

            new_distance = (
                await image_distance(source, method=distance_method, normalize=False)(current)
            ).value
            logger.info(f"[{iteration}] Projection complete (new_distance={new_distance:.4f}).")

            if not is_adversarial(current_trial):
                raise RuntimeError("Projection step failed to find an adversarial example.")

    return Search(search, name="hop_skip_jump")
