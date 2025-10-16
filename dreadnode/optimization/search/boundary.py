import typing as t

from loguru import logger

from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.transforms import Transform, TransformLike


def boundary_search(
    start: CandidateT,
    end: CandidateT,
    interpolate: TransformLike[tuple[CandidateT, CandidateT, float], CandidateT],
    *,
    tolerance: float = 1e-2,
    decision_objective: str | None = None,
    decision_threshold: float = 0.0,
    name: str = "boundary",
) -> Search[CandidateT]:
    """
    Performs a boundary search between two candidates to find a new candidate
    which lies on the decision boundary defined by the objective and threshold.

    This requires an `interpolate` transform that can produce a new candidate
    given two candidates and an alpha value between 0 and 1. Typically, this
    would apply to continuous candidate spaces such as images or embeddings.

    Args:
        start: A candidate on the left side of the decision boundary (`score <= [decision_threshold]`).
        end: A candidate on the right side of the decision boundary (`score > [decision_threshold]`).
        interpolate: A transform that takes two candidates and an alpha value between 0 and 1 and returns a candidate
                     that is constructed by interpolating between the two inputs.
        tolerance: The maximum acceptable difference between the upper and lower alpha values.
        decision_objective: The name of the objective to use for the decision. If None, uses the overall trial score.
        decision_threshold: The threshold value for the decision objective.
    """

    async def search(
        context: OptimizationContext,
    ) -> t.AsyncGenerator[Trial[CandidateT], None]:
        def is_successful(trial: Trial) -> bool:
            return trial.get_directional_score(decision_objective) > decision_threshold

        logger.info(
            f"Starting boundary search: "
            f"tolerance={tolerance:.5f}, "
            f"decision_objective='{decision_objective}', "
            f"decision_threshold={decision_threshold:.5f}"
        )

        if decision_objective and decision_objective not in context.objective_names:
            raise ValueError(
                f"Decision objective '{decision_objective}' not found in the optimization context."
            )

        start_trial = Trial(candidate=start)
        end_trial = Trial(candidate=end)

        yield start_trial
        yield end_trial

        await Trial.wait_for(start_trial, end_trial)

        if is_successful(start_trial):
            raise ValueError(
                f"start_candidate met the decision criteria ({decision_objective or 'score'} > {decision_threshold}): {start_trial.scores}."
            )

        if not is_successful(end_trial):
            raise ValueError(
                f"end_candidate did not meet the decision criteria ({decision_objective or 'score'} <= {decision_threshold}): {end_trial.scores}."
            )

        lower_bound_alpha = 0.0
        upper_bound_alpha = 1.0
        interpolate_transform = Transform(interpolate)

        adversarial_candidate = end
        iteration = 0

        while (upper_bound_alpha - lower_bound_alpha) > tolerance:
            iteration += 1
            midpoint_alpha = (lower_bound_alpha + upper_bound_alpha) / 2.0

            logger.info(
                f"[{iteration}] Interpolate: "
                f"lower={lower_bound_alpha:.5f}, "
                f"upper={upper_bound_alpha:.5f}, "
                f"midpoint={midpoint_alpha:.5f}"
            )

            candidate = await interpolate_transform((start, end, midpoint_alpha))
            trial = Trial(candidate=candidate)
            yield trial
            await trial

            if is_successful(trial):
                upper_bound_alpha = midpoint_alpha
                adversarial_candidate = trial.candidate
            else:
                lower_bound_alpha = midpoint_alpha

        logger.info(
            f"Boundary found within {tolerance:.5f} after {iteration} iterations: alpha={upper_bound_alpha:.5f}"
        )

        yield Trial(candidate=adversarial_candidate)

    return Search(search, name=name)


def bisection_image_search(
    start: Image,
    end: Image,
    *,
    tolerance: float = 1e-2,
    decision_objective: str | None = None,
    decision_threshold: float = 0.0,
) -> Search[Image]:
    """
    Performs a binary search between two images to find a new image
    which lies on the decision boundary defined by the objective and threshold.

    Args:
        start: An image on the left side of the decision boundary (`score <= [decision_threshold]`).
        end: An image on the right side of the decision boundary (`score > [decision_threshold]`).
        tolerance: The maximum acceptable difference between the upper and lower alpha values.
        decision_objective: The name of the objective to use for the decision. If None, uses the overall trial score.
        decision_threshold: The threshold value for the decision objective.
    """
    from dreadnode.transforms.image import interpolate_images

    async def interpolate(args: tuple[Image, Image, float]) -> Image:
        imgs, alpha = args[:2], args[2]
        return await interpolate_images(alpha)(imgs)

    return boundary_search(
        start=start,
        end=end,
        interpolate=interpolate,
        tolerance=tolerance,
        decision_objective=decision_objective,
        decision_threshold=decision_threshold,
        name="bisection_image",
    )
