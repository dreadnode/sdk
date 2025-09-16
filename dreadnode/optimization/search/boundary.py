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

    Args:
        start: A candidate on the left side of the decision boundary (`score <= [decision_threshold]`).
        end: A candidate on the right side of the decision boundary (`score > [decision_threshold]`).
        interpolate: A transform that takes two candidates and an alpha value between and returns a candidate
                     that is between them.
        tolerance: The maximum acceptable difference between the upper and lower alpha values.
        decision_objective: The name of the objective to use for the decision. If None, uses the overall trial score.
        decision_threshold: The threshold value for the decision objective.
    """

    async def search(context: OptimizationContext) -> t.AsyncGenerator[Trial[CandidateT], None]:
        def is_successful(trial: Trial) -> bool:
            return trial.get_directional_score(decision_objective) > decision_threshold

        logger.info("Starting boundary search")

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
                f"start_candidate was considered successful ({decision_objective or 'score'} > {decision_threshold}): {start_trial.scores}."
            )

        if not is_successful(end_trial):
            raise ValueError(
                f"end_candidate was not considered successful ({decision_objective or 'score'} <= {decision_threshold}): {end_trial.scores}."
            )

        # TODO(nick): When tolerance is met immediately, it can be confusing that
        # the attack just returns as search_exhausted. Maybe we add some kind of log
        # reason to be put in the Trial?

        lower_bound_alpha = 0.0
        upper_bound_alpha = 1.0
        interpolate_transform = Transform(interpolate)

        adversarial_candidate = end

        while (upper_bound_alpha - lower_bound_alpha) > tolerance:
            midpoint_alpha = (lower_bound_alpha + upper_bound_alpha) / 2.0
            midpoint_candidate = await interpolate_transform((start, end, midpoint_alpha))

            logger.info(
                f"Boundary search iteration: lower={lower_bound_alpha:.4f}, upper={upper_bound_alpha:.4f}, midpoint={midpoint_alpha:.4f}"
            )

            midpoint_trial = Trial(candidate=midpoint_candidate)
            yield midpoint_trial
            await midpoint_trial

            if is_successful(midpoint_trial):
                upper_bound_alpha = midpoint_alpha
                adversarial_candidate = midpoint_trial.candidate
            else:
                lower_bound_alpha = midpoint_alpha

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
