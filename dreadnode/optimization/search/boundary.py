import inspect
import typing as t

from dreadnode.data_types import Image
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.scorers.image import DistanceMethod, image_distance
from dreadnode.transforms import Transform, TransformLike


def boundary_search(
    start_candidate: CandidateT,
    end_candidate: CandidateT,
    interpolate: TransformLike[tuple[CandidateT, CandidateT], CandidateT],
    tolerable: t.Callable[[CandidateT, CandidateT], t.Awaitable[bool]],
    *,
    decision_objective: str | None = None,
    decision_threshold: float = 0.0,
) -> Search[CandidateT]:
    """
    Performs a boundary search between two candidates to find a new candidate
    which lies on the decision boundary defined by the objective and threshold.

    Args:
        start_candidate: A candidate expected to be unsuccessful (score <= [decision_threshold]).
        end_candidate: A candidate expected to be successful (score > [decision_threshold]).
        interpolate: A transform that takes two candidates and returns a candidate
                     that is between them.
        tolerable: A function that checks if the similarity (distance) between two candidates is within acceptable limits.
        decision_objective: The name of the objective to use for the decision. If None, uses the overall trial score.
        decision_threshold: The threshold value for the decision objective.
    """

    async def search(context: OptimizationContext) -> t.AsyncGenerator[Trial[CandidateT], None]:
        if decision_objective and decision_objective not in context.objective_names:
            raise ValueError(
                f"Decision objective '{decision_objective}' not found in the optimization context."
            )

        def is_successful(trial: Trial) -> bool:
            score_to_check = (
                trial.scores.get(decision_objective, 0.0) if decision_objective else trial.score
            )
            return score_to_check > decision_threshold

        start_trial = Trial(candidate=start_candidate)
        end_trial = Trial(candidate=end_candidate)
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

        original_bound = start_candidate
        adversarial_bound = end_candidate
        interpolate_transform = Transform(interpolate)

        while not await tolerable(original_bound, adversarial_bound):
            midpoint_candidate = await interpolate_transform((original_bound, adversarial_bound))
            if inspect.isawaitable(midpoint_candidate):
                midpoint_candidate = await midpoint_candidate

            midpoint_trial = Trial(candidate=midpoint_candidate)
            yield midpoint_trial
            await midpoint_trial

            if is_successful(midpoint_trial):
                adversarial_bound = midpoint_trial.candidate
            else:
                original_bound = midpoint_trial.candidate

        yield Trial(candidate=adversarial_bound)

    return Search(search, name="boundary_search")


def binary_image_search(
    start_image: Image,
    end_image: Image,
    *,
    tolerance: float = 5.0,  # relatively high because of image pixel precision
    distance_method: DistanceMethod = "l2",
    decision_objective: str | None = None,
    decision_threshold: float = 0.0,
) -> Search[Image]:
    """
    Performs a binary search between two images to find a new image
    which lies on the decision boundary defined by the objective and threshold.

    Args:
        start_image: An image expected to be unsuccessful (score <= [decision_threshold]).
        end_image: An image expected to be successful (score > [decision_threshold]).
        tolerance: The maximum acceptable distance between the start and end images.
        distance_method: The distance metric to use for measuring similarity.
        decision_objective: The name of the objective to use for the decision. If None,
    """
    from dreadnode.transforms.image import interpolate

    async def tolerable(img1: Image, img2: Image) -> bool:
        metric = await image_distance(img1, method=distance_method)(img2)
        return metric.value < tolerance

    return boundary_search(
        start_candidate=start_image,
        end_candidate=end_image,
        interpolate=interpolate(alpha=0.5),
        tolerable=tolerable,
        decision_objective=decision_objective,
        decision_threshold=decision_threshold,
    )
