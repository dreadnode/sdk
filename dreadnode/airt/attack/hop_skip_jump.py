import typing as t

from dreadnode.airt.attack.base import Attack
from dreadnode.airt.search.hop_skip_jump import hop_skip_jump_search
from dreadnode.data_types import Image
from dreadnode.meta import TaskInput
from dreadnode.optimization.stop import score_value
from dreadnode.scorers.base import Scorer, ScorerLike
from dreadnode.scorers.image import Norm, image_distance

if t.TYPE_CHECKING:
    from dreadnode.airt.target.base import Target


def hop_skip_jump_attack(
    target: "Target[Image, t.Any]",
    original: Image,
    is_adversarial: ScorerLike[Image],
    adversarial: Image | None = None,
    *,
    early_stopping_distance: float | None = None,
    norm: Norm = "l2",
    theta: float = 0.01,
    boundary_tolerance: float | None = None,
    step_size: float | None = None,
    min_evaluations: int = 50,
    max_evaluations: int = 100,
    max_iterations: int = 1_000,
    name: str = "hop_skip_jump_attack",
    description: str = "HopSkipJump adversarial image attack",
) -> Attack[Image, t.Any]:
    """
    Creates a HopSkipJump attack for black-box image classifier settings.

    See: HopSkipJumpAttack - https://arxiv.org/abs/1904.02144

    Args:
        target: The target model to attack.
        original: The original, unperturbed image.
        adversarial: An initial adversarial example. If not provided, a random search will be performed
            to find one that satisfies the adversarial objective and threshold.
        is_adversarial: The name of the objective to use for the adversarial decision.
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
    distance_scorer = image_distance(original, norm=norm, normalize=True).bind(TaskInput())
    is_adversarial_scorer = Scorer.fit(is_adversarial)

    search_strategy = hop_skip_jump_search(
        original,
        adversarial,
        adversarial_objective="is_adversarial",
        theta=theta,
        norm=norm,
        step_size=step_size,
        boundary_tolerance=boundary_tolerance,
        min_evaluations=min_evaluations,
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
    )

    attack = Attack[Image, t.Any](
        name=name,
        description=description,
        target=target,
        search_strategy=search_strategy,
        objectives={
            "is_adversarial": is_adversarial_scorer,
            "distance": distance_scorer,
        },
        directions=["maximize", "minimize"],
        max_evals=max_iterations * max_evaluations,
    )

    if early_stopping_distance is not None:
        attack = attack.add_stop_condition(
            score_value("distance", lte=early_stopping_distance)
            & score_value("is_adversarial", gt=0.0)
        )

    return attack
