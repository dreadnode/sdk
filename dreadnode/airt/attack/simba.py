import typing as t

from dreadnode.airt.attack.base import Attack
from dreadnode.airt.search.simba import simba_search
from dreadnode.data_types import Image
from dreadnode.optimization.stop import score_value
from dreadnode.scorers.base import Scorer, ScorerLike
from dreadnode.scorers.image import Norm

if t.TYPE_CHECKING:
    from dreadnode.airt.target.base import Target


def simba_attack(
    target: "Target[Image, t.Any]",
    original: Image,
    confidence: ScorerLike[Image],
    is_adversarial: ScorerLike[Image] | None = None,
    *,
    norm: Norm = "l2",
    theta: float = 0.1,
    num_masks: int = 1_000,
    seed: int | None = None,
    name: str = "simba_attack",
    description: str = "Simple Black-box adversarial image attack",
) -> Attack[Image, t.Any]:
    """
    Creates a SimBA (Simple Black-box Attack) for black-box image classifier settings.

    A series of random perturbations masks are created, and for every search
    iteration, a random mask is applied to the image. If the perturbation
    improves the adversarial objective, it is retained; otherwise, it is discarded.

    See: SimBA - https://arxiv.org/abs/1805.12317

    Args:
        target: The target model to attack.
        original: The original, non-adversarial image.
        confidence: A scorer that returns the confidence of the desired class.
            The attack will attempt to maximize this score.
        is_adversarial: An optional scorer that returns a positive value if the
            image is successfully adversarial. Used for early stopping.
        norm: The distance metric to use. Options are 'l2' (Euclidean
            distance), 'l1' (Manhattan distance), or 'linf' (Chebyshev distance).
        theta: The magnitude of each perturbation step.
        num_masks: The number of random noise masks to generate and use.
        seed: Optional random seed for reproducibility.
        name: The name of the attack instance.
    """
    confidence_scorer = Scorer.fit(confidence)
    is_adversarial_scorer = Scorer.fit(is_adversarial) if is_adversarial is not None else None

    search_strategy = simba_search(
        original,
        theta=theta,
        num_masks=num_masks,
        objective="confidence",
        norm=norm,
        seed=seed,
    )

    attack = Attack[Image, t.Any](
        name=name,
        description=description,
        target=target,
        search_strategy=search_strategy,
        objectives={
            "confidence": confidence_scorer,
        },
        directions=["maximize"],
        max_evals=num_masks,
    )

    if is_adversarial_scorer is not None:
        attack.add_objective(is_adversarial_scorer, name="is_adversarial", direction="maximize")
        attack.add_stop_condition(score_value("is_adversarial", gt=0.0))

    return attack
