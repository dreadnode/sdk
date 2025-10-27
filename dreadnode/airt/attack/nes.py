import typing as t

from dreadnode.airt.attack.base import Attack
from dreadnode.airt.search.nes import nes_search
from dreadnode.data_types import Image
from dreadnode.optimization.stop import score_value
from dreadnode.scorers.base import Scorer, ScorerLike

if t.TYPE_CHECKING:
    from dreadnode.airt.target.base import Target


def nes_attack(
    target: "Target[Image, t.Any]",
    original: Image,
    confidence: ScorerLike[Image],
    is_adversarial: ScorerLike[Image] | None = None,
    *,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
    num_samples: int = 64,
    sigma: float = 0.001,
    seed: int | None = None,
    name: str = "nes_attack",
    description: str = "Natural Evolution Strategies adversarial image attack",
) -> Attack[Image, t.Any]:
    """
    Creates a Natural Evolution Strategies (NES) attack for black-box image classifiers.

    This attack uses NES to estimate the full gradient of the confidence score,
    by probing the model with random perturbations in the positive and negative
    directions. It uses the Adam optimizer to adaptively update the image
    based on the gradient estimation.

    Args:
        target: The target model to attack.
        original: The original, non-adversarial image.
        confidence: A scorer that returns the confidence of the desired class.
                    The attack will attempt to maximize this score.
        is_adversarial: An optional scorer that returns a positive value if the
                        image is successfully adversarial. Used for early stopping.
        max_iterations: The number of main optimization steps to perform.
        learning_rate: The step size for updating the image based on the
                       estimated gradient.
        num_samples: The number of random directions to probe at each iteration.
                     Total queries per iteration will be (2 * num_samples) + 1.
        sigma: The exploration variance (magnitude of the random perturbations).
        seed: Optional random seed for reproducibility.
        name: The name of the attack instance.

    Returns:
        A configured Attack object ready to be run.
    """
    confidence_scorer = Scorer.fit(confidence)
    is_adversarial_scorer = Scorer.fit(is_adversarial) if is_adversarial is not None else None

    search_strategy = nes_search(
        original,
        objective="confidence",
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        num_samples=num_samples,
        sigma=sigma,
        seed=seed,
    )

    # The total number of trials is (2 probes per sample + 1 main update) for each iteration
    total_trials = max_iterations * (1 + 2 * num_samples)

    attack = Attack[Image, t.Any](
        name=name,
        description=description,
        target=target,
        search_strategy=search_strategy,
        objectives={
            "confidence": confidence_scorer,
        },
        directions=["maximize"],
        max_evals=total_trials,
    )

    if is_adversarial_scorer is not None:
        attack.add_objective(is_adversarial_scorer, name="is_adversarial", direction="maximize")
        attack.add_stop_condition(score_value("is_adversarial", gt=0.0))

    return attack
