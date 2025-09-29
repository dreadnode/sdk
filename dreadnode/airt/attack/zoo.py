import typing as t

from dreadnode.airt.attack.base import Attack
from dreadnode.airt.search.zoo import zoo_search
from dreadnode.data_types import Image
from dreadnode.optimization.stop import score_value
from dreadnode.scorers.base import Scorer, ScorerLike

if t.TYPE_CHECKING:
    from dreadnode.airt.target.base import Target


def zoo_attack(
    target: "Target[Image, t.Any]",
    original: Image,
    confidence: ScorerLike[Image],
    *,
    is_adversarial: ScorerLike[Image] | None = None,
    max_iterations: int = 1_000,
    learning_rate: float = 0.01,
    num_samples: int = 128,
    epsilon: float = 0.01,
    scaling_schedule: list[tuple[int, float]] | None = None,
    importance_sampling_freq: int = 10,
    seed: int | None = None,
    name: str = "zoo_attack",
    description: str = "Zeroth-Order adversarial image attack",
) -> Attack[Image, t.Any]:
    """
    Creates a Zeroth-Order Optimization (ZOO) attack for black-box image classifiers.

    See: ZOO - https://arxiv.org/abs/1708.03999

    Args:
        target: The target model to attack.
        original: The original, non-adversarial image.
        confidence: A scorer that returns the confidence of the desired class.
        is_adversarial: An optional scorer for early stopping.
        max_iterations: The maximum number of optimization steps to perform.
        learning_rate: The step size (eta) for the ADAM optimizer.
        num_samples: The number of coordinates to sample at each iteration.
        epsilon: The small value (h) for finite difference gradient estimation.
        scaling_schedule: A list of tuples `(iterations, ratio)` to define a hierarchical attack.
            - `iterations`: The number of iterations to run for this stage.
            - `ratio`: A float (0.0 to 1.0) for the attack space size relative
              to the original image's total pixels.
        importance_sampling_freq: Frequency (in iterations) for updating importance
                                  sampling probabilities. Set to 0 to disable.
        seed: Optional random seed for reproducibility.
        name: The name of the attack instance.
    """
    confidence_scorer = Scorer.fit(confidence)
    is_adversarial_scorer = Scorer.fit(is_adversarial) if is_adversarial is not None else None

    search_strategy = zoo_search(
        original,
        objective="confidence",
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        num_samples=num_samples,
        epsilon=epsilon,
        scaling_schedule=scaling_schedule,
        importance_sampling_freq=importance_sampling_freq,
        seed=seed,
    )

    # Total trials = (2 probes per sample + 1 main update) for each iteration
    total_trials = max_iterations * (1 + 2 * num_samples)

    attack = Attack[Image, t.Any](
        name=name,
        description=description,
        target=target,
        search_strategy=search_strategy,
        objectives={"confidence": confidence_scorer},
        directions=["maximize"],
        max_evals=total_trials,
    )

    if is_adversarial_scorer is not None:
        attack.add_objective(is_adversarial_scorer, name="is_adversarial", direction="maximize")
        attack.add_stop_condition(score_value("is_adversarial", gt=0.0))

    return attack
