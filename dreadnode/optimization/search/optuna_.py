import typing as t

import optuna
from loguru import logger

from dreadnode.common_types import AnyDict
from dreadnode.optimization.search.base import (
    Categorical,
    Float,
    Int,
    OptimizationContext,
    Search,
    SearchSpace,
)
from dreadnode.optimization.trial import Trial


def _convert_search_space(
    search_space: SearchSpace,
) -> dict[str, optuna.distributions.BaseDistribution]:
    optuna_space: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, dist in search_space.items():
        if isinstance(dist, Float):
            optuna_space[name] = optuna.distributions.FloatDistribution(
                low=dist.low, high=dist.high, log=dist.log, step=dist.step
            )
        elif isinstance(dist, Int):
            optuna_space[name] = optuna.distributions.IntDistribution(
                low=dist.low, high=dist.high, log=dist.log, step=dist.step
            )
        elif isinstance(dist, Categorical):
            optuna_space[name] = optuna.distributions.CategoricalDistribution(choices=dist.choices)
        elif isinstance(dist, list):
            optuna_space[name] = optuna.distributions.CategoricalDistribution(choices=dist)
        else:
            raise TypeError(f"Unsupported distribution type: {type(dist)}")
    return optuna_space


def optuna_search(
    search_space: SearchSpace,
    *,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> Search[AnyDict]:
    """
    Creates a search strategy that uses Optuna for Bayesian optimization.

    This strategy leverages Optuna's powerful samplers (like TPE) to intelligently
    explore a defined search space, learning from past trial results to suggest
    more promising candidates.

    Args:
        search_space: The search space to explore, defining parameter names and distributions.
        sampler: An optional Optuna sampler (e.g., TPESampler, NSGAIISampler).
    """

    async def search(
        context: OptimizationContext,
        *,
        search_space: SearchSpace = search_space,
        sampler: optuna.samplers.BaseSampler | None = sampler,
    ) -> t.AsyncGenerator[Trial[AnyDict], None]:
        optuna_study = optuna.create_study(directions=context.directions, sampler=sampler)
        optuna_search_space = _convert_search_space(search_space)
        objective_names = context.objective_names

        logger.info(
            "Starting Optuna search: "
            f"sampler={optuna_study.sampler.__class__.__name__}, "
            f"objectives={objective_names}, "
            f"search_space={search_space}"
        )

        while True:
            optuna_trial = optuna_study.ask(optuna_search_space)

            trial = Trial[AnyDict](candidate=optuna_trial.params)
            yield trial
            await trial

            if trial.status == "finished":
                # Provide scores in the correct order for multi-objective optimization.
                scores = [trial.scores.get(name, 0.0) for name in objective_names]
                optuna_study.tell(optuna_trial, scores)
            else:
                state = (
                    optuna.trial.TrialState.PRUNED
                    if trial.status == "pruned"
                    else optuna.trial.TrialState.FAIL
                )
                optuna_study.tell(optuna_trial, state=state)

    return Search(search, name="optuna")
