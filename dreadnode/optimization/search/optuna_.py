import typing as t

import optuna

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

if t.TYPE_CHECKING:
    from ulid import ULID


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


class OptunaSearch(Search[AnyDict]):
    """An adapter that uses an Optuna study as a search strategy."""

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        sampler: optuna.samplers.BaseSampler | None = None,
        trials_per_step: int = 1,
    ) -> None:
        """
        Initializes the OptunaSearch with the given search space and study.

        Args:
            search_space: The search space to explore.
            sampler: An optional Optuna sampler (e.g., NSGAIISampler for MOO).
            trials_per_step: The number of trials to suggest at each step.
        """
        self.trials_per_step = trials_per_step
        self._optuna_sampler = sampler
        self._optuna_study = optuna.create_study()
        self._optuna_search_space = _convert_search_space(search_space)
        self._trial_map: dict[ULID, optuna.trial.Trial] = {}
        self._objective_names: list[str] = []

    def reset(self, context: OptimizationContext) -> None:
        self._optuna_study = optuna.create_study(
            directions=context.directions,
            sampler=self._optuna_sampler,
        )
        self._objective_names = context.objective_names
        self._trial_map = {}

    async def suggest(self, step: int) -> t.AsyncIterator[Trial[AnyDict]]:  # noqa: ARG002
        for _ in range(self.trials_per_step):
            optuna_trial = self._optuna_study.ask(self._optuna_search_space)
            trial = Trial[AnyDict](
                candidate=optuna_trial.params,
            )
            self._trial_map[trial.id] = optuna_trial
            yield trial

    async def observe(self, trials: list[Trial[AnyDict]]) -> None:
        for trial in trials:
            optuna_trial = self._trial_map[trial.id]
            if trial.status == "finished":
                self._optuna_study.tell(
                    optuna_trial, [trial.scores.get(name, 0.0) for name in self._objective_names]
                )
            else:
                self._optuna_study.tell(
                    optuna_trial,
                    state=optuna.trial.TrialState.PRUNED
                    if trial.status == "pruned"
                    else optuna.trial.TrialState.FAIL,
                )
