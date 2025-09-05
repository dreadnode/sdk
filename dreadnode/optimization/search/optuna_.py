import typing as t

import optuna

from dreadnode.optimization.search.base import Categorical, Float, Int, Search, SearchSpace
from dreadnode.optimization.trial import Trial
from dreadnode.types import AnyDict

if t.TYPE_CHECKING:
    from uuid import UUID


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
        self, search_space: SearchSpace, *, study: optuna.study.Study | None = None
    ) -> None:
        self.optuna_study = study or optuna.create_study()
        self.optuna_search_space = _convert_search_space(search_space)
        self._trial_map: dict[UUID, optuna.trial.Trial] = {}

    async def suggest(self) -> list[Trial[AnyDict]]:
        optuna_trial = self.optuna_study.ask(self.optuna_search_space)
        candidate_params = optuna_trial.params

        trial = Trial[AnyDict](
            candidate=candidate_params,
        )
        self._trial_map[trial.id] = optuna_trial

        return [trial]

    def observe(self, trials: list[Trial[AnyDict]]) -> None:
        for trial in trials:
            optuna_trial = self._trial_map[trial.id]
            if trial.status == "success":
                self.optuna_study.tell(optuna_trial, trial.score)
            else:
                self.optuna_study.tell(
                    optuna_trial,
                    state=optuna.trial.TrialState.PRUNED
                    if trial.status == "pruned"
                    else optuna.trial.TrialState.FAIL,
                )
