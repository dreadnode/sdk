import rigging as rg
import typing as t

from dreadnode.metric import Metric
from dreadnode.scorers import Scorer

from loguru import logger


def data_obj_equivalence(expected: t.Any) -> "Scorer[t.Any]":
    """ """

    async def _evaluate(
        candidate: t.Any,
        expected: t.Any = expected,
        _recursion_depth: int = 0
    ) -> bool:
        score, total = 0.0, 0.0
        #logger.debug(f"Rec: {_recursion_depth} \n Exp: {expected} \n Cand: {candidate}")

        if type(expected) != type(candidate):
            score += 0.0
            total += 1.0

        elif type(expected) in [str, int, float, None]:
            score += 1.0 if expected == candidate else 0.0
            total += 1.0

        elif type(expected) is list:    
            try:
                if len(expected) == 0 and expected == candidate:
                    pass
                elif isinstance(expected[0], rg.Model):
                    expected = [dict(e) for e in expected]
                    candidate = [dict(c) for c in candidate]
                elif isinstance(expected[0], dict):
                    a_key = list(expected[0].keys())[0]
                    expected = sorted(expected, key=lambda item: item[a_key])
                    candidate = sorted(candidate, key=lambda item: item[a_key])
                else:
                    expected, candidate = sorted(expected), sorted(candidate)
            except Exception as e:
                logger.warning(
                    f"Tried to sort list types for object equivalency comparision, error occured: {e}"
                )

            for i, j in zip(expected, candidate):
                score += await _evaluate(
                    expected=i, candidate=j, _recursion_depth=(_recursion_depth + 1)
                )
                total += 1.0
            total += max(len(expected) - len(candidate), 0)

        elif isinstance(expected, (dict, rg.Model)):
            o1, o2 = dict(expected), dict(candidate)
            for k, v in o1.items():
                if v is None:
                    continue
                if k in o2:
                    score += await _evaluate(
                        expected=v,
                        candidate=o2[k],
                        _recursion_depth=(_recursion_depth + 1),
                    )
                total += 1.0

        if total == 0.0:
            total = total + (1 / 10**6)

        return Metric(value=float(score) / float(total), attributes={})

    # async def _wrapper(chat: rg.Chat) -> rg.Chat:
    #     chat.metadata["metrics"]["episode_reward"] = await _evaluate(
    #         expected=expected, candidate=chat.metadata["output"], _recursion_depth=0
    #     )
    #     return chat

    return Scorer(_evaluate, name="data_obj_equivalence")
