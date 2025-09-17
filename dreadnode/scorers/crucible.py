import re
import typing as t

import aiohttp

from dreadnode.metric import Metric
from dreadnode.scorers import Scorer


async def _check_flag(
    flag: str, challenge_id: str, api_key: str, platform_url: str = "https://platform.dreadnode.io"
) -> bool:
    async with aiohttp.ClientSession() as session:
        url = f"{platform_url}/api/challenges/{challenge_id}/submit-flag"
        headers = {"X-API-Key": api_key}
        payload = {"challenge": challenge_id, "flag": flag}

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return bool(data.get("correct", False))

    return False


def contains_crucible_flag(
    challenge: str,
    api_key: str,
    *,
    platform_url: str = "https://platform.dreadnode.io",
    score_for_partial_flag: float = 0.5,
    name: str = "contains_crucible_flag",
) -> Scorer[t.Any]:
    from dreadnode import tag

    async def evaluate(
        obj: t.Any,
        *,
        challenge: str = challenge,
        api_key: str = api_key,
        platform_url: str = platform_url,
        score_for_partial_flag: float = score_for_partial_flag,
    ) -> Metric:
        text = str(obj)
        metric = Metric(value=0.0, attributes={"challenge": challenge})

        matches = re.findall(r"gAAAAA[a-zA-Z0-9\-_=]+", text)
        if not matches:
            return metric

        metric.value = score_for_partial_flag  # Partial credit for flag-like values
        metric.attributes["matches"] = matches

        for match in matches:
            if await _check_flag(match, challenge, api_key, platform_url):
                metric.value = 1.0
                metric.attributes["flag"] = match
                tag("found_flag")

        return metric

    return Scorer(evaluate, name=name)
