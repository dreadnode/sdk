import re

import httpx

from dreadnode.agent.tools.base import tool

ARCHIVE_SUCCESS_CODE = 200


@tool
def search_archive(domain: str) -> list:
    response = httpx.get(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": domain,
            "matchType": "domain",
            "fl": "timestamp,original",
            "collapse": "digest",
            "output": "json",
        },
    )

    snapshots = response.json()[1:]

    for snap in snapshots:
        if re.search(r"/robots\.txt$", snap[1]):
            timestamp = snap[0]
            url = snap[1]
            archive_url = f"https://web.archive.org/web/{timestamp}if_/{url}"

            response = httpx.get(archive_url)
            if response.status_code == ARCHIVE_SUCCESS_CODE:
                return {
                    "timestamp": timestamp,
                    "url": url,
                    "archive_url": archive_url,
                    "content": response.text,
                }
            return {
                "timestamp": timestamp,
                "url": url,
                "archive_url": archive_url,
                "error": f"Failed to retrieve content: {response.status_code}",
            }

    return None
