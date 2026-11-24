"""Batch reverse search: find the lineages of several alerts in one request.

Usage: python reverse_search_batch.py [<object_id> ...]

Without arguments (e.g. when pasted in a REPL or notebook), two real alerts
are used.

The server address is read from the FINK_FAT_URL environment variable
(default http://localhost:8080). Requires `pip install requests`.
"""

import os
import sys

import requests

BASE_URL = os.environ.get("FINK_FAT_URL", "http://localhost:8080")

def lineages_of_alerts(object_ids: list[str]) -> dict:
    """Return the batch response: `results` (known alerts) and `unknown_object_ids`."""
    response = requests.post(
        f"{BASE_URL}/api/v1/alerts/lineages",
        json={"object_ids": object_ids},
        timeout=60,
    )
    if response.status_code == 400:
        raise ValueError(response.json()["message"])
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    body = lineages_of_alerts(sys.argv[1:])
    for result in body["results"]:
        if not result["lineages"]:
            print(f"{result['object_id']}: known alert, but not part of any lineage")
        for lineage in result["lineages"]:
            print(
                f"{result['object_id']} -> {lineage['lineage_designation']} "
                f"(id {lineage['lineage_id']}): best branch {lineage['best_branch_id']}, "
                f"matching branches {lineage['matching_branch_ids']} -> "
                f"{BASE_URL}{lineage['url']}"
            )
    for object_id in body["unknown_object_ids"]:
        print(f"{object_id}: unknown alert")
