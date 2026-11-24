"""Reverse search: find the lineages that contain a given alert.

Usage: python reverse_search.py [<object_id>]

Without argument (e.g. when pasted in a REPL or notebook), a real alert is used.

The server address is read from the FINK_FAT_URL environment variable
(default http://localhost:8080). Requires `pip install requests`.
"""

import os
import sys

import requests

BASE_URL = os.environ.get("FINK_FAT_URL", "http://localhost:8080")


def lineages_of_alert(object_id: str) -> list[dict] | None:
    """Return the lineages containing the alert, or None if the alert is unknown."""
    response = requests.get(
        f"{BASE_URL}/api/v1/alerts/{object_id}/lineages", timeout=30
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()["lineages"]


if __name__ == "__main__":
    object_id = sys.argv[1]
    lineages = lineages_of_alert(object_id)
    if lineages is None:
        print(f"{object_id}: unknown alert")
    elif not lineages:
        print(f"{object_id}: known alert, but not part of any lineage")
    for lineage in lineages or []:
        print(
            f"{lineage['lineage_designation']} (id {lineage['lineage_id']}): "
            f"best branch {lineage['best_branch_id']}, "
            f"matching branches {lineage['matching_branch_ids']} -> "
            f"{BASE_URL}{lineage['url']}"
        )
