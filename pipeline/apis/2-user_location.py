#!/usr/bin/python3
"""Fetch a GitHub user's location using the GitHub API."""

import sys
import requests
from datetime import datetime, timezone


if __name__ == "__main__":
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data.get("location") or "None")
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset = int(response.headers.get("X-Ratelimit-Reset", 0))
        now = datetime.now(timezone.utc).timestamp()
        minutes = max(0, int((reset - now) / 60))
        print(f"Reset in {minutes} min")
