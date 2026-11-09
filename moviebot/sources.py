"""Clients for the two live sources: DoesTheDogDie votes and TMDB movie details.

Both raise OSError (urllib.error.URLError, timeouts) when the service can't be
reached, and ValueError on a bad response; callers decide whether that is fatal.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

# DTDD sits behind Cloudflare, which rejects Python's default User-Agent (error 1010).
_USER_AGENT = "moviebot/0.1 (student project)"


def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 15) -> dict:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT, **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class DtddClient:
    base_url = "https://www.doesthedogdie.com"

    def __init__(self, api_key: str):
        self._headers = {"X-API-KEY": api_key}

    def _get(self, path: str) -> dict:
        # DTDD allows 30 requests a minute and a movie costs 2, so one search can reach the limit.
        # Waiting beats failing: a movie that wasn't checked can't be filtered out.
        for retries_left in (2, 1, 0):
            try:
                return _get_json(f"{self.base_url}{path}", self._headers)
            except urllib.error.HTTPError as e:
                if e.code != 429 or not retries_left:
                    raise
                wait = float(e.headers.get("Retry-After") or 60)
                if wait > 60:  # past the minute window: the monthly quota is gone
                    raise
            time.sleep(wait)

    def media(self, imdb_id: str) -> dict | None:
        """DTDD's page for one movie, or None if it has none. About 1.5 MB: every topic with its votes."""
        found = self._get(f"/dddsearch?imdb={urllib.parse.quote(imdb_id)}")
        items = [item for item in found.get("items", []) if item.get("imdbId") == imdb_id]
        if not items:
            return None
        return self._get(f"/media/{items[0]['id']}")

    def votes(self, imdb_id: str) -> dict[int, tuple[int, int]]:
        """(yes, no) votes per DTDD topic id; empty if DTDD has no page for the movie."""
        media = self.media(imdb_id)
        if media is None:
            return {}
        return {s["TopicId"]: (s["yesSum"] or 0, s["noSum"] or 0) for s in media["topicItemStats"]}


class TmdbClient:
    base_url = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str):
        self._api_key = api_key  # a v3 key; it goes in the query string

    def _get(self, path: str, **params) -> dict:
        return _get_json(f"{self.base_url}{path}?{urllib.parse.urlencode({'api_key': self._api_key, **params})}")

    def find_by_imdb(self, imdb_id: str) -> dict | None:
        """TMDB's summary of a movie (id, title, overview, release_date, ...), or None."""
        results = self._get(f"/find/{urllib.parse.quote(imdb_id)}", external_source="imdb_id")["movie_results"]
        return results[0] if results else None

    def search(self, title: str) -> list[dict]:
        return self._get("/search/movie", query=title)["results"]

    def details(self, tmdb_id: int) -> dict:
        """Full details, including imdb_id and genre names."""
        return self._get(f"/movie/{int(tmdb_id)}")
