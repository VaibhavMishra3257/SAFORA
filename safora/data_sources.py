"""Data collection and normalization helpers.

AniList metadata is catalog information. Google Trends is a relative-interest
proxy. Neither source alone proves commercial demand.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import requests

ANILIST_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title { romaji english native }
      genres
      averageScore
      popularity
      episodes
      startDate { year month day }
      format
      status
      countryOfOrigin
    }
  }
}
"""

def collect_anilist(output_path: str | Path, pages: int = 5, per_page: int = 50) -> pd.DataFrame:
    """Fetch public anime metadata via AniList GraphQL and save CSV."""
    if pages < 1 or per_page < 1 or per_page > 50:
        raise ValueError("pages must be >=1 and per_page must be between 1 and 50")
    records = []
    for page in range(1, pages + 1):
        response = requests.post(
            ANILIST_URL,
            json={"query": ANILIST_QUERY, "variables": {"page": page, "perPage": per_page}},
            timeout=30,
            headers={"User-Agent": "SAFORA-hackathon-prototype/0.1"}
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"AniList API error: {payload['errors']}")
        for item in payload["data"]["Page"]["media"]:
            title_obj = item.get("title") or {}
            date_obj = item.get("startDate") or {}
            records.append({
                "anilist_id": item.get("id"),
                "title": title_obj.get("english") or title_obj.get("romaji") or title_obj.get("native"),
                "title_romaji": title_obj.get("romaji"),
                "genres": ", ".join(item.get("genres") or []),
                "average_score": item.get("averageScore"),
                "popularity": item.get("popularity"),
                "episodes": item.get("episodes"),
                "year": date_obj.get("year"),
                "format": item.get("format"),
                "status": item.get("status"),
                "country_of_origin": item.get("countryOfOrigin"),
                "metadata_source": "AniList",
            })
        time.sleep(1.0)
    df = pd.DataFrame(records).drop_duplicates(subset=["anilist_id"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def collect_google_trends(
    titles: list[str], output_path: str | Path, geo: str = "IN",
    timeframe: str = "today 5-y", sleep_seconds: float = 8.0
) -> pd.DataFrame:
    """Collect Trends via unofficial pytrends. This may be rate-limited/break."""
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise RuntimeError("Install pytrends: pip install pytrends") from exc
    if not titles:
        raise ValueError("Provide at least one title/search term")
    client = TrendReq(hl="en-IN", tz=330, timeout=(10, 30), retries=1, backoff_factor=1)
    frames = []
    for term in titles:
        client.build_payload([term], timeframe=timeframe, geo=geo)
        data = client.interest_over_time()
        if data.empty or term not in data.columns:
            print(f"Warning: no Trends data returned for {term}")
            continue
        data = data.reset_index()
        date_col = "date" if "date" in data.columns else data.columns[0]
        out = pd.DataFrame({
            "date": pd.to_datetime(data[date_col]),
            "title": term,
            "search_interest": pd.to_numeric(data[term], errors="coerce"),
            "geo": geo,
            "source": "Google Trends",
            "query_term": term,
            "timeframe": timeframe,
        })
        if "isPartial" in data.columns:
            out["is_partial"] = data["isPartial"]
        frames.append(out)
        time.sleep(max(1.0, sleep_seconds))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "title", "search_interest", "geo", "source"]
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        result.to_csv(path, index=False)
    return result

def normalize_trends_csv(input_path: str | Path, output_path: str | Path, title: str,
                         geo: str = "IN") -> pd.DataFrame:
    """Normalize a simple Google Trends CSV export with date and interest columns."""
    raw = pd.read_csv(input_path)
    if raw.empty or len(raw.columns) < 2:
        raise ValueError("CSV must contain a date column and an interest column")
    date_col = next((c for c in raw.columns if c.lower() in {"date", "week", "month", "time"}), raw.columns[0])
    possible = [c for c in raw.columns if c != date_col and "partial" not in c.lower()]
    if not possible:
        raise ValueError("Could not find an interest column")
    interest_col = possible[0]
    result = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "title": title,
        "search_interest": pd.to_numeric(raw[interest_col], errors="coerce"),
        "geo": geo,
        "source": "Google Trends CSV export",
    }).dropna(subset=["date", "search_interest"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
