"""Alpha Vantage NEWS_SENTIMENT integration -- trending market news + sentiment.

Free-tier Alpha Vantage keys are capped at 25 requests/day (see the README's
API comparison table in section 3), so this module is deliberately thrifty:
the app wraps `fetch_trending_news` in `st.cache_data(ttl=...)` (see
app/app.py) so repeated page loads with the same filters don't re-hit the
API, and this module itself makes at most one request per call -- no retry
loops, no pagination.

Alpha Vantage's failure modes (bad/missing key, malformed params, and the
free-tier rate limit) all come back as HTTP 200 with an "Error Message",
"Note", or "Information" key in the JSON body instead of a non-2xx status
code. `_check_for_api_error` checks for those explicitly rather than
trusting the HTTP status alone -- a `requests.raise_for_status()` call would
silently let a rate-limited response through as "success" with no feed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

API_URL = "https://www.alphavantage.co/query"
REQUEST_TIMEOUT_S = 15

# Order matches Alpha Vantage's own sentiment_score_definition thresholds
# (<= -0.35 Bearish ... >= 0.35 Bullish), used both for bucket counts and
# for a stable left-to-right axis on the sentiment-mix chart.
SENTIMENT_ORDER = ["Bearish", "Somewhat-Bearish", "Neutral", "Somewhat-Bullish", "Bullish"]

# The `topics` values Alpha Vantage documents for NEWS_SENTIMENT.
NEWS_TOPICS = [
    "blockchain",
    "earnings",
    "ipo",
    "mergers_and_acquisitions",
    "financial_markets",
    "economy_fiscal",
    "economy_monetary",
    "economy_macro",
    "energy_transportation",
    "finance",
    "life_sciences",
    "manufacturing",
    "real_estate",
    "retail_wholesale",
    "technology",
]


@dataclass
class TickerSentiment:
    ticker: str
    relevance_score: float
    sentiment_score: float
    sentiment_label: str


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    time_published: dt.datetime
    summary: str
    overall_sentiment_score: float
    overall_sentiment_label: str
    tickers: List[TickerSentiment] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    @property
    def primary_ticker(self) -> Optional[TickerSentiment]:
        """The most-relevant ticker mentioned in this article, if any --
        used as the headline's "top ticker" in the UI."""
        return max(self.tickers, key=lambda t: t.relevance_score) if self.tickers else None


class NewsFetchError(Exception):
    """Anything that stops us from returning a usable feed -- a missing or
    invalid API key, Alpha Vantage's rate limit, or a network failure.
    `.message` is short and safe to show directly in the UI."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _parse_time_published(raw: str) -> dt.datetime:
    # Alpha Vantage's own format, e.g. "20260924T020801".
    return dt.datetime.strptime(raw, "%Y%m%dT%H%M%S")


def parse_feed(raw: dict) -> List[NewsItem]:
    """Turn a raw NEWS_SENTIMENT JSON body into a list of NewsItem. A row
    missing a field this module relies on is skipped rather than raising --
    one malformed row from a live, third-party feed shouldn't blank the
    whole page."""
    items: List[NewsItem] = []
    for row in raw.get("feed", []) or []:
        try:
            tickers = [
                TickerSentiment(
                    ticker=t["ticker"],
                    relevance_score=float(t["relevance_score"]),
                    sentiment_score=float(t["ticker_sentiment_score"]),
                    sentiment_label=t["ticker_sentiment_label"],
                )
                for t in row.get("ticker_sentiment", []) or []
            ]
            items.append(
                NewsItem(
                    title=row["title"],
                    url=row["url"],
                    source=row.get("source", "") or row.get("source_domain", ""),
                    time_published=_parse_time_published(row["time_published"]),
                    summary=row.get("summary", ""),
                    overall_sentiment_score=float(row.get("overall_sentiment_score", 0.0)),
                    overall_sentiment_label=row.get("overall_sentiment_label", "Neutral"),
                    tickers=tickers,
                    topics=[t["topic"] for t in row.get("topics", []) or [] if "topic" in t],
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return items


def _check_for_api_error(raw: dict) -> None:
    """Raise NewsFetchError if `raw` is one of Alpha Vantage's HTTP-200
    error bodies (bad key, bad params, or the free-tier rate limit)."""
    if not isinstance(raw, dict):
        raise NewsFetchError("Alpha Vantage returned an unexpected response shape.")
    for key in ("Error Message", "Note", "Information"):
        if key in raw:
            raise NewsFetchError(str(raw[key]))


def fetch_trending_news(
    api_key: str,
    tickers: Optional[str] = None,
    topics: Optional[str] = None,
    sort: str = "LATEST",
    limit: int = 20,
) -> List[NewsItem]:
    """Fetch and parse NEWS_SENTIMENT. Raises NewsFetchError on any failure
    (missing/invalid key, rate limit, network/timeout, bad JSON) with a
    message that's safe to show directly in the UI. `tickers` is a
    comma-separated string (e.g. "AAPL,TSLA"); `topics` is one of
    NEWS_TOPICS; `sort` is "LATEST" or "RELEVANCE"."""
    if not api_key:
        raise NewsFetchError(
            "No Alpha Vantage API key set. Add one in the sidebar, or as "
            "ALPHAVANTAGE_API_KEY in .streamlit/secrets.toml."
        )

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "sort": sort,
        "limit": str(limit),
    }
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics

    try:
        resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as exc:
        raise NewsFetchError(f"Couldn't reach Alpha Vantage: {exc}") from exc
    except ValueError as exc:
        raise NewsFetchError("Alpha Vantage returned an unreadable response.") from exc

    _check_for_api_error(raw)
    return parse_feed(raw)


def sentiment_distribution(items: List[NewsItem]) -> Dict[str, int]:
    """Count of items per overall_sentiment_label, keyed in SENTIMENT_ORDER
    (0 for a bucket with no items, rather than a missing key) -- so the
    chart always shows all five bars."""
    counts = {label: 0 for label in SENTIMENT_ORDER}
    for item in items:
        counts[item.overall_sentiment_label] = counts.get(item.overall_sentiment_label, 0) + 1
    return counts
