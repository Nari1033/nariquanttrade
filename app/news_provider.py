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
import re
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


# ---------------------------------------------------------------------------
# "Top movers" -- a heuristic guess at which fetched headlines are most
# likely to move their primary ticker's price meaningfully (an FDA approval,
# an acquisition, a guidance cut -- not a routine insider Form 4 sale or a
# same-direction-as-the-market dip). This is NOT a prediction of direction,
# just "worth a second look" -- a plain keyword/relevance heuristic, not a
# model, so it will miss things and occasionally flag something mundane.
# ---------------------------------------------------------------------------

# Catalyst-style phrases, weighted by how decisively they tend to move a
# stock on their own. Matched case-insensitively as substrings of
# "title summary", so e.g. "phase 3" also catches "Phase 3 trial".
CATALYST_KEYWORDS: Dict[str, float] = {
    # Regulatory / clinical (biotech & pharma catalysts are often the
    # single biggest single-day movers of any sector)
    "fda approv": 3.0,
    "fda reject": 3.0,
    "fda declin": 2.5,
    "complete response letter": 3.0,
    "clinical hold": 2.5,
    "phase 3": 2.5,
    "phase iii": 2.5,
    "phase 2": 2.0,
    "phase ii": 2.0,
    "clinical trial": 1.5,
    "trial success": 3.0,
    "trial met": 2.5,
    "trial fail": 3.0,
    "primary endpoint": 2.5,
    "breakthrough therapy": 2.0,
    "recall": 2.0,
    # Corporate actions
    "to be acquired": 3.0,
    "acquisition": 2.0,
    "acquire": 1.5,
    "merger": 2.0,
    "buyout": 2.5,
    "takeover": 2.5,
    "tender offer": 2.0,
    "bankruptcy": 3.0,
    "chapter 11": 3.0,
    "delisting": 2.5,
    "spinoff": 1.5,
    "spin-off": 1.5,
    # Earnings / guidance surprises
    "guidance cut": 2.5,
    "cuts guidance": 2.5,
    "raises guidance": 2.0,
    "profit warning": 2.5,
    "beats estimates": 1.0,
    "misses estimates": 1.0,
    # Legal / regulatory trouble
    "sec investigation": 2.5,
    "fraud": 2.0,
    "indictment": 2.5,
    "class action": 1.0,
    "data breach": 2.0,
    # Leadership shakeups
    "ceo resigns": 2.0,
    "ceo steps down": 2.0,
    "ceo fired": 2.5,
    # Deals
    "licensing deal": 1.5,
    "contract win": 1.5,
    "partnership": 0.5,
}

# Words too common to mean anything for duplicate-detection.
_TITLE_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is",
    "are", "its", "after", "before", "stock", "shares", "inc", "corp",
    "co", "ltd", "at", "with", "into", "as",
}


def matched_catalysts(item: NewsItem) -> List[str]:
    """Which CATALYST_KEYWORDS phrases appear in this item's title or
    summary -- exposed (not just used internally by impact_score) so the
    UI can show *why* something was flagged as a potential mover."""
    text = f"{item.title} {item.summary}".lower()
    return [kw for kw in CATALYST_KEYWORDS if kw in text]


def impact_score(item: NewsItem) -> float:
    """Heuristic score for how likely this headline is to move its primary
    ticker's price meaningfully. Combines three signals: catalyst-keyword
    hits (weighted by keyword), how extreme the sentiment is in EITHER
    direction (a strong bearish surprise is just as price-moving as a
    strong bullish one), and how squarely the article is about one
    specific ticker (relevance_score) -- a roundup piece that mentions ten
    names in passing shouldn't outrank a story that's entirely about one."""
    keyword_score = sum(CATALYST_KEYWORDS[kw] for kw in matched_catalysts(item))
    sentiment_score = abs(item.overall_sentiment_score) * 2.0
    primary = item.primary_ticker
    relevance_score = (primary.relevance_score if primary else 0.0) * 1.5
    return keyword_score + sentiment_score + relevance_score


def _title_tokens(title: str) -> set:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in _TITLE_STOPWORDS and len(w) > 2}


def _are_near_duplicates(a: NewsItem, b: NewsItem) -> bool:
    """True if `a` and `b` look like separate outlets covering the exact
    same story -- same primary ticker AND enough title-word overlap
    (Jaccard similarity >= 0.5) that they're very unlikely to be two
    different stories about the same company."""
    a_primary = a.primary_ticker.ticker if a.primary_ticker else None
    b_primary = b.primary_ticker.ticker if b.primary_ticker else None
    if a_primary is None or a_primary != b_primary:
        return False
    ta, tb = _title_tokens(a.title), _title_tokens(b.title)
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    return jaccard >= 0.5


def top_movers(items: List[NewsItem], n: int = 5) -> List[NewsItem]:
    """The `n` headlines most likely to move their primary ticker
    meaningfully, ranked by impact_score (highest first), with
    near-duplicate coverage of the same underlying story collapsed to just
    the single highest-scoring copy -- so the result never shows the same
    event twice even if several outlets ran near-identical headlines on
    it."""
    ranked = sorted(items, key=impact_score, reverse=True)
    selected: List[NewsItem] = []
    for candidate in ranked:
        if any(_are_near_duplicates(candidate, kept) for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= n:
            break
    return selected
