"""Local on-disk cache for live (yfinance) price data.

The whole point: don't hit the live API every time the Scanner or Backtest
tab runs. A fetched ticker/period/interval is written to a CSV under
data/cache/ and reused for any request within `max_age_hours` of the last
fetch, instead of calling yfinance again. Sample data never touches this
cache -- it's already local, there's nothing to save.

Nothing in this module talks to the network. It's pure file I/O so it's
fully testable without yfinance installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from engine.data_utils import to_dataframe

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe_ticker = ticker.strip().upper().replace("/", "-")
    return CACHE_DIR / f"{safe_ticker}_{period}_{interval}.csv"


def cache_age_hours(ticker: str, period: str, interval: str = "1d") -> Optional[float]:
    """Hours since this (ticker, period, interval) was last cached, or None
    if nothing is cached for it yet."""
    path = _cache_path(ticker, period, interval)
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def read_cache(
    ticker: str, period: str, interval: str = "1d", max_age_hours: float = 12.0
) -> Optional[pd.DataFrame]:
    """Return the cached DataFrame if a fresh-enough (<= max_age_hours)
    entry exists, else None. A corrupt cache file is treated as a miss
    rather than raising."""
    age = cache_age_hours(ticker, period, interval)
    if age is None or age > max_age_hours:
        return None
    try:
        return to_dataframe(pd.read_csv(_cache_path(ticker, period, interval)))
    except Exception:
        return None


def write_cache(df: pd.DataFrame, ticker: str, period: str, interval: str = "1d") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(_cache_path(ticker, period, interval), index=False)


def list_cache_entries() -> List[dict]:
    """List every cached (ticker, period, interval), with age and row
    count, for display in the UI."""
    if not CACHE_DIR.exists():
        return []
    entries = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        ticker, period, interval = parts
        try:
            n_rows = max(sum(1 for _ in open(path)) - 1, 0)
        except OSError:
            n_rows = None
        entries.append(
            {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "age_hours": round((time.time() - path.stat().st_mtime) / 3600.0, 1),
                "rows": n_rows,
            }
        )
    return entries


def clear_cache(ticker: Optional[str] = None) -> int:
    """Delete cached files. Only `ticker`'s entries (any period/interval) if
    given, otherwise the whole cache. Returns how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    pattern = f"{ticker.strip().upper()}_*.csv" if ticker else "*.csv"
    deleted = 0
    for path in CACHE_DIR.glob(pattern):
        path.unlink()
        deleted += 1
    return deleted
