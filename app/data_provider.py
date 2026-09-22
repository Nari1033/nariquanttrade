"""Price history data sources for the app.

Two sources are supported:

- "sample": bundled SYNTHETIC csvs under data/sample_prices/ (see
  scripts/generate_sample_data.py). Works with zero setup, zero network,
  zero API key -- this is what makes the app runnable immediately. Only
  daily bars exist on disk; weekly/monthly are produced by resampling them
  on the fly, and intraday intervals aren't available at all.
- "yfinance": real historical data via the free `yfinance` package (no API
  key required). Needs `pip install yfinance` and internet access, neither
  of which is available in this dev sandbox, so this path is not covered by
  the automated tests -- it's plain, standard yfinance usage.

Both paths return the same canonical shape (a pandas DataFrame as produced
by engine.data_utils.to_dataframe), so the rest of the app never has to
care which source it's looking at.

Live fetches are cached to disk (see app.cache) so the app doesn't have to
hit yfinance on every scan/backtest -- a fetch is reused for
`max_age_hours` before it's fetched again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app import cache
from engine.data_utils import resample_ohlcv, to_dataframe, trim_date_range

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_prices"


def list_sample_tickers() -> List[str]:
    return sorted(p.stem for p in SAMPLE_DATA_DIR.glob("*.csv"))


def load_sample_ticker(
    ticker: str,
    start=None,
    end=None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Load bundled sample data, optionally trimmed to [start, end] and/or
    resampled to a coarser interval ('1d' the underlying data already is,
    '1wk'/'1mo' are derived by resampling). Raises ValueError for an
    interval the sample data can't produce (e.g. intraday)."""
    path = SAMPLE_DATA_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        raise ValueError(
            f"No bundled sample data for '{ticker}'. Available: {list_sample_tickers()}"
        )
    df = to_dataframe(pd.read_csv(path))
    df = trim_date_range(df, start=start, end=end)
    if interval != "1d":
        df = resample_ohlcv(df, interval)
    return df


def load_sample_universe(start=None, end=None, interval: str = "1d") -> Dict[str, pd.DataFrame]:
    return {
        t: load_sample_ticker(t, start=start, end=end, interval=interval)
        for t in list_sample_tickers()
    }


def range_key(period: Optional[str], start, end) -> str:
    """A cache-key-safe token for whichever of (period) or (start, end) was
    used to fetch a series. Public so callers (e.g. the GUI) can compute the
    same key to look up cache_age_hours() for a given request shape."""
    if start is not None or end is not None:
        s = pd.Timestamp(start).date().isoformat() if start is not None else "earliest"
        e = pd.Timestamp(end).date().isoformat() if end is not None else "latest"
        return f"{s}_to_{e}"
    return period or "max"


def _download_from_yfinance(
    ticker: str, period: Optional[str], start, end, interval: str
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is not installed. Run: pip install yfinance") from exc

    kwargs = dict(interval=interval, progress=False, auto_adjust=False, group_by="column")
    if start is not None or end is not None:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period or "5y"

    raw = yf.download(ticker, **kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")

    # yfinance can return MultiIndex columns even for a single ticker
    # depending on version; flatten to simple lowercase names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.rename(columns=str.lower)
    raw.index.name = "date"
    return to_dataframe(raw[["open", "high", "low", "close", "volume"]])


def fetch_yfinance_ticker(
    ticker: str,
    period: Optional[str] = "5y",
    start=None,
    end=None,
    interval: str = "1d",
    use_cache: bool = True,
    max_age_hours: float = 12.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch real historical OHLCV data for one ticker via yfinance, backed
    by a local disk cache.

    Either pass `period` ('1y','2y','5y','10y','ytd','max', ...) or an
    explicit `start`/`end` date range -- if start/end are given they take
    priority over period, matching yfinance's own behavior.

    Caching: a fresh (<= max_age_hours old) cached copy is returned without
    touching the network. Set `force_refresh=True` to bypass the cache and
    re-fetch; set `use_cache=False` to disable caching for this call
    entirely (no read, no write). If a live fetch fails (offline, rate
    limited, bad symbol that used to be good, ...) and *any* cached copy
    exists -- even a stale one -- it's returned as a fallback rather than
    raising.
    """
    ticker = ticker.strip().upper()
    key = range_key(period, start, end)

    if use_cache and not force_refresh:
        cached = cache.read_cache(ticker, key, interval, max_age_hours=max_age_hours)
        if cached is not None:
            return cached

    try:
        df = _download_from_yfinance(ticker, period, start, end, interval)
    except Exception:
        if use_cache:
            stale = cache.read_cache(ticker, key, interval, max_age_hours=float("inf"))
            if stale is not None:
                return stale
        raise

    if use_cache:
        cache.write_cache(df, ticker, key, interval)
    return df


def fetch_yfinance_universe(
    tickers: List[str],
    period: Optional[str] = "1y",
    start=None,
    end=None,
    interval: str = "1d",
    use_cache: bool = True,
    max_age_hours: float = 12.0,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch several tickers, skipping any that fail (bad symbol, no data,
    rate limited, etc.) rather than aborting the whole scan. See
    fetch_yfinance_ticker for the caching parameters."""
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t.strip().upper()] = fetch_yfinance_ticker(
                t.strip(),
                period=period,
                start=start,
                end=end,
                interval=interval,
                use_cache=use_cache,
                max_age_hours=max_age_hours,
                force_refresh=force_refresh,
            )
        except Exception:
            continue
    return out


def get_price_history(
    ticker: str,
    source: str = "sample",
    period: Optional[str] = "5y",
    start=None,
    end=None,
    interval: str = "1d",
    **cache_kwargs,
) -> pd.DataFrame:
    if source == "sample":
        return load_sample_ticker(ticker, start=start, end=end, interval=interval)
    if source == "yfinance":
        return fetch_yfinance_ticker(
            ticker, period=period, start=start, end=end, interval=interval, **cache_kwargs
        )
    raise ValueError(f"Unknown source: {source!r} (expected 'sample' or 'yfinance')")
