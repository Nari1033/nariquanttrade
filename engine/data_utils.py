"""Normalize the various accepted input shapes into a canonical DataFrame."""

from __future__ import annotations

import pandas as pd

from .models import PriceBar, PriceHistory

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def to_dataframe(bars: PriceHistory) -> pd.DataFrame:
    """Normalize `bars` into a DataFrame sorted ascending by date, indexed by
    a DatetimeIndex named 'date', with float columns open/high/low/close/volume.

    Accepts:
      - a list of PriceBar
      - a list of dicts with keys date/open/high/low/close/volume
      - a pandas DataFrame already containing those columns (with either a
        'date' column or a DatetimeIndex)
    """
    if isinstance(bars, pd.DataFrame):
        df = bars.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
    else:
        records = []
        for row in bars:
            if isinstance(row, PriceBar):
                bar = row
            elif isinstance(row, dict):
                bar = PriceBar.from_dict(row)
            else:
                raise TypeError(
                    f"Unsupported price bar type: {type(row)!r}. Use PriceBar, "
                    "dict, or a pandas DataFrame."
                )
            records.append(
                {
                    "date": pd.Timestamp(bar.date),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )
        if not records:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        df = pd.DataFrame.from_records(records).set_index("date")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Price history is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].astype(float)
    # Drop bars with a missing OHLC price -- most commonly a live source's
    # (e.g. yfinance) row for the most recent/in-progress session, returned
    # with NaN open/high/low/close before that day's trading has produced a
    # real price. Left in, a single NaN close propagates through every
    # downstream calculation that touches it (equity curve, SMAs, returns),
    # silently turning a whole backtest's results into "nan%" instead of a
    # real number. Volume alone being missing isn't reason to drop a bar
    # (some sources omit it for otherwise-valid days), so it's just zeroed.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def trim_date_range(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Slice a canonical (to_dataframe'd) DataFrame to [start, end] inclusive.
    `start`/`end` may be None (open-ended), a date, or anything pd.Timestamp
    accepts. No-op if both are None."""
    if start is None and end is None:
        return df
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    return df.loc[start_ts:end_ts]


_RESAMPLE_RULES = {
    # (offset alias, label, closed) chosen so the resulting bin label is the
    # *first* calendar day of the bucket (e.g. the Monday of that trading
    # week, or the 1st of that month) rather than pandas' default
    # week/month-*end* label -- which can fall days after the last real
    # trading day in the bucket, and after a caller's end-date boundary.
    "1d": None,  # already daily, no resampling needed
    "1wk": ("W-MON", "left", "left"),
    "1mo": ("MS", "left", "left"),
}


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample a daily canonical DataFrame up to a coarser interval
    ('1wk' or '1mo'). Used for the bundled (daily-only) sample data so the
    interval picker still does something meaningful without live data.
    Raises ValueError for an interval this can't produce from daily bars
    (e.g. '1h' -- there's no intraday data to downsample from).
    """
    if interval not in _RESAMPLE_RULES:
        raise ValueError(
            f"Can't resample daily data to interval {interval!r}. "
            f"Supported: {sorted(_RESAMPLE_RULES)}."
        )
    spec = _RESAMPLE_RULES[interval]
    if spec is None:
        return df
    rule, label, closed = spec
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label=label, closed=closed).agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
