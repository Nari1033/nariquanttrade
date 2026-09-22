"""Technical indicators: currently just Simple Moving Averages."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average. NaN until `window` observations are available
    (no partial-window averages, matching how SMA-50/SMA-200 are normally
    defined for trading signals)."""
    if window < 1:
        raise ValueError("window must be a positive integer")
    return series.rolling(window=window, min_periods=window).mean()


def add_sma_columns(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (50, 200),
    price_col: str = "close",
) -> pd.DataFrame:
    """Return a copy of `df` with an `sma_{window}` column added for each
    window in `windows`."""
    out = df.copy()
    for window in windows:
        out[f"sma_{window}"] = sma(out[price_col], window)
    return out
