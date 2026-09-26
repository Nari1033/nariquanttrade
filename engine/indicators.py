"""Technical indicators: Simple Moving Averages and RSI."""

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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, using Wilder's original smoothing (an
    exponential moving average with alpha=1/period applied to gains and
    losses separately) -- the standard RSI definition. NaN for the first
    `period` bars while the smoothing warms up, a value in [0, 100] after
    that.

    Edge cases in the average-loss-is-zero region (a stretch with no down
    bars at all within the smoothing window):
      - average gain > 0 (a pure uptrend) -> RSI = 100, the conventional
        "maximally overbought" reading rather than a division by zero.
      - average gain == 0 too (price hasn't moved at all) -> RSI = 50,
        since there's no directional information to read either way.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    no_loss = avg_loss == 0
    result = result.where(~no_loss, 100.0)
    result = result.where(~(no_loss & (avg_gain == 0)), 50.0)
    return result
