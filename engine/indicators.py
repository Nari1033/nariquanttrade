"""Technical indicators: Simple/Exponential Moving Averages, RSI, and ATR."""

from __future__ import annotations

import numpy as np
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


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average with span=period (the conventional
    definition -- alpha = 2/(period+1)). NaN until `period` observations
    are available, same warm-up convention as sma()."""
    if period < 1:
        raise ValueError("period must be a positive integer")
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, using Wilder's original smoothing (same
    alpha=1/period exponential smoothing as rsi()'s gain/loss averages).

    True Range for a bar is the largest of: high-low, |high - prev_close|,
    |low - prev_close| -- the first bar has no prev_close, so its True
    Range is just high-low. NaN for the first `period` bars while the
    smoothing warms up, matching rsi()'s convention.

    `df` must have 'high', 'low', 'close' columns (e.g. the output of
    engine.data_utils.to_dataframe).
    """
    if period < 1:
        raise ValueError("period must be a positive integer")
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's original smoothing, the textbook convention behind RSI's
    average gain/loss: seed with a plain simple average of the first
    `period` valid values, then from there weight the running average
    (period-1)/period against 1/period for each new value -- i.e.
    avg[t] = (avg[t-1] * (period - 1) + values[t]) / period.

    This differs from a plain EMA (alpha=1/period, decaying from the very
    first raw observation) only in how the first value is seeded, but
    that seed matters: an EMA-seeded-from-bar-0 series takes dozens of
    bars to converge to the same numbers this produces immediately after
    warm-up, which is why Wilder's original RSI/ATR figures (and most
    trading platforms) use this seeding rather than a plain EMA. NaN
    until `period` valid (non-NaN) values have been seen.
    """
    arr = values.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    valid = ~np.isnan(arr)
    if valid.any():
        first_valid = int(np.argmax(valid))
        seed_end = first_valid + period  # exclusive end of the seed window
        if seed_end <= len(arr):
            out[seed_end - 1] = arr[first_valid:seed_end].mean()
            for i in range(seed_end, len(arr)):
                out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return pd.Series(out, index=values.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, using Wilder's original smoothing (see
    _wilder_smooth) applied to gains and losses separately -- the
    standard, textbook RSI definition, matching TradingView and most
    trading platforms bar-for-bar (not just asymptotically). NaN for the
    first `period` bars while the smoothing warms up, a value in [0, 100]
    after that.

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
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    no_loss = avg_loss == 0
    result = result.where(~no_loss, 100.0)
    result = result.where(~(no_loss & (avg_gain == 0)), 50.0)
    return result


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average: a linearly-weighted average over the last
    `period` observations, with the most recent observation weighted
    heaviest (weight `period`) and the oldest weighted lightest (weight 1)
    -- so it reacts to new data faster than a plain SMA but smoother than
    an EMA. NaN until `period` observations are available, same warm-up
    convention as sma()/ema()."""
    if period < 1:
        raise ValueError("period must be a positive integer")
    weights = pd.Series(range(1, period + 1), dtype=float)
    weight_sum = weights.sum()
    return series.rolling(window=period, min_periods=period).apply(
        lambda window: (window * weights.to_numpy()).sum() / weight_sum, raw=True
    )
