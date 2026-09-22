"""Crossover detection and multi-ticker scanning.

'Golden Cross' = fast SMA (e.g. SMA-50) crosses from at-or-below to above the
slow SMA (e.g. SMA-200). 'Death Cross' is the opposite. These functions also
generalize to a price series crossing a single SMA (e.g. "price crosses above
its 50-day SMA").
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from .data_utils import to_dataframe
from .indicators import add_sma_columns
from .models import PriceHistory


def crossover_series(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Return an int Series aligned to `fast`'s index: +1 on the bar where
    `fast` crosses from <= `slow` to > `slow` ("golden" cross), -1 on the bar
    where it crosses from >= `slow` to < `slow` ("death" cross), 0 otherwise.
    Bars where either series is still NaN (e.g. during SMA warm-up) never
    register a cross.
    """
    diff = fast - slow
    prev_diff = diff.shift(1)
    valid = diff.notna() & prev_diff.notna()

    golden = valid & (diff > 0) & (prev_diff <= 0)
    death = valid & (diff < 0) & (prev_diff >= 0)

    result = pd.Series(0, index=diff.index, dtype=int)
    result[golden] = 1
    result[death] = -1
    return result


def _recent_window(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    return df.tail(lookback_days)


def golden_cross_recent(
    bars: PriceHistory,
    fast_window: int = 50,
    slow_window: int = 200,
    lookback_days: int = 3,
) -> bool:
    """True if SMA-`fast_window` crossed above SMA-`slow_window` on any of the
    last `lookback_days` trading days (inclusive of the most recent bar)."""
    df = add_sma_columns(to_dataframe(bars), windows=(fast_window, slow_window))
    cross = crossover_series(df[f"sma_{fast_window}"], df[f"sma_{slow_window}"])
    return bool((_recent_window(cross.to_frame("c"), lookback_days)["c"] == 1).any())


def death_cross_recent(
    bars: PriceHistory,
    fast_window: int = 50,
    slow_window: int = 200,
    lookback_days: int = 3,
) -> bool:
    """True if SMA-`fast_window` crossed below SMA-`slow_window` on any of the
    last `lookback_days` trading days."""
    df = add_sma_columns(to_dataframe(bars), windows=(fast_window, slow_window))
    cross = crossover_series(df[f"sma_{fast_window}"], df[f"sma_{slow_window}"])
    return bool((_recent_window(cross.to_frame("c"), lookback_days)["c"] == -1).any())


def price_cross_sma_recent(
    bars: PriceHistory,
    sma_window: int = 50,
    lookback_days: int = 3,
    direction: str = "above",
    price_col: str = "close",
) -> bool:
    """True if `price_col` crossed above (direction='above') or below
    (direction='below') its SMA-`sma_window` on any of the last
    `lookback_days` trading days."""
    if direction not in ("above", "below"):
        raise ValueError("direction must be 'above' or 'below'")
    df = add_sma_columns(to_dataframe(bars), windows=(sma_window,))
    cross = crossover_series(df[price_col], df[f"sma_{sma_window}"])
    target = 1 if direction == "above" else -1
    return bool((_recent_window(cross.to_frame("c"), lookback_days)["c"] == target).any())


def scan_universe(
    price_histories: Dict[str, PriceHistory],
    strategy_fn: Callable[..., bool] = golden_cross_recent,
    **strategy_kwargs,
) -> List[str]:
    """Apply `strategy_fn` to every (ticker -> price history) pair and return
    the list of tickers for which it returned True. Tickers whose data can't
    be evaluated (e.g. too short a history) are skipped rather than raising.
    """
    matches: List[str] = []
    for ticker, bars in price_histories.items():
        try:
            if strategy_fn(bars, **strategy_kwargs):
                matches.append(ticker)
        except Exception:
            continue
    return matches
