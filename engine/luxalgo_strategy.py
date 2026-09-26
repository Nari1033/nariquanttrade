"""LuxAlgo "Market Flow Trend Lines & Liquidity"-inspired strategies:
scan + backtest, in two flavors.

IMPORTANT CAVEAT: the real "Market Flow Trend Lines & Liquidity [LuxAlgo]"
indicator is a protected-source (closed) TradingView script -- its exact
formula for trend lines, liquidity zones, and gap detection is not public.
Everything below is our own reasonable, from-scratch interpretation of the
*published description* of what it does (auto trend lines from swing
highs/lows, liquidity zones around those same swings, an opening range,
gap flags, and ATR-based take-profits) -- not a byte-for-byte replica of
the original. Same spirit as this app's other strategies: a clearly
documented, testable approximation rather than a black box.

Two entry points, sharing all of the building blocks below:

  - `backtest_trendline_breakout_core` / `trendline_breakout_recent`:
    the "core piece" -- just the auto trend-line breakout plus ATR-based
    take-profit and stop-loss. This is a self-contained momentum/breakout
    strategy on its own.

  - `backtest_market_flow_full` / `market_flow_full_recent`: the "full
    package" -- the same trend-line breakout, but only taken when it also
    clears three extra filters: it breaks the day's Opening Range, it's
    trading in a "green" (rising, price-above) EMA trend, and it's
    happening near a recent liquidity zone (a cluster of prior swing
    highs/lows) rather than out in open air. Also skips new entries on a
    day that gapped down hard at the open. Both flavors use the same
    ATR-based take-profit/stop-loss risk management, but the full package
    supports up to 3 scaled take-profit levels (matching the real
    indicator's "up to 3 ATR based take profits") by splitting the
    position into equal legs at entry, each with its own target -- the
    core piece is just the one-leg special case of the same mechanism.

Building blocks (all causal -- no lookahead; a swing point is only usable
once enough bars have passed to confirm it was actually a swing):

  - Pivot highs/lows: a classic fractal definition -- bar i is a pivot high
    if it's strictly the highest high within `pivot_lookback` bars on
    either side (and a pivot low, the mirror). Flat/tied extremes never
    qualify, so dead-flat price data never manufactures a pivot.
  - A "trend line" is a least-squares line fit through the last
    `trendline_points` *confirmed* pivot highs (resistance) or pivot lows
    (support), projected forward to the current bar. A breakout is close
    crossing up through the resistance line.
  - A "liquidity zone" is loosely "price trading back within
    `liquidity_proximity_atr_mult` x ATR of *any* confirmed pivot high or
    low from the last `liquidity_lookback_bars` bars" -- a simple proxy
    for "this level has been fought over before," without the frequency
    clustering the real indicator may or may not do internally.
  - The Opening Range is the high/low of the first `opening_range_minutes`
    of each trading day, known from the bar right after that window
    closes onward. On daily bars (one bar per day) there's no such window,
    so the Opening Range is simply unavailable and its filter passes
    through rather than blocking every trade -- this lets the full-package
    strategy still run (minus that one filter) on the app's daily data
    sources, not just live intraday data.
  - A gap is just that day's open vs. the prior day's close -- this one
    *is* meaningful on daily bars too.

Long-only, like every other strategy in this app (no shorting).
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import atr as atr_indicator
from .indicators import ema as ema_indicator
from .models import PriceHistory


# ---------------------------------------------------------------------------
# Pivot / trend line building blocks
# ---------------------------------------------------------------------------


def _pivot_mask(series: pd.Series, left: int, right: int, kind: str) -> pd.Series:
    """Boolean Series, True at bar i if series[i] is the strict, unique
    extreme (max for kind='high', min for kind='low') within the window
    [i-left, i+right]. The last `right` bars can never be confirmed yet
    (not enough bars after them) and are always False -- this is what
    keeps every downstream use of this mask causal."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        center = values[i]
        if np.isnan(center) or np.isnan(window).any():
            continue
        if kind == "high":
            is_extreme = center == window.max()
        else:
            is_extreme = center == window.min()
        if is_extreme and np.sum(window == center) == 1:
            mask[i] = True
    return pd.Series(mask, index=series.index)


def _causal_trendline(series: pd.Series, mask: pd.Series, right: int, n_points: int) -> pd.Series:
    """Least-squares line fit through the last `n_points` *confirmed*
    pivots (per `mask`), projected forward to every bar -- a pivot at
    position p is usable starting at bar p+right (the earliest bar at
    which `_pivot_mask` could have marked it True). NaN until at least
    `n_points` pivots have been confirmed."""
    n = len(series)
    values = series.to_numpy(dtype=float)
    pivot_positions = np.nonzero(mask.to_numpy())[0]
    out = np.full(n, np.nan)
    window_positions: List[int] = []
    ptr = 0
    for i in range(n):
        while ptr < len(pivot_positions) and pivot_positions[ptr] <= i - right:
            window_positions.append(pivot_positions[ptr])
            ptr += 1
        if len(window_positions) >= n_points:
            xs = np.array(window_positions[-n_points:], dtype=float)
            ys = values[np.array(window_positions[-n_points:])]
            if np.ptp(xs) == 0:
                continue
            slope, intercept = np.polyfit(xs, ys, 1)
            out[i] = slope * i + intercept
    return pd.Series(out, index=series.index)


def _liquidity_zone_flags(
    close: pd.Series,
    high_mask: pd.Series,
    high_values: pd.Series,
    low_mask: pd.Series,
    low_values: pd.Series,
    right: int,
    lookback_bars: int,
    atr: pd.Series,
    proximity_atr_mult: float,
) -> pd.Series:
    """True at bar i if close[i] is within `proximity_atr_mult` x ATR[i] of
    any confirmed pivot high or low (from either series) whose position
    falls in the trailing `lookback_bars` window -- our proxy for "trading
    near a recent liquidity zone." False (never blocking) until at least
    one such pivot is confirmed and ATR has warmed up."""
    n = len(close)
    close_vals = close.to_numpy(dtype=float)
    atr_vals = atr.to_numpy(dtype=float)
    high_positions = np.nonzero(high_mask.to_numpy())[0]
    low_positions = np.nonzero(low_mask.to_numpy())[0]
    high_vals_arr = high_values.to_numpy(dtype=float)
    low_vals_arr = low_values.to_numpy(dtype=float)

    out = np.zeros(n, dtype=bool)
    recent: deque = deque()  # (confirm_position, price_level)
    hi_ptr = 0
    lo_ptr = 0
    for i in range(n):
        while hi_ptr < len(high_positions) and high_positions[hi_ptr] <= i - right:
            p = high_positions[hi_ptr]
            recent.append((p, high_vals_arr[p]))
            hi_ptr += 1
        while lo_ptr < len(low_positions) and low_positions[lo_ptr] <= i - right:
            p = low_positions[lo_ptr]
            recent.append((p, low_vals_arr[p]))
            lo_ptr += 1
        while recent and recent[0][0] < i - lookback_bars:
            recent.popleft()
        if not recent or np.isnan(atr_vals[i]):
            continue
        prox = proximity_atr_mult * atr_vals[i]
        c = close_vals[i]
        if any(abs(c - level) <= prox for _, level in recent):
            out[i] = True
    return pd.Series(out, index=close.index)


def _opening_range_levels(df: pd.DataFrame, minutes: int) -> Tuple[pd.Series, pd.Series]:
    """(or_high, or_low) Series aligned to df's index: the high/low of the
    first `minutes` of each trading day, forward-filled from the bar right
    after that window closes through the rest of that same day, and NaN
    before/during the window and on any day with fewer than 2 bars (daily
    data -- there's no sub-day window to measure, so the Opening Range is
    simply unavailable rather than degenerately equal to that one bar)."""
    or_high = pd.Series(np.nan, index=df.index)
    or_low = pd.Series(np.nan, index=df.index)
    day_keys = df.index.normalize()
    for day in pd.unique(day_keys):
        day_bars = df.loc[day_keys == day]
        if len(day_bars) < 2:
            continue
        day_start = day_bars.index[0]
        window_end = day_start + pd.Timedelta(minutes=minutes)
        or_window = day_bars[day_bars.index < window_end]
        if or_window.empty:
            continue
        or_h = float(or_window["high"].max())
        or_l = float(or_window["low"].min())
        confirmed_from = or_window.index[-1]
        after = day_bars.index[day_bars.index > confirmed_from]
        or_high.loc[after] = or_h
        or_low.loc[after] = or_l
    return or_high, or_low


def _daily_gap_pct(df: pd.DataFrame) -> pd.Series:
    """That day's open vs. the prior *trading* day's close, as a percent,
    forward-filled across every bar of that same day (so an intraday
    strategy can check "did today gap down hard" on any bar, not just the
    first). NaN for the very first day (no prior close to compare to)."""
    day_keys = df.index.normalize()
    daily_open = df.groupby(day_keys)["open"].first()
    daily_close = df.groupby(day_keys)["close"].last()
    prev_close = daily_close.shift(1)
    gap_pct = (daily_open - prev_close) / prev_close * 100.0
    return gap_pct.reindex(day_keys).set_axis(df.index)


# ---------------------------------------------------------------------------
# Shared signal computation
# ---------------------------------------------------------------------------


def _validate_common(pivot_lookback: int, trendline_points: int, atr_period: int) -> None:
    if pivot_lookback < 1:
        raise ValueError("pivot_lookback must be a positive integer")
    if trendline_points < 2:
        raise ValueError("trendline_points must be at least 2 (a line needs 2+ points)")
    if atr_period < 1:
        raise ValueError("atr_period must be a positive integer")


def trendline_signals(
    df: pd.DataFrame,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (bullish_breakout, resistance_line, support_line, atr,
    pivot_high_mask, pivot_low_mask). bullish_breakout is True on the bar
    close crosses up through the resistance trend line (built from
    confirmed pivot highs)."""
    _validate_common(pivot_lookback, trendline_points, atr_period)
    high_mask = _pivot_mask(df["high"], pivot_lookback, pivot_lookback, "high")
    low_mask = _pivot_mask(df["low"], pivot_lookback, pivot_lookback, "low")
    resistance_line = _causal_trendline(df["high"], high_mask, pivot_lookback, trendline_points)
    support_line = _causal_trendline(df["low"], low_mask, pivot_lookback, trendline_points)
    atr = atr_indicator(df, period=atr_period)

    close = df["close"]
    prev_close = close.shift(1)
    prev_line = resistance_line.shift(1)
    crossed_up = (close > resistance_line) & (prev_close <= prev_line)
    bullish_breakout = (crossed_up & resistance_line.notna() & prev_line.notna()).fillna(False)
    return bullish_breakout, resistance_line, support_line, atr, high_mask, low_mask


def trendline_breakout_recent(
    bars: PriceHistory,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
    lookback_days: int = 3,
) -> bool:
    """True if a trend-line breakout fired within the last `lookback_days`
    bars. Used by the Scanner tab -- works on any bar granularity
    (intraday or daily)."""
    df = to_dataframe(bars)
    min_bars = max(pivot_lookback * 2 + 1, atr_period) * trendline_points + 5
    if len(df) < min_bars:
        return False
    try:
        bullish_breakout, *_ = trendline_signals(
            df, pivot_lookback=pivot_lookback, trendline_points=trendline_points, atr_period=atr_period
        )
    except ValueError:
        return False
    return bool(bullish_breakout.iloc[-lookback_days:].any())


def market_flow_full_signals(
    df: pd.DataFrame,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
    ema_period: int = 20,
    opening_range_minutes: int = 15,
    liquidity_lookback_bars: int = 50,
    liquidity_proximity_atr_mult: float = 1.0,
    gap_filter_pct: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (full_bullish_entry, atr). full_bullish_entry is the core
    trend-line breakout AND-ed with: clears the day's Opening Range (or
    passes through if unavailable -- see module docstring), a rising
    EMA with price above it, near a recent liquidity zone, and not on a
    day that gapped down more than `gap_filter_pct`% at the open."""
    if ema_period < 1:
        raise ValueError("ema_period must be a positive integer")
    if opening_range_minutes < 1:
        raise ValueError("opening_range_minutes must be a positive integer")
    if liquidity_lookback_bars < 1:
        raise ValueError("liquidity_lookback_bars must be a positive integer")
    if liquidity_proximity_atr_mult <= 0:
        raise ValueError("liquidity_proximity_atr_mult must be positive")
    if gap_filter_pct < 0:
        raise ValueError("gap_filter_pct must be non-negative (0 disables the gap filter)")

    bullish_breakout, resistance_line, support_line, atr, high_mask, low_mask = trendline_signals(
        df, pivot_lookback=pivot_lookback, trendline_points=trendline_points, atr_period=atr_period
    )

    close = df["close"]
    ema_line = ema_indicator(close, period=ema_period)
    ema_rising = ema_line > ema_line.shift(1)
    ema_ok = (close > ema_line) & ema_rising.fillna(False)

    or_high, _or_low = _opening_range_levels(df, opening_range_minutes)
    or_ok = or_high.isna() | (close > or_high)

    liquidity_ok = _liquidity_zone_flags(
        close, high_mask, df["high"], low_mask, df["low"],
        pivot_lookback, liquidity_lookback_bars, atr, liquidity_proximity_atr_mult,
    )
    # Pass through (don't block) until at least one pivot has ever been
    # confirmed anywhere in the series -- otherwise this filter would
    # reject every single bar for the first `pivot_lookback` bars of the
    # whole dataset, which is a warm-up artifact, not a real "no liquidity
    # nearby" verdict.
    any_pivot_confirmed_yet = (high_mask.cumsum() + low_mask.cumsum()).shift(pivot_lookback).fillna(0) > 0
    liquidity_ok = liquidity_ok | ~any_pivot_confirmed_yet

    if gap_filter_pct > 0:
        gap_pct = _daily_gap_pct(df)
        gap_blocked = gap_pct <= -gap_filter_pct
        gap_ok = ~gap_blocked.fillna(False)
    else:
        gap_ok = pd.Series(True, index=df.index)

    full_bullish_entry = (
        bullish_breakout & ema_ok.fillna(False) & or_ok.fillna(True) & liquidity_ok & gap_ok
    )
    return full_bullish_entry.fillna(False), atr


def market_flow_full_recent(
    bars: PriceHistory,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
    ema_period: int = 20,
    opening_range_minutes: int = 15,
    liquidity_lookback_bars: int = 50,
    liquidity_proximity_atr_mult: float = 1.0,
    gap_filter_pct: float = 2.0,
    lookback_days: int = 3,
) -> bool:
    """True if a full-package entry fired within the last `lookback_days`
    bars. Used by the Scanner tab."""
    df = to_dataframe(bars)
    min_bars = max(pivot_lookback * 2 + 1, atr_period, ema_period, liquidity_lookback_bars) * 2 + 5
    if len(df) < min_bars:
        return False
    try:
        full_bullish_entry, _atr = market_flow_full_signals(
            df,
            pivot_lookback=pivot_lookback,
            trendline_points=trendline_points,
            atr_period=atr_period,
            ema_period=ema_period,
            opening_range_minutes=opening_range_minutes,
            liquidity_lookback_bars=liquidity_lookback_bars,
            liquidity_proximity_atr_mult=liquidity_proximity_atr_mult,
            gap_filter_pct=gap_filter_pct,
        )
    except ValueError:
        return False
    return bool(full_bullish_entry.iloc[-lookback_days:].any())


# ---------------------------------------------------------------------------
# Shared backtest simulation: entry signal + N scaled ATR take-profit legs
# ---------------------------------------------------------------------------


def _simulate_scaled_breakout(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    atr: pd.Series,
    tp_atr_mults: List[float],
    stop_loss_atr_mult: float,
    initial_capital: float,
) -> Tuple[List[Trade], pd.Series, float]:
    """Long/flat simulation shared by the core and full-package strategies.

    On an entry signal (when flat and ATR is available), splits equity
    into len(tp_atr_mults) equal legs, each targeting entry_price +
    tp_atr_mults[k] * ATR-at-entry -- so `tp_atr_mults=[2.0]` (one leg) is
    exactly the "core" single-target behavior, and a 3-element list is the
    full package's scaled take-profits. All legs share one stop-loss level
    (entry_price - stop_loss_atr_mult * ATR-at-entry, checked against each
    bar's intrabar low). A new entry is only taken once every leg from the
    previous one has closed (one position at a time, like every other
    strategy in this app). Any leg(s) still open when the data runs out
    are marked to market on the final bar (exit_reason "period_end")."""
    trades: List[Trade] = []
    closes = df["close"]
    lows = df["low"]
    last_i = len(df) - 1
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    n_legs = len(tp_atr_mults)
    open_legs: List[dict] = []  # each: entry_date, entry_price, shares, target, leg_idx

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]
        low = lows.iloc[i]
        at_period_end = i == last_i

        if open_legs:
            stop_price = open_legs[0]["stop_price"]
            still_open: List[dict] = []
            for leg in open_legs:
                hit_stop = bool(low <= leg["stop_price"])
                hit_target = bool(df["high"].iloc[i] >= leg["target"])
                if hit_stop or hit_target or at_period_end:
                    if hit_stop:
                        exit_price, exit_reason = leg["stop_price"], "stop_loss"
                    elif hit_target:
                        exit_price, exit_reason = leg["target"], f"take_profit_{leg['leg_idx'] + 1}"
                    else:
                        exit_price, exit_reason = close, "period_end"
                    trade_return = (exit_price - leg["entry_price"]) / leg["entry_price"]
                    equity += leg["shares"] * exit_price
                    trades.append(
                        Trade(
                            entry_date=leg["entry_date"],
                            entry_price=leg["entry_price"],
                            exit_date=dt,
                            exit_price=exit_price,
                            return_pct=trade_return * 100,
                            is_win=trade_return > 0,
                            closed_at_period_end=(exit_reason == "period_end"),
                            meta={"exit_reason": exit_reason, "leg": leg["leg_idx"] + 1, "of_legs": n_legs},
                        )
                    )
                else:
                    still_open.append(leg)
            open_legs = still_open

        if not open_legs and not at_period_end and bool(entry_signal.iloc[i]) and pd.notna(atr.iloc[i]):
            entry_atr = atr.iloc[i]
            stop_price = close - stop_loss_atr_mult * entry_atr
            per_leg_capital = equity / n_legs
            for leg_idx, mult in enumerate(tp_atr_mults):
                shares = per_leg_capital / close
                open_legs.append(
                    {
                        "entry_date": dt,
                        "entry_price": close,
                        "shares": shares,
                        "target": close + mult * entry_atr,
                        "stop_price": stop_price,
                        "leg_idx": leg_idx,
                    }
                )
            equity = 0.0

        mark_to_market = sum(leg["shares"] * close for leg in open_legs)
        equity_curve.iloc[i] = equity + mark_to_market

    final_equity = equity_curve.iloc[-1] if len(equity_curve) else initial_capital
    return trades, equity_curve, final_equity


def _build_luxalgo_result(
    df: pd.DataFrame,
    trades: List[Trade],
    equity_curve: pd.Series,
    final_equity: float,
    initial_capital: float,
    ticker: Optional[str],
    strategy_name: str,
) -> BacktestResult:
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate_pct = (wins / total_trades * 100) if total_trades else 0.0
    strategy_return_pct = (final_equity - initial_capital) / initial_capital * 100

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    buy_hold_return_pct = (last_close - first_close) / first_close * 100
    buy_hold_curve = (df["close"] / first_close) * initial_capital

    return BacktestResult(
        ticker=ticker,
        strategy_name=strategy_name,
        start_date=df.index[0],
        end_date=df.index[-1],
        fast_window=None,
        slow_window=None,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )


# ---------------------------------------------------------------------------
# Core piece: trend-line breakout + single ATR take-profit + ATR stop
# ---------------------------------------------------------------------------


def backtest_trendline_breakout_core(
    bars: PriceHistory,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
    take_profit_atr_mult: float = 2.0,
    stop_loss_atr_mult: float = 1.5,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """The "core piece": buy at the close on a trend-line breakout (close
    crosses up through the resistance line fit from recent confirmed pivot
    highs); exit at a single ATR-based take-profit, an ATR-based stop-loss
    (checked first if both would fire the same bar), or period end.

    Raises ValueError if there isn't enough data, or for invalid
    pivot_lookback/trendline_points/atr_period/take_profit_atr_mult/
    stop_loss_atr_mult.
    """
    if take_profit_atr_mult <= 0:
        raise ValueError("take_profit_atr_mult must be positive")
    if stop_loss_atr_mult <= 0:
        raise ValueError("stop_loss_atr_mult must be positive")

    df = to_dataframe(bars)
    min_bars = max(pivot_lookback * 2 + 1, atr_period) * trendline_points + 5
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(pivot_lookback={pivot_lookback}, atr_period={atr_period})"
        )

    bullish_breakout, _res, _sup, atr, _hi, _lo = trendline_signals(
        df, pivot_lookback=pivot_lookback, trendline_points=trendline_points, atr_period=atr_period
    )

    trades, equity_curve, final_equity = _simulate_scaled_breakout(
        df, bullish_breakout, atr, [take_profit_atr_mult], stop_loss_atr_mult, initial_capital
    )
    return _build_luxalgo_result(
        df, trades, equity_curve, final_equity, initial_capital, ticker, "trendline_breakout_core"
    )


# ---------------------------------------------------------------------------
# Full package: trend-line breakout + OR/EMA/liquidity/gap filters +
# up to 3 scaled ATR take-profits
# ---------------------------------------------------------------------------


def backtest_market_flow_full(
    bars: PriceHistory,
    pivot_lookback: int = 5,
    trendline_points: int = 3,
    atr_period: int = 14,
    ema_period: int = 20,
    opening_range_minutes: int = 15,
    liquidity_lookback_bars: int = 50,
    liquidity_proximity_atr_mult: float = 1.0,
    gap_filter_pct: float = 2.0,
    take_profit_1_atr_mult: float = 1.0,
    take_profit_2_atr_mult: float = 2.0,
    take_profit_3_atr_mult: float = 3.0,
    stop_loss_atr_mult: float = 1.5,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """The "full package": the same trend-line breakout as the core piece,
    but only entered when it also clears the Opening Range, a rising EMA,
    and a nearby liquidity zone, and isn't on a hard gap-down day (see
    market_flow_full_signals / the module docstring for exactly what each
    of those means and how they degrade gracefully on daily bars).

    On entry, capital is split into 3 equal legs targeting take_profit_1/
    2/3_atr_mult respectively (all sharing one ATR-based stop-loss) --
    matching the real indicator's "up to 3 ATR based take profits."

    Raises ValueError if there isn't enough data, if take_profit_1 <
    take_profit_2 < take_profit_3 isn't satisfied (the legs are meant to
    scale out further as price runs further), or for any of the other
    invalid parameter combinations market_flow_full_signals checks.
    """
    if not (0 < take_profit_1_atr_mult < take_profit_2_atr_mult < take_profit_3_atr_mult):
        raise ValueError(
            "take_profit_1_atr_mult < take_profit_2_atr_mult < take_profit_3_atr_mult "
            "must all hold (and all be positive) -- the three legs are meant to scale "
            "out at increasingly distant targets"
        )
    if stop_loss_atr_mult <= 0:
        raise ValueError("stop_loss_atr_mult must be positive")

    df = to_dataframe(bars)
    min_bars = max(pivot_lookback * 2 + 1, atr_period, ema_period, liquidity_lookback_bars) * 2 + 5
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(pivot_lookback={pivot_lookback}, atr_period={atr_period}, "
            f"ema_period={ema_period}, liquidity_lookback_bars={liquidity_lookback_bars})"
        )

    full_bullish_entry, atr = market_flow_full_signals(
        df,
        pivot_lookback=pivot_lookback,
        trendline_points=trendline_points,
        atr_period=atr_period,
        ema_period=ema_period,
        opening_range_minutes=opening_range_minutes,
        liquidity_lookback_bars=liquidity_lookback_bars,
        liquidity_proximity_atr_mult=liquidity_proximity_atr_mult,
        gap_filter_pct=gap_filter_pct,
    )

    tp_mults = [take_profit_1_atr_mult, take_profit_2_atr_mult, take_profit_3_atr_mult]
    trades, equity_curve, final_equity = _simulate_scaled_breakout(
        df, full_bullish_entry, atr, tp_mults, stop_loss_atr_mult, initial_capital
    )
    return _build_luxalgo_result(
        df, trades, equity_curve, final_equity, initial_capital, ticker, "market_flow_full"
    )
