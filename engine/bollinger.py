"""Bollinger Band Mean Reversion strategy: scan + backtest.

Concept: price tends to revert to its moving average (the middle band)
after hitting an extreme high or low (the outer bands). This is a
two-bar reversal pattern at each band, not a same-bar wick-touch:

  - Long entry ("oversold"): a bar closes BELOW the lower band (the
    "signal candle" -- the oversold extreme), followed by a reversal bar
    that closes back ABOVE the lower band (back inside the bands). The
    position is entered at the reversal bar's close.
  - Short entry ("overbought"): the mirror at the upper band -- a bar
    closes above the upper band, followed by a reversal bar closing back
    below it. This app is long-only (no shorting, like every other
    strategy here), so this condition is used as a profit-taking/exit
    trigger for an existing long position instead of opening a short.
  - Profit target: the middle band (the window-period SMA). Checked
    against the bar's intrabar high (a limit-style target), and the
    position exits AT the middle band level, not the bar's close.
  - Stop-loss: `atr_multiple` ATR(atr_period)s beyond the *signal
    candle's* low (the oversold bar that triggered entry, one bar before
    the actual entry bar) -- not the entry bar's own low, and not a flat
    percentage. Computed once at entry and held fixed for the trade,
    checked against the bar's intrabar low, and the position exits AT
    the stop price.
  - If more than one exit condition is true on the same bar, stop-loss
    takes priority (risk management first), then the profit target,
    then the band-rejection exit.

Bands: middle = SMA(window); upper/lower = middle +/- num_std * rolling
stdev(window). Same NaN-for-the-first-`window`-bars warm-up convention as
engine.indicators.sma. ATR uses engine.indicators.atr (Wilder's
smoothing), same NaN-for-the-first-`atr_period`-bars warm-up.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import atr as atr_indicator
from .models import PriceHistory


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (middle, upper, lower) band Series aligned to `close`'s
    index. NaN for the first `window` bars."""
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def bollinger_signals(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """Returns (buy_signal, sell_signal) boolean Series aligned to df's
    index -- the two-bar reversal-at-the-band pattern (see module
    docstring).

    buy_signal: True on the bar whose close reverts back ABOVE the lower
    band, when the PREVIOUS bar closed below it (the oversold "signal
    candle"). sell_signal is the upper-band mirror -- used here as a
    profit-taking/exit trigger for an existing long position, since this
    app doesn't model shorting. Both False during warm-up.
    """
    _, upper, lower = bollinger_bands(df["close"], window=window, num_std=num_std)
    close = df["close"]

    closed_below_lower = (close < lower).fillna(False)
    closed_above_upper = (close > upper).fillna(False)
    reverted_above_lower = (close > lower).fillna(False)
    reverted_below_upper = (close < upper).fillna(False)

    buy_signal = closed_below_lower.shift(1, fill_value=False) & reverted_above_lower
    sell_signal = closed_above_upper.shift(1, fill_value=False) & reverted_below_upper
    return buy_signal, sell_signal


def bollinger_oversold_recent(
    bars: PriceHistory, window: int = 20, num_std: float = 2.0, lookback_days: int = 3
) -> bool:
    """True if a lower-band reversal buy signal fired within the last
    `lookback_days` bars. Used by the Scanner tab."""
    df = to_dataframe(bars)
    if len(df) < window + 1:
        return False
    buy_signal, _ = bollinger_signals(df, window=window, num_std=num_std)
    return bool(buy_signal.iloc[-lookback_days:].any())


def backtest_bollinger_mean_reversion(
    bars: PriceHistory,
    window: int = 20,
    num_std: float = 2.0,
    atr_period: int = 14,
    atr_multiple: float = 1.0,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Long/flat mean-reversion backtest.

    Entry: a two-bar reversal at the lower band -- the prior bar closed
    below the lower band (the oversold "signal candle"), and this bar's
    close reverts back above it. Enters at this bar's close.

    Exit, whichever comes first:
      - Stop-loss: `atr_multiple` ATR(atr_period)s below the *signal
        candle's* low (not the entry bar's own low), fixed at entry and
        checked against each bar's intrabar low. Exits at the stop price.
      - Profit target: price reaches the middle band (the window-period
        SMA), checked against each bar's intrabar high. Exits at the
        middle band level.
      - Band-rejection exit: the mirror two-bar reversal at the upper
        band (prior bar closed above it, this bar's close reverts back
        below). This app is long-only, so this ends the position instead
        of opening a short. Exits at this bar's close.
      - Period end: a position still open when the data runs out is
        marked to market on the final bar, same convention as every
        other backtester here.
    When more than one of these triggers on the same bar, stop-loss takes
    priority, then the profit target, then the band-rejection exit.

    Buy-and-hold is computed over the full supplied period.

    Raises ValueError if there isn't enough data to compute the bands
    and ATR.
    """
    df = to_dataframe(bars)
    min_bars = max(window, atr_period) + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(band window={window}, ATR period={atr_period})"
        )

    middle, _, _ = bollinger_bands(df["close"], window=window, num_std=num_std)
    buy_signal, sell_signal = bollinger_signals(df, window=window, num_std=num_std)
    atr_series = atr_indicator(df, period=atr_period)

    trades: List[Trade] = []
    in_position = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    shares_held = 0.0
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    last_i = len(df) - 1

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]
        exited_this_bar = False

        if in_position:
            mid_today = middle.iloc[i]
            hit_stop = bool(lows.iloc[i] <= stop_price)
            hit_target = bool(pd.notna(mid_today) and highs.iloc[i] >= mid_today)
            hit_sell_signal = bool(sell_signal.iloc[i])
            at_period_end = i == last_i

            if hit_stop or hit_target or hit_sell_signal or at_period_end:
                if hit_stop:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif hit_target:
                    exit_price, exit_reason = float(mid_today), "profit_target"
                elif hit_sell_signal:
                    exit_price, exit_reason = close, "band_rejection"
                else:
                    exit_price, exit_reason = close, "period_end"

                trade_return = (exit_price - entry_price) / entry_price
                equity = shares_held * exit_price
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=dt,
                        exit_price=exit_price,
                        return_pct=trade_return * 100,
                        is_win=trade_return > 0,
                        closed_at_period_end=(exit_reason == "period_end"),
                        meta={"exit_reason": exit_reason, "stop_price": round(stop_price, 2)},
                    )
                )
                in_position = False
                shares_held = 0.0
                entry_date = None
                entry_price = None
                stop_price = None
                exited_this_bar = True

        if (
            not in_position
            and not exited_this_bar
            and i >= 1
            and bool(buy_signal.iloc[i])
            and i != last_i
        ):
            in_position = True
            entry_date = dt
            entry_price = close
            shares_held = equity / entry_price
            # The *signal candle* is the previous bar (the one that
            # closed below the lower band) -- its low is the "extreme
            # wick" the stop sits beyond, not the entry/reversal bar's.
            signal_candle_low = lows.iloc[i - 1]
            signal_candle_atr = atr_series.iloc[i - 1]
            if pd.isna(signal_candle_atr):
                signal_candle_atr = 0.0
            stop_price = signal_candle_low - atr_multiple * signal_candle_atr

        equity_curve.iloc[i] = shares_held * close if in_position else equity

    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate_pct = (wins / total_trades * 100) if total_trades else 0.0
    strategy_return_pct = (equity_curve.iloc[-1] - initial_capital) / initial_capital * 100

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    buy_hold_return_pct = (last_close - first_close) / first_close * 100
    buy_hold_curve = (df["close"] / first_close) * initial_capital

    return BacktestResult(
        ticker=ticker,
        strategy_name="bollinger_mean_reversion",
        start_date=df.index[0],
        end_date=df.index[-1],
        fast_window=None,
        slow_window=window,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )
