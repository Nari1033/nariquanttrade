"""RSI(14) Momentum Breakout strategy: scan + backtest.

Concept: this is a MOMENTUM strategy built on the Relative Strength Index,
not the textbook RSI mean-reversion strategy (buy under 30, sell over 70).
Here, RSI rising into strength is read as the entry signal rather than the
exit:

  - Buy signal: RSI(period) rises up through the buy threshold (touches or
    crosses 70 from below) -- momentum accelerating into strength.
  - Sell/exit signal: a "faded momentum" exit. Once RSI has risen to/above
    the overbought threshold (80) at some point since entering the
    position, sell as soon as RSI falls back to/below the exit threshold
    (60) -- momentum peaked and is now rolling over.

    (An earlier version of this strategy sold only when RSI itself
    crossed an extreme fixed level like 90. Empirically, on real daily
    data that almost never happens -- across several large-cap names over
    a 5-year window, RSI crossed 70 dozens of times each but never once
    crossed 90 -- so positions just sat open for years waiting for an
    exit that wasn't coming. The overbought-then-pullback design below
    fires far more often, since 80 is a routine overbought reading and
    60 is a routine pullback level.)

  - Stop-loss: closes the position early if price falls hard against it
    before the rollover exit condition is ever met -- RSI can stay
    elevated (or the position can just be wrong) while price keeps
    sliding, so this is the same risk-management fallback
    engine.bollinger uses (stop_loss_pct).

Long-only, like every other strategy in this app (no shorting). See
engine.indicators.rsi for the RSI calculation itself (Wilder's smoothing).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import rsi
from .models import PriceHistory


def rsi_signals(
    df: pd.DataFrame,
    period: int = 14,
    buy_threshold: float = 70.0,
    overbought_threshold: float = 80.0,
    exit_threshold: float = 60.0,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (buy_signal, rsi_series).

    buy_signal: a boolean Series aligned to df's index, True on the bar
    where RSI(period) rises up through buy_threshold (below it on the
    previous bar, at or above it now). Bars where RSI is still NaN
    (warm-up) never register a signal.

    rsi_series: the raw RSI(period) values themselves, for the caller
    (the backtest simulation loop) to evaluate the overbought-then-pullback
    exit condition, which is inherently stateful per open position --
    "has RSI reached overbought_threshold since *this* entry" isn't
    something a single context-free boolean series can express on its
    own, the way a one-shot threshold crossing can.

    Raises ValueError if overbought_threshold <= buy_threshold (the whole
    point is riding momentum from "just broke out" to "overbought," not
    the other way around), or if exit_threshold >= overbought_threshold
    (the pullback level has to sit below the overbought level, or "comes
    back down to it" doesn't mean anything).
    """
    if overbought_threshold <= buy_threshold:
        raise ValueError("overbought_threshold must be greater than buy_threshold")
    if exit_threshold >= overbought_threshold:
        raise ValueError("exit_threshold must be less than overbought_threshold")

    r = rsi(df["close"], period=period)
    prev = r.shift(1)
    valid = r.notna() & prev.notna()

    buy_signal = valid & (r >= buy_threshold) & (prev < buy_threshold)
    return buy_signal.fillna(False), r


def rsi_breakout_recent(
    bars: PriceHistory,
    period: int = 14,
    buy_threshold: float = 70.0,
    overbought_threshold: float = 80.0,
    exit_threshold: float = 60.0,
    lookback_days: int = 3,
) -> bool:
    """True if a buy signal (RSI rising up through buy_threshold) fired
    within the last `lookback_days` bars. Used by the criteria scan."""
    df = to_dataframe(bars)
    if len(df) < period + 1:
        return False
    buy_signal, _ = rsi_signals(
        df,
        period=period,
        buy_threshold=buy_threshold,
        overbought_threshold=overbought_threshold,
        exit_threshold=exit_threshold,
    )
    return bool(buy_signal.iloc[-lookback_days:].any())


def backtest_rsi_momentum(
    bars: PriceHistory,
    period: int = 14,
    buy_threshold: float = 70.0,
    overbought_threshold: float = 80.0,
    exit_threshold: float = 60.0,
    stop_loss_pct: float = 15.0,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Long/flat momentum backtest.

    Buy at the close when RSI(period) rises up through buy_threshold.
    While the position is open, watch for RSI to reach overbought_threshold
    at any point (it may take a while, or never happen); once it has, sell
    at the close of the first subsequent bar where RSI has fallen back to
    or below exit_threshold ("faded momentum" exit, exit_reason
    "rsi_rollover"). Independently of that, stop out early if the position
    draws down stop_loss_pct% from its entry price (checked against the
    bar's intrabar low, exit_reason "stop_loss") -- checked first each bar,
    since a hard stop should win over a same-day RSI reading either way. A
    position still open when the data runs out is marked to market on the
    final bar (exit_reason "period_end"), same convention as every other
    backtester here. Buy-and-hold is computed over the full supplied
    period.

    Raises ValueError if there isn't enough data to compute RSI, or for
    the threshold ordering (see rsi_signals).
    """
    df = to_dataframe(bars)
    min_bars = period + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest (RSI period={period})"
        )

    buy_signal, r = rsi_signals(
        df,
        period=period,
        buy_threshold=buy_threshold,
        overbought_threshold=overbought_threshold,
        exit_threshold=exit_threshold,
    )

    trades: List[Trade] = []
    in_position = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    seen_overbought = False
    shares_held = 0.0
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    closes = df["close"]
    lows = df["low"]
    last_i = len(df) - 1

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]
        rsi_val = r.iloc[i]
        exited_this_bar = False

        if in_position:
            if not seen_overbought and pd.notna(rsi_val) and rsi_val >= overbought_threshold:
                seen_overbought = True

            stop_price = entry_price * (1 - stop_loss_pct / 100.0)
            hit_stop = bool(lows.iloc[i] <= stop_price)
            hit_rollover = bool(
                seen_overbought and pd.notna(rsi_val) and rsi_val <= exit_threshold
            )
            at_period_end = i == last_i

            if hit_stop or hit_rollover or at_period_end:
                if hit_stop:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif hit_rollover:
                    exit_price, exit_reason = close, "rsi_rollover"
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
                        meta={"exit_reason": exit_reason},
                    )
                )
                in_position = False
                shares_held = 0.0
                entry_date = None
                entry_price = None
                seen_overbought = False
                exited_this_bar = True

        if not in_position and not exited_this_bar and bool(buy_signal.iloc[i]) and i != last_i:
            in_position = True
            entry_date = dt
            entry_price = close
            shares_held = equity / entry_price
            seen_overbought = False

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
        strategy_name="rsi_momentum",
        start_date=df.index[0],
        end_date=df.index[-1],
        fast_window=None,
        slow_window=period,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )
