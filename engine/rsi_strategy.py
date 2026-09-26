"""RSI(14) Momentum Breakout strategy: scan + backtest.

Concept: this is a MOMENTUM strategy built on the Relative Strength Index,
not the textbook RSI mean-reversion strategy (buy under 30, sell over 70).
Here, RSI rising into strength is read as the entry signal rather than the
exit, and only a further, more extreme reading is taken as profit-taking:

  - Buy signal: RSI(period) rises up through the buy threshold (touches or
    crosses 70 from below) -- momentum accelerating into strength.
  - Sell/exit signal: RSI(period) rises further and touches/crosses the
    (higher) sell threshold (90) -- an extreme reading taken as the
    profit-taking exit.
  - Stop-loss: closes the position early if price falls hard against it
    instead of RSI ever reaching the sell threshold -- RSI can stall or
    roll over well below 90 while price keeps sliding, so this is the same
    risk-management fallback engine.bollinger uses (stop_loss_pct).

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
    sell_threshold: float = 90.0,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (buy_signal, sell_signal) boolean Series aligned to df's
    index. buy_signal: RSI(period) rises up through buy_threshold (below
    it on the previous bar, at or above it now). sell_signal: the same
    kind of rising crossing, but at sell_threshold. Bars where RSI is
    still NaN (warm-up) never register a signal.

    Raises ValueError if sell_threshold <= buy_threshold, since the whole
    point of this strategy is that the exit level sits above the entry
    level -- ride momentum from "just broke out" to "extremely
    overbought," not the other way around.
    """
    if sell_threshold <= buy_threshold:
        raise ValueError("sell_threshold must be greater than buy_threshold")

    r = rsi(df["close"], period=period)
    prev = r.shift(1)
    valid = r.notna() & prev.notna()

    buy_signal = valid & (r >= buy_threshold) & (prev < buy_threshold)
    sell_signal = valid & (r >= sell_threshold) & (prev < sell_threshold)
    return buy_signal.fillna(False), sell_signal.fillna(False)


def rsi_breakout_recent(
    bars: PriceHistory,
    period: int = 14,
    buy_threshold: float = 70.0,
    sell_threshold: float = 90.0,
    lookback_days: int = 3,
) -> bool:
    """True if a buy signal (RSI rising up through buy_threshold) fired
    within the last `lookback_days` bars. Used by the Scanner tab."""
    df = to_dataframe(bars)
    if len(df) < period + 1:
        return False
    buy_signal, _ = rsi_signals(
        df, period=period, buy_threshold=buy_threshold, sell_threshold=sell_threshold
    )
    return bool(buy_signal.iloc[-lookback_days:].any())


def backtest_rsi_momentum(
    bars: PriceHistory,
    period: int = 14,
    buy_threshold: float = 70.0,
    sell_threshold: float = 90.0,
    stop_loss_pct: float = 15.0,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Long/flat momentum backtest: buy at the close when RSI(period)
    rises up through buy_threshold; sell at the close when RSI rises up
    through sell_threshold, *or* stop out when the position has drawn down
    stop_loss_pct% from its entry price (checked against the bar's
    intrabar low), whichever comes first. A position still open when the
    data runs out is marked to market on the final bar, same convention as
    every other backtester here. Buy-and-hold is computed over the full
    supplied period.

    Raises ValueError if there isn't enough data to compute RSI, or if
    sell_threshold <= buy_threshold (see rsi_signals).
    """
    df = to_dataframe(bars)
    min_bars = period + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest (RSI period={period})"
        )

    buy_signal, sell_signal = rsi_signals(
        df, period=period, buy_threshold=buy_threshold, sell_threshold=sell_threshold
    )

    trades: List[Trade] = []
    in_position = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    shares_held = 0.0
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    closes = df["close"]
    lows = df["low"]
    last_i = len(df) - 1

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]
        exited_this_bar = False

        if in_position:
            stop_price = entry_price * (1 - stop_loss_pct / 100.0)
            hit_stop = bool(lows.iloc[i] <= stop_price)
            hit_sell_signal = bool(sell_signal.iloc[i])
            at_period_end = i == last_i

            if hit_stop or hit_sell_signal or at_period_end:
                if hit_stop:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif hit_sell_signal:
                    exit_price, exit_reason = close, "rsi_target"
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
                exited_this_bar = True

        if not in_position and not exited_this_bar and bool(buy_signal.iloc[i]) and i != last_i:
            in_position = True
            entry_date = dt
            entry_price = close
            shares_held = equity / entry_price

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
