"""RSI(9) + EMA(3) + WMA(21) strategy: scan + backtest.

Concept: three lines, all derived from the *same* RSI series (not from
price or volume directly, despite the line names below -- see the
Strategy registration in app/strategies.py for how this was confirmed):

  - "Strength" (black): the raw RSI(rsi_period) value, in [0, 100]. Its
    conventional 50 midline (above = strength, below = weakness) is shown
    on the chart as a reference level but does not itself drive the
    buy/sell rule below.
  - "Price" (green): a fast EMA(ema_period) smoothing of the Strength
    line -- reacts quickly to changes in RSI.
  - "Volume" (red): a slower WMA(wma_period) smoothing of the Strength
    line -- a lagging baseline the fast line crosses.

All three live on the same 0-100 RSI scale, so "above/below" is a direct
numeric comparison, not a normalized/rescaled one.

  - Buy signal: the bar where BOTH Strength and Price first move above
    Volume (a fresh transition into that state while flat -- not every
    bar the condition holds, same "fires once, not continuously"
    convention as every crossover-style strategy in this app).
  - Sell/exit signal: the mirror -- the bar where BOTH Strength and Price
    first move below Volume, while a position is open.
  - Bars where Strength and Price disagree about which side of Volume
    they're on are a "mixed" state: no new signal fires, and an existing
    position (or flat state) simply continues.

Long-only, like every other strategy in this app (no shorting), and no
stop-loss -- this is a moving-average-crossover-style strategy (mirroring
golden_cross / price_cross_sma), not a mean-reversion one, so it doesn't
carry the stop-loss safety net that engine.bollinger / engine.rsi_strategy
need for their mean-reversion entries.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import ema, rsi, wma
from .models import PriceHistory


def rsi_ema_wma_lines(
    df: pd.DataFrame,
    rsi_period: int = 9,
    ema_period: int = 3,
    wma_period: int = 21,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (strength, price_line, volume_line) -- the raw RSI, its
    fast EMA, and its slow WMA, all aligned to df's index. NaN during
    warm-up (see rsi_ema_wma_signals for the combined warm-up length)."""
    strength = rsi(df["close"], period=rsi_period)
    price_line = ema(strength, period=ema_period)
    volume_line = wma(strength, period=wma_period)
    return strength, price_line, volume_line


def rsi_ema_wma_signals(
    df: pd.DataFrame,
    rsi_period: int = 9,
    ema_period: int = 3,
    wma_period: int = 21,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (buy_signal, sell_signal, strength, price_line, volume_line).

    buy_signal: True on the bar where Strength and Price are both above
    Volume, but at least one of them wasn't on the previous bar (a fresh
    transition into the "both above" state). sell_signal is the mirror
    for "both below." Both are False during warm-up (NaN comparisons)."""
    strength, price_line, volume_line = rsi_ema_wma_lines(
        df, rsi_period=rsi_period, ema_period=ema_period, wma_period=wma_period
    )

    strength_above = (strength > volume_line).fillna(False)
    price_above = (price_line > volume_line).fillna(False)
    strength_below = (strength < volume_line).fillna(False)
    price_below = (price_line < volume_line).fillna(False)

    bullish = strength_above & price_above
    bearish = strength_below & price_below

    buy_signal = bullish & ~bullish.shift(1, fill_value=False)
    sell_signal = bearish & ~bearish.shift(1, fill_value=False)
    return buy_signal, sell_signal, strength, price_line, volume_line


def rsi_ema_wma_bullish_recent(
    bars: PriceHistory,
    rsi_period: int = 9,
    ema_period: int = 3,
    wma_period: int = 21,
    lookback_days: int = 3,
) -> bool:
    """True if a buy signal fired within the last `lookback_days` bars.
    Used by the Scanner tab."""
    df = to_dataframe(bars)
    min_bars = rsi_period + wma_period + 2
    if len(df) < min_bars:
        return False
    buy_signal, _, _, _, _ = rsi_ema_wma_signals(
        df, rsi_period=rsi_period, ema_period=ema_period, wma_period=wma_period
    )
    return bool(buy_signal.iloc[-lookback_days:].any())


def backtest_rsi_ema_wma(
    bars: PriceHistory,
    rsi_period: int = 9,
    ema_period: int = 3,
    wma_period: int = 21,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Long/flat backtest for the RSI(9)+EMA(3)+WMA(21) crossover.

    Buy at the close on a buy_signal bar (Strength & Price both freshly
    above Volume) while flat. Sell at the close on a sell_signal bar
    (Strength & Price both freshly below Volume) while holding. A
    position still open when the data runs out is marked to market on the
    final bar (exit_reason "period_end"), same convention as every other
    backtester here. Buy-and-hold is computed over the full supplied
    period.

    Raises ValueError if there isn't enough data to compute all three
    lines.
    """
    df = to_dataframe(bars)
    min_bars = rsi_period + wma_period + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(RSI period={rsi_period} + WMA period={wma_period})"
        )

    buy_signal, sell_signal, _, _, _ = rsi_ema_wma_signals(
        df, rsi_period=rsi_period, ema_period=ema_period, wma_period=wma_period
    )

    trades: List[Trade] = []
    in_position = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    shares_held = 0.0
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    closes = df["close"]
    last_i = len(df) - 1

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]
        exited_this_bar = False

        if in_position:
            hit_sell_signal = bool(sell_signal.iloc[i])
            at_period_end = i == last_i

            if hit_sell_signal or at_period_end:
                exit_price = close
                exit_reason = "bearish_crossover" if hit_sell_signal else "period_end"

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
        strategy_name="rsi9_ema3_wma21",
        start_date=df.index[0],
        end_date=df.index[-1],
        fast_window=None,
        slow_window=wma_period,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )
