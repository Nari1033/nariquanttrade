"""The 34 EMA High/Low Strategy: a stop-and-reverse momentum/breakout
system built from three 34-period Exponential Moving Averages (EMAs) --
one on High, one on Low, and one on Close.

Concept
-------
emaHigh and emaLow define a dynamic channel that tracks the market's
short-term fluctuations:

  - emaHigh  = EMA(ema_period) of the High price
  - emaLow   = EMA(ema_period) of the Low price
  - emaClose = EMA(ema_period) of the Close price (plotted/returned for
    context; no entry rule below uses it directly -- the source strategy
    description calls for all three lines, but only High/Low drive buy
    and sell decisions)

A breakout is genuinely a *breakout*: the close has to cross from
at-or-inside the channel to outside it, not merely still be outside it
from a prior bar -- otherwise every single day of an extended trend would
re-"signal", which is meaningless once a position is already open in that
direction.

  - Bullish breakout: close crosses up through emaHigh -> go long at that
    bar's close.
  - Bearish breakdown: close crosses down through emaLow -> go short at
    that bar's close.

Stop-and-reverse, not long/flat
--------------------------------
Unlike every long-only strategy elsewhere in this app, this one is always
either long or short once it has ever triggered (flat only before the
very first signal): a bearish breakdown while long closes the long AND
opens a short in the same bar, and a bullish breakout while short closes
the short AND opens a long in the same bar. This matches the strategy's
own framing of itself as "a trend-following and breakout system" built
around a channel, not a simple "exit to cash" crossover system -- the
channel itself is the thing being traded, and price is always on one side
of it (long above a breakout, short below a breakdown) once it has broken
out at least once.

Short-selling mechanics are simplified the same way this app treats every
other position: fully-invested notional sizing (shares = equity / entry
price), no margin interest, no borrow cost, no fees or slippage.

EMA34+DAY-1 variant
-------------------
Adds one gate on top of the exact same breakout/breakdown signals: which
*direction* is even allowed to fire today depends on the color of
*yesterday's* candle (close vs. open, not high/low):

  - Yesterday was green (close > open): only a bullish breakout can fire
    today -- no new shorts today, even if price breaks down through
    emaLow.
  - Yesterday was red (close < open): only a bearish breakdown can fire
    today -- no new longs today, even if price breaks up through emaHigh.
  - Yesterday was a doji (close == open, neither green nor red): neither
    direction is allowed to fire today -- there's no signal from
    yesterday's candle to gate on, so this variant conservatively sits
    out rather than guessing.

This only gates *new* entries/reversals -- it never forces an exit out of
an already-open position on its own; an open position still only closes
(or reverses) the normal way, via the opposite breakout condition, and
that opposite condition simply can't fire on a day it's gated off.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import ema as ema_indicator
from .models import PriceHistory


def ema_high_low_lines(
    df: pd.DataFrame, ema_period: int = 34
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ema_high, ema_low, ema_close): the three EMAs this
    strategy is built from, applied to the High, Low, and Close columns
    respectively."""
    if ema_period < 2:
        raise ValueError("ema_period must be at least 2")
    ema_high = ema_indicator(df["high"], period=ema_period)
    ema_low = ema_indicator(df["low"], period=ema_period)
    ema_close = ema_indicator(df["close"], period=ema_period)
    return ema_high, ema_low, ema_close


def ema_high_low_signals(
    df: pd.DataFrame,
    ema_period: int = 34,
    day_minus_1_filter: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (long_signal, short_signal, ema_high, ema_low, ema_close).

    long_signal / short_signal are boolean Series, True only on the bar
    the close genuinely crosses the channel (not merely sits outside it --
    see module docstring) -- a one-shot event per breakout, not a
    persistent "is price above/below" condition.

    When `day_minus_1_filter` is True (the EMA34+DAY-1 variant), each
    day's signals are additionally masked by the *previous* day's candle
    color (close vs. open): long_signal is only ever True following a
    green candle, short_signal only ever True following a red candle (see
    module docstring for the doji case).
    """
    ema_high, ema_low, ema_close = ema_high_low_lines(df, ema_period=ema_period)
    close = df["close"]
    prev_close = close.shift(1)
    prev_high = ema_high.shift(1)
    prev_low = ema_low.shift(1)

    long_signal = (close > ema_high) & (prev_close <= prev_high)
    long_signal = (long_signal & ema_high.notna() & prev_high.notna()).fillna(False)

    short_signal = (close < ema_low) & (prev_close >= prev_low)
    short_signal = (short_signal & ema_low.notna() & prev_low.notna()).fillna(False)

    if day_minus_1_filter:
        prev_open = df["open"].shift(1)
        prev_day_close = df["close"].shift(1)
        prev_green = (prev_day_close > prev_open).fillna(False)
        prev_red = (prev_day_close < prev_open).fillna(False)
        long_signal = long_signal & prev_green
        short_signal = short_signal & prev_red

    return long_signal, short_signal, ema_high, ema_low, ema_close


def ema_high_low_recent(
    bars: PriceHistory,
    ema_period: int = 34,
    day_minus_1_filter: bool = False,
    lookback_days: int = 3,
) -> bool:
    """True if a bullish breakout OR a bearish breakdown fired within the
    last `lookback_days` bars. Used by the Scanner tab -- direction isn't
    distinguished here, just "would this strategy have just entered or
    reversed a position."""
    df = to_dataframe(bars)
    min_bars = ema_period + 2
    if len(df) < min_bars:
        return False
    try:
        long_signal, short_signal, *_ = ema_high_low_signals(
            df, ema_period=ema_period, day_minus_1_filter=day_minus_1_filter
        )
    except ValueError:
        return False
    recent_long = long_signal.iloc[-lookback_days:].any()
    recent_short = short_signal.iloc[-lookback_days:].any()
    return bool(recent_long or recent_short)


def _settle_leg(
    dt,
    close: float,
    position: str,
    entry_date,
    entry_price: float,
    shares: float,
    closed_at_period_end: bool,
) -> Tuple[Trade, float]:
    """Close out whichever leg (long or short) is open, returning the
    finished Trade plus the realized equity after closing it."""
    exit_price = close
    if position == "long":
        trade_return = (exit_price - entry_price) / entry_price
        equity = shares * exit_price
    else:
        trade_return = (entry_price - exit_price) / entry_price
        equity = shares * (2 * entry_price - exit_price)
    trade = Trade(
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=dt,
        exit_price=exit_price,
        return_pct=trade_return * 100,
        is_win=trade_return > 0,
        closed_at_period_end=closed_at_period_end,
        meta={"leg": position},
    )
    return trade, equity


def backtest_ema_high_low(
    bars: PriceHistory,
    ema_period: int = 34,
    day_minus_1_filter: bool = False,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Stop-and-reverse backtest of the 34 EMA High/Low channel breakout
    (see module docstring). Flat until the first breakout/breakdown
    fires; always long or short after that, reversing directly from one
    to the other whenever the opposite condition fires. No new
    entry/reversal is taken on the final bar (no time left to manage it);
    a position still open when the data runs out is simply marked to
    market there and recorded as a trade (closed_at_period_end=True),
    same convention as every other backtester in this app. Buy-and-hold
    is computed over the full supplied period.

    Set `day_minus_1_filter=True` to run the EMA34+DAY-1 variant instead
    (see module docstring): new entries/reversals are gated by the
    previous day's candle color.

    Raises ValueError if there isn't enough data to compute the EMAs.
    """
    df = to_dataframe(bars)
    min_bars = ema_period + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest (EMA period={ema_period})"
        )

    long_signal, short_signal, ema_high, ema_low, ema_close = ema_high_low_signals(
        df, ema_period=ema_period, day_minus_1_filter=day_minus_1_filter
    )

    trades: List[Trade] = []
    position: Optional[str] = None  # "long", "short", or None (flat)
    entry_date = None
    entry_price: Optional[float] = None
    shares_held = 0.0
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)

    closes = df["close"]
    last_i = len(df) - 1

    for i in range(len(df)):
        dt = df.index[i]
        close = closes.iloc[i]

        if i != last_i:
            is_long_signal = bool(long_signal.iloc[i])
            is_short_signal = bool(short_signal.iloc[i])

            # Long signal wins precedence on the rare bar where both fire
            # at once (e.g. emaHigh/emaLow momentarily inverted) -- an
            # arbitrary but deterministic tie-break, since both can't
            # actually be acted on in the same bar.
            if position != "long" and is_long_signal:
                if position == "short":
                    trade, equity = _settle_leg(
                        dt, close, "short", entry_date, entry_price, shares_held, False
                    )
                    trades.append(trade)
                position = "long"
                entry_date = dt
                entry_price = close
                shares_held = equity / entry_price
            elif position != "short" and is_short_signal:
                if position == "long":
                    trade, equity = _settle_leg(
                        dt, close, "long", entry_date, entry_price, shares_held, False
                    )
                    trades.append(trade)
                position = "short"
                entry_date = dt
                entry_price = close
                shares_held = equity / entry_price

        if position == "long":
            equity_curve.iloc[i] = shares_held * close
        elif position == "short":
            equity_curve.iloc[i] = shares_held * (2 * entry_price - close)
        else:
            equity_curve.iloc[i] = equity

        if i == last_i and position is not None:
            trade, equity = _settle_leg(
                dt, close, position, entry_date, entry_price, shares_held, True
            )
            trades.append(trade)
            equity_curve.iloc[i] = equity
            position = None

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
        strategy_name="ema_high_low_day_minus_1" if day_minus_1_filter else "ema_high_low",
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
