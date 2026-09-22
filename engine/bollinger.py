"""Bollinger Band Mean Reversion strategy: scan + backtest.

Concept: price tends to revert to its moving average (the middle band)
after hitting an extreme high or low (the outer bands). Implemented
long-only here, like every other strategy in this app (no shorting):

  - Buy signal ("oversold"): the bar's low touches or dips below the
    lower band, but its close reverts back above it -- a lower-band
    rejection.
  - Sell/exit signal ("overbought"): the mirror at the upper band -- the
    bar's high touches or spikes above it, but the close reverts back
    below it. Used here as the profit-taking exit rather than a short
    entry, since this app doesn't model shorting.
  - Stop-loss: closes the position early if price keeps falling against
    it instead of reverting, since a strong trend can make price "hug"
    the outer band rather than bounce off it -- the risk-management rule
    the strategy spec calls out explicitly.

Bands: middle = SMA(window); upper/lower = middle +/- num_std * rolling
stdev(window). Same NaN-for-the-first-`window`-bars warm-up convention as
engine.indicators.sma.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
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
    index. buy_signal: the bar's low touched/dipped below the lower band
    but its close reverted back above it. sell_signal: the mirror at the
    upper band (the exit signal here, not a short entry)."""
    _, upper, lower = bollinger_bands(df["close"], window=window, num_std=num_std)
    buy_signal = (df["low"] <= lower) & (df["close"] > lower)
    sell_signal = (df["high"] >= upper) & (df["close"] < upper)
    return buy_signal.fillna(False), sell_signal.fillna(False)


def bollinger_oversold_recent(
    bars: PriceHistory, window: int = 20, num_std: float = 2.0, lookback_days: int = 3
) -> bool:
    """True if a lower-band-rejection buy signal fired within the last
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
    stop_loss_pct: float = 10.0,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Long/flat mean-reversion backtest: buy at the close on a
    lower-band rejection; sell at the close on an upper-band rejection,
    *or* stop out when the position has drawn down `stop_loss_pct`% from
    its entry price (checked against the bar's intrabar low), whichever
    comes first. A position still open when the data runs out is marked
    to market on the final bar, same convention as the other backtesters
    here. Buy-and-hold is computed over the full supplied period.

    Raises ValueError if there isn't enough data to compute the bands.
    """
    df = to_dataframe(bars)
    min_bars = window + 2
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest (band window={window})"
        )

    buy_signal, sell_signal = bollinger_signals(df, window=window, num_std=num_std)

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
