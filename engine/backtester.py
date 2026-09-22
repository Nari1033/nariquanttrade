"""Crossover backtester(s).

Every strategy here follows the same long/flat shape: go long (100% of
capital) at the close on the day a "fast" line crosses above a "slow" line;
exit to cash at the close on the day it crosses back below. What differs
per strategy is just which two series play fast/slow:

  - backtest_sma_crossover: SMA-fast vs SMA-slow (the "Golden Cross" /
    "Death Cross" strategy).
  - backtest_price_sma_crossover: price itself vs its own SMA-N (the
    "price crosses its SMA" strategy).

Both share one simulation core (_simulate_crossover) so new crossover-style
strategies are cheap to add -- see app/strategies.py for how the GUI turns
each one into its own Scanner + Backtest sub-tab.

Every strategy is compared against a simple buy-and-hold of the same
instrument over the same period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .data_utils import to_dataframe
from .indicators import add_sma_columns
from .models import PriceHistory
from .scanner import crossover_series


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    return_pct: float
    is_win: bool
    closed_at_period_end: bool = False
    """True if this trade was still open when the data ran out and was
    marked-to-market on the final bar rather than closed by a real
    crossover-back signal."""
    meta: dict = field(default_factory=dict)
    """Strategy-specific extras (e.g. an options strategy's exit_reason,
    short_strike, long_strike). Empty for the plain crossover strategies.
    Keeps Trade reusable across strategy types without growing new named
    fields for every strategy that gets added."""


@dataclass
class BacktestResult:
    ticker: Optional[str]
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    fast_window: Optional[int]
    """SMA window used as the 'fast' line, or None when the fast line is
    the raw price itself (e.g. backtest_price_sma_crossover)."""
    slow_window: Optional[int]
    initial_capital: float
    strategy_return_pct: float
    buy_hold_return_pct: float
    total_trades: int
    win_rate_pct: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    buy_hold_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "strategy_name": self.strategy_name,
            "start_date": str(self.start_date.date()) if self.start_date is not None else None,
            "end_date": str(self.end_date.date()) if self.end_date is not None else None,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "strategy_return_pct": round(self.strategy_return_pct, 2),
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
        }


def annualized_return_pct(total_return_pct: float, days: int) -> Optional[float]:
    """Convert a total percentage return earned over `days` calendar days
    into a compound annual growth rate (CAGR), in percent: what that same
    rate of return would compound to over a full year.

    Returns None if `days` isn't positive, or if the total return implies a
    non-positive ending value (e.g. a -100%+ return) -- annualizing that is
    mathematically undefined (a fractional power of a non-positive number).
    Short windows can produce extreme-looking annualized numbers; that's an
    inherent property of extrapolating a short period out to a full year,
    not a bug.
    """
    if days <= 0:
        return None
    growth = 1.0 + total_return_pct / 100.0
    if growth <= 0:
        return None
    return (growth ** (365.0 / days) - 1.0) * 100.0


def _simulate_crossover(df: pd.DataFrame, fast: pd.Series, slow: pd.Series, initial_capital: float):
    """Shared long/flat simulation: buy at the close when `fast` crosses
    above `slow`, sell at the close when it crosses back below. Returns
    (trades, equity_curve, final_equity)."""
    cross = crossover_series(fast, slow)

    trades: List[Trade] = []
    in_position = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    shares_held = 0.0
    equity = initial_capital

    equity_curve = pd.Series(index=df.index, dtype=float)

    for dt, row in df.iterrows():
        signal = cross.loc[dt]
        close = row["close"]

        if not in_position and signal == 1:
            in_position = True
            entry_date = dt
            entry_price = close
            shares_held = equity / entry_price
        elif in_position and signal == -1:
            exit_price = close
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
                )
            )
            in_position = False
            shares_held = 0.0
            entry_date = None
            entry_price = None

        equity_curve.loc[dt] = shares_held * close if in_position else equity

    if in_position:
        last_dt = df.index[-1]
        last_close = df["close"].iloc[-1]
        trade_return = (last_close - entry_price) / entry_price
        equity = shares_held * last_close
        trades.append(
            Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=last_dt,
                exit_price=last_close,
                return_pct=trade_return * 100,
                is_win=trade_return > 0,
                closed_at_period_end=True,
            )
        )
        equity_curve.loc[last_dt] = equity

    return trades, equity_curve, equity


def _build_result(
    df: pd.DataFrame,
    trades: List[Trade],
    equity_curve: pd.Series,
    final_equity: float,
    initial_capital: float,
    ticker: Optional[str],
    strategy_name: str,
    fast_window: Optional[int],
    slow_window: Optional[int],
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
        fast_window=fast_window,
        slow_window=slow_window,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )


def backtest_sma_crossover(
    bars: PriceHistory,
    fast_window: int = 50,
    slow_window: int = 200,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Run the Golden Cross / Death Cross backtest over `bars`.

    - Buys at the close on a golden-cross bar (SMA-fast crosses above SMA-slow).
    - Sells at the close on a death-cross bar (SMA-fast crosses below SMA-slow).
    - Fully invested while in a position; in cash otherwise (no leverage,
      no shorting, no fees/slippage modeled).
    - If a position is still open when the data ends, it is marked to market
      on the final bar so the return is fully realized and counted as a trade.
    - Buy-and-hold return is computed over the *entire* supplied period
      (first close to last close), for an apples-to-apples comparison against
      the strategy over the same timeframe.

    Raises ValueError if fewer than 2 bars are supplied.
    """
    df = add_sma_columns(to_dataframe(bars), windows=(fast_window, slow_window))
    if len(df) < 2:
        raise ValueError("Need at least 2 price bars to run a backtest")

    trades, equity_curve, final_equity = _simulate_crossover(
        df, df[f"sma_{fast_window}"], df[f"sma_{slow_window}"], initial_capital
    )
    return _build_result(
        df, trades, equity_curve, final_equity, initial_capital, ticker,
        "golden_cross", fast_window, slow_window,
    )


def backtest_price_sma_crossover(
    bars: PriceHistory,
    sma_window: int = 50,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    """Run the "price crosses its own SMA" backtest over `bars`.

    - Buys at the close on the bar where price crosses above its SMA-N.
    - Sells at the close on the bar where price crosses back below it.
    - Same fully-invested/in-cash, mark-to-market-at-period-end, and
      buy-and-hold-over-the-full-period conventions as backtest_sma_crossover
      (see its docstring) -- just with price standing in for the fast line
      and a single SMA-N as the slow line.

    Raises ValueError if fewer than 2 bars are supplied.
    """
    df = add_sma_columns(to_dataframe(bars), windows=(sma_window,))
    if len(df) < 2:
        raise ValueError("Need at least 2 price bars to run a backtest")

    trades, equity_curve, final_equity = _simulate_crossover(
        df, df["close"], df[f"sma_{sma_window}"], initial_capital
    )
    return _build_result(
        df, trades, equity_curve, final_equity, initial_capital, ticker,
        "price_cross_sma", None, sma_window,
    )
