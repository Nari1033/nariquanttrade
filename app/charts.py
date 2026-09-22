"""Matplotlib chart builders for the Streamlit app.

Kept separate from app.py (and free of any `import streamlit`) so these can
be unit-tested / sanity-checked without a Streamlit runtime.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit's st.pyplot doesn't need a GUI backend
import matplotlib.pyplot as plt
import pandas as pd

from engine.backtester import Trade


def plot_price_with_signals(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    trades: Optional[List[Trade]] = None,
    title: str = "",
    upper_band: Optional[pd.Series] = None,
    lower_band: Optional[pd.Series] = None,
):
    """Price + both SMAs, with upward triangles at trade entries and
    downward triangles at trade exits. `upper_band`/`lower_band` are
    optional -- when a strategy is built around a band concept (e.g.
    Bollinger Bands), passing both draws them as a shaded envelope around
    price so the trade markers have their trigger context visible, not
    just a bare SMA line."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df.index, df["close"], label="Close", color="#1f77b4", linewidth=1.2)
    fast_col = f"sma_{fast_window}"
    slow_col = f"sma_{slow_window}"
    if fast_col in df.columns:
        ax.plot(df.index, df[fast_col], label=f"SMA {fast_window}", color="#ff7f0e", linewidth=1.0)
    if slow_col in df.columns:
        ax.plot(df.index, df[slow_col], label=f"SMA {slow_window}", color="#9467bd", linewidth=1.0)

    if upper_band is not None and lower_band is not None:
        ax.plot(df.index, upper_band, label="Upper band", color="#9467bd", linewidth=0.8, linestyle="--")
        ax.plot(df.index, lower_band, label="Lower band", color="#9467bd", linewidth=0.8, linestyle="--")
        ax.fill_between(df.index, lower_band, upper_band, color="#9467bd", alpha=0.08)

    if trades:
        # Marker y-position is always the underlying's close price on that
        # date -- NOT trade.entry_price/exit_price. For the SMA crossover
        # strategies those happen to be the same thing (entry_price IS the
        # close), but for an options strategy entry_price/exit_price are a
        # credit/debit dollar amount on a totally different scale than the
        # stock price, so plotting those directly would put markers miles
        # off the price line (or off the chart entirely).
        entry_dates = [t.entry_date for t in trades if t.entry_date in df.index]
        entry_prices = [df.loc[t.entry_date, "close"] for t in trades if t.entry_date in df.index]
        exit_dates = [t.exit_date for t in trades if t.exit_date in df.index]
        exit_prices = [df.loc[t.exit_date, "close"] for t in trades if t.exit_date in df.index]
        ax.scatter(
            entry_dates, entry_prices, marker="^", color="green", s=90, zorder=5, label="Buy"
        )
        ax.scatter(
            exit_dates, exit_prices, marker="v", color="red", s=90, zorder=5, label="Sell"
        )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_equity_curves(
    strategy_curve: pd.Series, buy_hold_curve: pd.Series, title: str = "Strategy vs Buy & Hold"
):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(strategy_curve.index, strategy_curve.values, label="Strategy", color="#2ca02c")
    ax.plot(buy_hold_curve.index, buy_hold_curve.values, label="Buy & Hold", color="#7f7f7f")
    ax.set_title(title)
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
