"""The Modified Wheel Strategy: the plain Wheel (see engine.wheel's module
docstring for the full three-step cycle and bookkeeping model) plus one
addition -- a down-market filter that skips opening *new* cash-secured
puts while the market looks like it's in an ongoing decline.

Why a filter, and why only on puts
-----------------------------------
The plain Wheel deliberately has no trend filter: you're meant to be happy
to own the stock through a dip. But "happy to own it eventually" doesn't
mean "indifferent to when you start selling puts into a falling market" --
selling a put right as a decline is accelerating both caps the credit
you're paid relative to the risk taken on, and raises the odds of being
assigned stock at a strike that still has further to fall before the
later covered-calls leg can dig back out. So this strategy only gates the
PUT-selling step (opening a new short put while flat). Once assigned,
covered-call selling against the shares you already own is left exactly
as in the plain Wheel -- by then the shares are owned regardless of
market direction, and selling calls against them is how the cycle digs
back toward even, not a new risk decision.

The down-market filter, and where it comes from
------------------------------------------------
Built entirely from two indicators this app already has elsewhere
(engine.indicators.sma, used by the crossover strategies and Bollinger's
middle band; engine.indicators.rsi, used by the RSI strategies) --  not a
new indicator invented for this strategy. A day is flagged "bearish" when
BOTH hold:

  - close is below its own `trend_sma_window`-day SMA (a long-term
    downtrend context -- same idea as the "Golden/Death Cross" and bull
    put spread/cash-secured put strategies' trend filters elsewhere in
    this app, just inverted: those gate entries to an *uptrend*, this
    gates them away from a confirmed *downtrend*).
  - RSI(`rsi_period`) is below `rsi_threshold` (short-term weakness is
    still active, not already turned back up).

This combination was chosen empirically, not assumed: tested against 10
years of real SPY daily data (2016-2026), candidate signals built from
this app's existing indicators (SMA trend, RSI, Bollinger %B, ATR
volatility expansion, trend-line breakdown, raw momentum, and
combinations of these) were each measured by how often the 10/20/40
trading-day return *following* a flagged day was actually negative.
`close below SMA(200)` AND `RSI(14) below 40` was the most consistently
selective of the combinations built purely from this app's existing
indicator functions -- it flagged ~5% of days while catching
substantially more of the actual subsequent declines than the plain SMA
trend filter alone (which mostly just lags into "buy-the-dip" days on a
long-run uptrending instrument like SPY, since a plain trend filter fires
on the way down just as often as near a bottom). It is not a perfect
predictor -- no such thing exists for daily returns -- just the best
simple, explainable, already-available-indicators combination found.

`trend_sma_window`, `rsi_period`, and `rsi_threshold` are all left as
tunable parameters (not hardcoded), the same way this app exposes every
other strategy's indicator settings, so they can be swept per-instrument
on the Parameter Sweep page rather than assumed to be universal constants.

Everything else (bookkeeping, option pricing, DTE/day-of-month handling,
simplifications) is identical to engine.wheel -- see that module's
docstring for the full list of caveats that apply here too.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .backtester import BacktestResult, Trade
from .calendar_filters import day_of_month_ok
from .data_utils import to_dataframe
from .indicators import rsi as rsi_indicator
from .indicators import sma as sma_indicator
from .models import PriceHistory
from .options_pricing import (
    black_scholes_price,
    realized_volatility,
    strike_for_delta,
    strike_for_put_delta_magnitude,
)

CONTRACT_MULTIPLIER = 100.0


def bearish_filter_signal(
    close: pd.Series,
    trend_sma_window: int = 200,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
) -> pd.Series:
    """True on bars flagged as an ongoing down-market: close below its own
    `trend_sma_window`-day SMA AND RSI(`rsi_period`) below `rsi_threshold`.
    False (never blocks) during warm-up, before either indicator has
    enough history -- same "don't block on a warm-up artifact" convention
    used elsewhere in this app (e.g. market_flow_full_signals' liquidity
    filter)."""
    if trend_sma_window < 2:
        raise ValueError("trend_sma_window must be at least 2")
    if rsi_period < 1:
        raise ValueError("rsi_period must be a positive integer")
    trend_sma = sma_indicator(close, trend_sma_window)
    rsi_series = rsi_indicator(close, rsi_period)
    below_trend = (close < trend_sma).fillna(False)
    rsi_weak = (rsi_series < rsi_threshold).fillna(False)
    return below_trend & rsi_weak


def backtest_modified_wheel_strategy(
    bars: PriceHistory,
    put_delta: float = 0.20,
    call_delta: float = 0.20,
    dte_days: int = 30,
    entry_day_of_month: int = 0,
    vol_window: int = 20,
    trend_sma_window: int = 200,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
    risk_free_rate_pct: float = 4.5,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    min_bars = max(vol_window, trend_sma_window, rsi_period) + 5
    df = to_dataframe(bars)
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(vol_window={vol_window}, trend_sma_window={trend_sma_window}, rsi_period={rsi_period})"
        )

    sigma_series = realized_volatility(df["close"], window=vol_window)
    bearish = bearish_filter_signal(
        df["close"], trend_sma_window=trend_sma_window, rsi_period=rsi_period, rsi_threshold=rsi_threshold
    )
    r = risk_free_rate_pct / 100.0

    trades: List[Trade] = []
    cash = initial_capital
    shares_held = 0  # 0 or 100 -- one contract's worth, same "always 1 contract" convention as the rest of this app
    equity_curve = pd.Series(index=df.index, dtype=float)

    # The single open short leg (put or call), or None. dict: type,
    # entry_date, expiration_date, strike, credit.
    open_leg = None

    dates = df.index.to_list()
    last_i = len(dates) - 1

    for i, dt in enumerate(dates):
        close = float(df["close"].loc[dt])
        sigma = sigma_series.loc[dt]
        have_vol = pd.notna(sigma) and sigma > 0

        # --- manage an existing leg: reprice, check expiration, maybe settle ---
        if open_leg is not None:
            leg_type = open_leg["type"]
            strike = open_leg["strike"]
            credit = open_leg["credit"]
            days_left = (open_leg["expiration_date"] - dt).days
            T_remaining = days_left / 365.0

            if T_remaining <= 0 or not have_vol:
                cost_to_close = (
                    max(strike - close, 0.0) if leg_type == "put" else max(close - strike, 0.0)
                )
            else:
                cost_to_close = black_scholes_price(leg_type, close, strike, T_remaining, r, sigma)

            settle_reason = "expiration" if T_remaining <= 0 else None
            if settle_reason is None and i == last_i:
                settle_reason = "period_end"

            if settle_reason is not None:
                if settle_reason == "expiration":
                    if leg_type == "put":
                        exit_reason = "assigned" if close < strike else "expired_worthless"
                    else:
                        exit_reason = "called_away" if close >= strike else "expired_worthless"
                else:
                    exit_reason = "period_end"

                return_pct = (credit - cost_to_close) / credit * 100.0
                trades.append(
                    Trade(
                        entry_date=open_leg["entry_date"],
                        entry_price=credit,
                        exit_date=dt,
                        exit_price=cost_to_close,
                        return_pct=return_pct,
                        is_win=return_pct > 0,
                        closed_at_period_end=(exit_reason == "period_end"),
                        meta={
                            "exit_reason": exit_reason,
                            "short_strike": round(strike, 2),
                            "leg": leg_type,
                            "entry_day_of_month": entry_day_of_month,
                        },
                    )
                )

                if exit_reason == "assigned":
                    cash -= strike * CONTRACT_MULTIPLIER
                    shares_held = 100
                    open_leg = None
                elif exit_reason == "called_away":
                    cash += strike * CONTRACT_MULTIPLIER
                    shares_held = 0
                    open_leg = None
                elif exit_reason == "expired_worthless":
                    open_leg = None
                # period_end: leave open_leg set so today's mark below
                # still reflects it; nothing was really settled.

        # --- consider opening a new leg (not on the final bar -- no time
        # left to manage it). calendar_ok is the same day-of-month timing
        # gate the plain Wheel has; bearish_ok is the new down-market
        # filter, and it ONLY applies to opening a new PUT (flat ->
        # selling cash-secured puts) -- covered-call selling against
        # shares already held is never gated by it (see module docstring).
        calendar_ok = day_of_month_ok(dt, entry_day_of_month)
        leg_type = "put" if shares_held == 0 else "call"
        bearish_ok = not (leg_type == "put" and bool(bearish.loc[dt]))
        if open_leg is None and i != last_i and calendar_ok and have_vol and bearish_ok:
            T_entry = dte_days / 365.0
            try:
                if leg_type == "put":
                    strike = strike_for_put_delta_magnitude(close, put_delta, T_entry, r, sigma)
                else:
                    strike = strike_for_delta("call", close, call_delta, T_entry, r, sigma)
                credit = black_scholes_price(leg_type, close, strike, T_entry, r, sigma)
            except ValueError:
                credit = 0.0
            if credit > 0.01 and strike > 0:
                cash += credit * CONTRACT_MULTIPLIER
                open_leg = {
                    "type": leg_type,
                    "entry_date": dt,
                    "expiration_date": dt + pd.Timedelta(days=dte_days),
                    "strike": strike,
                    "credit": credit,
                }

        # --- mark today's equity ---
        if open_leg is not None:
            leg_type = open_leg["type"]
            strike = open_leg["strike"]
            days_left = max((open_leg["expiration_date"] - dt).days, 0)
            T_mtm = days_left / 365.0
            if T_mtm > 0 and have_vol:
                mtm_cost = black_scholes_price(leg_type, close, strike, T_mtm, r, sigma)
            else:
                mtm_cost = (
                    max(strike - close, 0.0) if leg_type == "put" else max(close - strike, 0.0)
                )
            equity_curve.loc[dt] = cash + shares_held * close - mtm_cost * CONTRACT_MULTIPLIER
        else:
            equity_curve.loc[dt] = cash + shares_held * close

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
        strategy_name="modified_wheel",
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
