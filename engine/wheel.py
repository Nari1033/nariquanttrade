"""The Wheel Strategy backtester.

A three-step cycle, repeated for as long as there's data:
  1. While flat (no shares held), sell a cash-secured put around
     `put_delta` (e.g. 0.20 -> a ~20-delta put, ~80% odds of expiring
     worthless), `dte_days` out.
  2. If that put expires in the money, take assignment: buy 100 shares at
     the strike (the premium already collected softens the effective cost
     basis, but isn't separately modeled as lowering it -- see below).
  3. While holding shares, sell a covered call around `call_delta`,
     `dte_days` out, over and over. If a call expires in the money, the
     shares are called away at its strike and the cycle goes back to step 1.
     If a call expires worthless, the shares are kept and another call is
     sold immediately.

Unlike Bull Put Spread and Cash-Secured Put, there is deliberately no
trend filter here: the wheel's whole premise is being run on a stock
you're happy to own through a dip, not on timing entries, so puts are sold
whenever flat regardless of trend.

Bookkeeping is real cash + share inventory (not the "mark P&L since
inception" style the single-leg options backtesters use), because the
wheel actually holds an asset (100 shares) between option legs whose value
moves independently of any option's price:
  - Selling an option (put or call) credits `credit * 100` to cash
    immediately.
  - A put assigned at expiration debits `strike * 100` from cash and sets
    shares_held = 100.
  - A call assigned (shares called away) at expiration credits
    `strike * 100` to cash and sets shares_held = 0.
  - An option that expires worthless costs nothing further -- the credit
    already collected is realized as-is.
  - Every day's equity = cash + shares_held * close - (the open option's
    current cost to close) * 100, so the still-open leg is marked to
    market like every other backtester here, and, once shares are held,
    their own price moves flow straight into equity too.

Each option leg (one put-selling stretch, or one call-selling stretch) is
recorded as one `Trade`, same shape as Cash-Secured Put's: `entry_price`/
`exit_price` are the credit received / cost to close (a cash-settled
stand-in for real assignment -- see options_pricing.py's module docstring
for the general Black-Scholes caveats), and `meta["short_strike"]` /
`meta["exit_reason"]` follow the same convention so the existing trade-log
UI needs no changes. `meta["leg"]` ("put" or "call") is new -- app.py shows
it as an extra "Leg" column, generically, whenever any strategy's trades
carry it.

Simplifications, stated explicitly (same spirit as the other options
backtesters):
  - DTE is calendar days, not real Friday-only option-chain expirations.
  - Always exactly 1 contract / 100 shares -- no position sizing.
  - No bid/ask spread or commissions. No early assignment before
    expiration is modeled (American-style options can be assigned early
    in reality; this backtester, like the others in this app, only
    settles at the modeled expiration date).
  - The premium collected selling a put is credited to cash right away
    and is not separately netted against the assigned shares' cost basis
    -- economically it's already captured (assignment cash-out is
    `strike * 100`, and the credit sits in cash from having sold the put),
    it's just not folded into a single "effective cost basis" number the
    way a broker's position view might show it.
  - A leg still open when the data runs out is marked to market on the
    final bar and logged as a trade with `exit_reason="period_end"`, same
    convention as every other backtester here -- but unlike a true
    expiration, this does NOT flip share ownership, since nothing was
    actually assigned; it's a display-only mark of an unresolved position.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .models import PriceHistory
from .options_pricing import (
    black_scholes_price,
    realized_volatility,
    strike_for_delta,
    strike_for_put_delta_magnitude,
)

CONTRACT_MULTIPLIER = 100.0


def backtest_wheel_strategy(
    bars: PriceHistory,
    put_delta: float = 0.20,
    call_delta: float = 0.20,
    dte_days: int = 30,
    vol_window: int = 20,
    risk_free_rate_pct: float = 4.5,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    min_bars = vol_window + 5
    df = to_dataframe(bars)
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest (vol window={vol_window})"
        )

    sigma_series = realized_volatility(df["close"], window=vol_window)
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
        # left to manage it) ---
        if open_leg is None and i != last_i and have_vol:
            T_entry = dte_days / 365.0
            leg_type = "put" if shares_held == 0 else "call"
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
        strategy_name="wheel",
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
