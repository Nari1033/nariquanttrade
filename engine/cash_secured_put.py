"""Cash-Secured Put backtester.

Setup: sell a single out-of-the-money put (no protective long leg -- the
"cash secured" part means the trader sets aside strike * 100 in cash and is
willing to be assigned the stock at the strike, rather than buying a
further-OTM put to cap the downside the way Bull Put Spread does).

Rules modeled here, matching the common "sell puts for income" recipe:
  - Enter a new position whenever flat (no open position) and the trend
    filter passes (underlying close > its SMA-`trend_sma_window` -- still
    only sell puts in an uptrend/neutral market, not into a confirmed
    downtrend).
  - Strike: placed at `short_delta` (e.g. 0.30 = a ~30-delta put) at entry,
    `dte_entry` calendar days to expiration.
  - Exit at whichever comes first: the put's value has decayed to
    `profit_target_pct`% of the credit received (i.e. you've captured that
    much of the max profit -- the "sell at 50% max" rule), it has grown to
    `stop_loss_pct`% *beyond* the credit received, or expiration.

Option prices are modeled with Black-Scholes, using the underlying's own
realized volatility as an implied-volatility proxy (see
engine/options_pricing.py) -- see that module's docstring for the same
caveats that apply to Bull Put Spread (no real historical IV, no skew).

Simplifications, stated explicitly (same conventions as Bull Put Spread):
  - DTE is calendar days added to the entry date, not real Friday-only
    option-chain expirations.
  - Always 1 contract per trade -- no position sizing relative to capital
    or the cash actually required to secure the put (strike * 100).
  - No bid/ask spread or commissions modeled. Assignment at expiration is
    modeled as a cash settlement equal to the option's intrinsic value
    (max(strike - close, 0)) rather than simulating buying and later
    selling the underlying shares -- economically equivalent P&L, ignoring
    the timing/tax difference of actually holding the stock.
  - A position still open when the data runs out is marked to market on
    the final bar (closed at that day's theoretical value) and counted as
    a trade, same convention as every other backtester in this app.

Reuses the same `Trade`/`BacktestResult` shape as engine.backtester so it
drops into the same Streamlit backtest panel: `entry_price`/`exit_price`
here are the credit received / cost to close (dollars per share, not a
stock price), and `return_pct` is % of the credit captured (or lost). The
strike is stored under `meta["short_strike"]` (reusing Bull Put Spread's
key name, since it *is* the strike of the short put) so the existing
trade-log UI shows it and labels the price columns "Credit/debit" with no
app.py changes needed; `meta["long_strike"]` is simply absent here since
there's no long leg.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .backtester import BacktestResult, Trade
from .data_utils import to_dataframe
from .indicators import add_sma_columns
from .models import PriceHistory
from .options_pricing import black_scholes_price, realized_volatility, strike_for_put_delta_magnitude

CONTRACT_MULTIPLIER = 100.0


def backtest_cash_secured_put(
    bars: PriceHistory,
    dte_entry: int = 45,
    short_delta: float = 0.30,
    profit_target_pct: float = 50.0,
    stop_loss_pct: float = 200.0,
    trend_sma_window: int = 200,
    vol_window: int = 20,
    risk_free_rate_pct: float = 4.5,
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
) -> BacktestResult:
    min_bars = max(trend_sma_window, vol_window) + 5
    df = add_sma_columns(to_dataframe(bars), windows=(trend_sma_window,))
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} price bars to run this backtest "
            f"(trend SMA window={trend_sma_window}, vol window={vol_window})"
        )

    trend_col = f"sma_{trend_sma_window}"
    sigma_series = realized_volatility(df["close"], window=vol_window)
    r = risk_free_rate_pct / 100.0

    trades: List[Trade] = []
    equity = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)
    position = None  # dict: entry_date, expiration_date, strike, credit

    dates = df.index.to_list()
    last_i = len(dates) - 1

    for i, dt in enumerate(dates):
        close = float(df["close"].loc[dt])
        sigma = sigma_series.loc[dt]
        trend_val = df[trend_col].loc[dt]
        trend_ok = pd.notna(trend_val) and close > trend_val

        # --- manage an existing position: reprice, check exit, maybe close ---
        if position is not None:
            days_left = (position["expiration_date"] - dt).days
            T_remaining = days_left / 365.0
            have_vol = pd.notna(sigma) and sigma > 0

            if T_remaining <= 0 or not have_vol:
                cost_to_close = max(position["strike"] - close, 0.0)
                exit_reason = "expiration"
            else:
                cost_to_close = black_scholes_price(
                    "put", close, position["strike"], T_remaining, r, sigma
                )
                credit = position["credit"]
                profit_trigger = credit * (1 - profit_target_pct / 100.0)
                loss_trigger = credit * (1 + stop_loss_pct / 100.0)
                if cost_to_close <= profit_trigger:
                    exit_reason = "profit_target"
                elif cost_to_close >= loss_trigger:
                    exit_reason = "stop_loss"
                else:
                    exit_reason = None

            if exit_reason is None and i == last_i:
                exit_reason = "period_end"

            if exit_reason is not None:
                credit = position["credit"]
                pnl_dollars = (credit - cost_to_close) * CONTRACT_MULTIPLIER
                return_pct = (credit - cost_to_close) / credit * 100.0
                equity += pnl_dollars
                trades.append(
                    Trade(
                        entry_date=position["entry_date"],
                        entry_price=credit,
                        exit_date=dt,
                        exit_price=cost_to_close,
                        return_pct=return_pct,
                        is_win=return_pct > 0,
                        closed_at_period_end=(exit_reason == "period_end"),
                        meta={
                            "exit_reason": exit_reason,
                            "short_strike": round(position["strike"], 2),
                            "dte_entry": dte_entry,
                        },
                    )
                )
                position = None

        # --- consider opening a new position (not on the final bar -- no
        # time left to manage it) ---
        if position is None and i != last_i and trend_ok and pd.notna(sigma) and sigma > 0:
            T_entry = dte_entry / 365.0
            try:
                strike = strike_for_put_delta_magnitude(close, short_delta, T_entry, r, sigma)
                credit = black_scholes_price("put", close, strike, T_entry, r, sigma)
            except ValueError:
                credit = 0.0
            if credit > 0.01 and strike > 0:
                position = {
                    "entry_date": dt,
                    "expiration_date": dt + pd.Timedelta(days=dte_entry),
                    "strike": strike,
                    "credit": credit,
                }

        # --- mark today's equity ---
        if position is not None:
            days_left = max((position["expiration_date"] - dt).days, 0)
            T_mtm = days_left / 365.0
            if T_mtm > 0 and pd.notna(sigma) and sigma > 0:
                mtm_cost = black_scholes_price("put", close, position["strike"], T_mtm, r, sigma)
            else:
                mtm_cost = max(position["strike"] - close, 0.0)
            unrealized = (position["credit"] - mtm_cost) * CONTRACT_MULTIPLIER
            equity_curve.loc[dt] = equity + unrealized
        else:
            equity_curve.loc[dt] = equity

    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate_pct = (wins / total_trades * 100) if total_trades else 0.0
    strategy_return_pct = (equity - initial_capital) / initial_capital * 100

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    buy_hold_return_pct = (last_close - first_close) / first_close * 100
    buy_hold_curve = (df["close"] / first_close) * initial_capital

    return BacktestResult(
        ticker=ticker,
        strategy_name="cash_secured_put",
        start_date=df.index[0],
        end_date=df.index[-1],
        fast_window=None,
        slow_window=trend_sma_window,
        initial_capital=initial_capital,
        strategy_return_pct=strategy_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        trades=trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
    )
