"""Strategy registry.

Each `Strategy` bundles a scanner function and a backtester function under
one label, plus the numeric knobs that are specific to it. `app.py` loops
over `STRATEGIES` once and renders a "Scanner" + "Backtest" pair of
sub-tabs for each entry -- it doesn't otherwise know or care what any given
strategy does.

To add a new strategy: write its scan_fn/backtest_fn in `engine/` (they
just need the signatures `scan_fn(bars, lookback_days=..., **params) ->
bool` and `backtest_fn(bars, initial_capital=..., ticker=..., **params) ->
BacktestResult`), then append one `Strategy(...)` entry below. Nothing in
app.py needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from engine.backtester import (
    BacktestResult,
    backtest_price_sma_crossover,
    backtest_sma_crossover,
)
from engine.bollinger import backtest_bollinger_mean_reversion, bollinger_bands, bollinger_oversold_recent
from engine.options_backtester import backtest_bull_put_spread
from engine.cash_secured_put import backtest_cash_secured_put
from engine.wheel import backtest_wheel_strategy
from engine.scanner import golden_cross_recent, price_cross_sma_recent


@dataclass
class NumberParam:
    """One numeric knob for a strategy, rendered as an st.number_input.
    `key` must match the keyword argument name on both the strategy's
    scan_fn and backtest_fn."""

    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float = 1.0
    is_int: bool = True
    help: Optional[str] = None
    is_sma_window: bool = False
    """True if this param's value is a bar-count SMA window (e.g. a
    50/200-day moving average). app.py uses this -- instead of assuming
    every param is one -- to decide which sma_N columns to compute for the
    chart and how much warm-up history to fetch before a custom backtest
    start date. False for params like an options strategy's delta or
    profit-target percentage, which aren't SMA windows at all."""


@dataclass
class Strategy:
    id: str
    label: str
    description: str
    params: List[NumberParam]
    scan_fn: Callable[..., bool]
    backtest_fn: Callable[..., BacktestResult]
    band_fn: Optional[Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]]] = None
    """Optional: given (df, typed_params), returns (upper_band, lower_band)
    Series to overlay on the Backtest tab's price chart -- for a strategy
    built around a band concept (e.g. Bollinger Bands). None for every
    strategy that doesn't have one; app.py only draws the overlay when a
    strategy provides it, so this needs no changes elsewhere to add."""


def _price_cross_sma_scan(bars, **kwargs) -> bool:
    # Pin direction="above" -- "price crosses below its SMA" would be a
    # natural *second* strategy entry later, not a toggle on this one.
    return price_cross_sma_recent(bars, direction="above", **kwargs)


def _bollinger_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    # Only the band window/width matter for "is there a fresh oversold
    # signal to scan for" -- stop_loss_pct only matters once a backtest is
    # actually managing an open position.
    return bollinger_oversold_recent(
        bars,
        window=int(kwargs.get("window", 20)),
        num_std=float(kwargs.get("num_std", 2.0)),
        lookback_days=lookback_days,
    )


def _bollinger_band_fn(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    _, upper, lower = bollinger_bands(
        df["close"], window=int(params.get("window", 20)), num_std=float(params.get("num_std", 2.0))
    )
    return upper, lower


def _bull_put_spread_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    # The Scanner tab's job is just "would this strategy be eligible to
    # enter a new position soon" -- reuse the trend filter (price crossing
    # above its own trend SMA) as that eligibility signal, and ignore the
    # rest of this strategy's params (delta, spread width, profit/stop %,
    # DTE), which only matter once you're actually pricing a trade in the
    # backtester, not for a quick scan.
    trend_sma_window = kwargs.get("trend_sma_window", 200)
    return price_cross_sma_recent(
        bars, sma_window=int(trend_sma_window), lookback_days=lookback_days, direction="above"
    )


def _cash_secured_put_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    # Same eligibility idea as the bull put spread scan: only worth
    # scanning for a fresh put-selling opportunity when price has just
    # crossed above its own trend SMA.
    trend_sma_window = kwargs.get("trend_sma_window", 200)
    return price_cross_sma_recent(
        bars, sma_window=int(trend_sma_window), lookback_days=lookback_days, direction="above"
    )


def _wheel_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    # The wheel has no trend filter -- selling a put is always the next
    # step once flat, regardless of trend (that's the whole premise: you'd
    # be happy to own the stock). So "eligible" here just means "enough
    # price history to price an option at all," not a timing signal.
    vol_window = int(kwargs.get("vol_window", 20))
    return len(bars) >= vol_window + 5


STRATEGIES: List[Strategy] = [
    Strategy(
        id="golden_cross",
        label="Golden Cross (SMA 50 / SMA 200)",
        description=(
            "Buy when the fast SMA crosses above the slow SMA (a 'Golden Cross'); "
            "sell when it crosses back below (a 'Death Cross')."
        ),
        params=[
            NumberParam("fast_window", "Fast SMA window", 50, 2, 200, is_sma_window=True),
            NumberParam("slow_window", "Slow SMA window", 200, 5, 400, is_sma_window=True),
        ],
        scan_fn=golden_cross_recent,
        backtest_fn=backtest_sma_crossover,
    ),
    Strategy(
        id="price_cross_sma",
        label="Price crosses SMA 50",
        description=(
            "Buy when price closes above its own SMA-N; sell when it closes "
            "back below it."
        ),
        params=[
            NumberParam("sma_window", "SMA window", 50, 2, 400, is_sma_window=True),
        ],
        scan_fn=_price_cross_sma_scan,
        backtest_fn=backtest_price_sma_crossover,
    ),
    Strategy(
        id="bollinger_mean_reversion",
        label="Bollinger Band Mean Reversion",
        description=(
            "Prices tend to revert to their average after hitting an extreme. Buy "
            "when price dips below (or touches) the lower band and closes back "
            "inside; sell when it spikes above (or touches) the upper band and "
            "closes back inside. A stop loss is included since a strong trend can "
            "make price hug a band instead of reverting."
        ),
        params=[
            NumberParam(
                "window", "Band window", 20, 5, 100, step=1, is_int=True, is_sma_window=True,
                help="Bars used for the middle band (SMA) and the rolling standard deviation.",
            ),
            NumberParam(
                "num_std", "Band width (std devs)", 2.0, 1.0, 4.0, step=0.25, is_int=False,
                help="Upper/lower bands sit this many standard deviations from the middle band.",
            ),
            NumberParam(
                "stop_loss_pct", "Stop loss (% below entry)", 10.0, 1.0, 50.0, step=1.0, is_int=False,
                help="Close the position if price falls this far below the entry price, even without a sell signal.",
            ),
        ],
        scan_fn=_bollinger_scan,
        backtest_fn=backtest_bollinger_mean_reversion,
        band_fn=_bollinger_band_fn,
    ),
    Strategy(
        id="bull_put_spread",
        label="Bull Put Spread (options)",
        description=(
            "Sell an OTM put and buy a further-OTM put for a net credit, only "
            "when price is above its own trend SMA (uptrend filter). Exit at a "
            "profit target, a stop loss, or expiration -- whichever comes first. "
            "Option prices are modeled with Black-Scholes using the underlying's "
            "own realized volatility as an implied-volatility proxy (no real "
            "historical options data is used) -- see the engine.options_pricing "
            "module docstring for the caveats."
        ),
        params=[
            NumberParam(
                "dte_entry", "Entry DTE (days)", 30, 5, 90, step=1, is_int=True,
                help="Days to expiration at entry.",
            ),
            NumberParam(
                "short_delta", "Short put delta", 0.30, 0.05, 0.50, step=0.05, is_int=False,
                help="Target delta magnitude of the short (sold) put -- higher = closer to the money.",
            ),
            NumberParam(
                "spread_width_pct", "Spread width (% of spot)", 3.0, 0.5, 15.0, step=0.5, is_int=False,
                help="How far below the short strike the long (protective) put sits, as a % of the stock price.",
            ),
            NumberParam(
                "profit_target_pct", "Profit target (% of credit)", 50.0, 10.0, 90.0, step=5.0, is_int=False,
                help="Close the position once this much of the credit received has been captured.",
            ),
            NumberParam(
                "stop_loss_pct", "Stop loss (% of credit)", 100.0, 25.0, 300.0, step=25.0, is_int=False,
                help="Close the position if its cost to close grows to this much *beyond* the credit received.",
            ),
            NumberParam(
                "trend_sma_window", "Trend filter SMA window", 200, 20, 400, step=10, is_int=True,
                is_sma_window=True,
                help="Only enter new positions while price is above this SMA (trade only in an uptrend).",
            ),
            NumberParam(
                "entry_day_of_month", "Entry day of month (0 = any)", 0, 0, 28, step=1, is_int=True,
                help=(
                    'Only open a new position within 5 calendar days of this day of the month (e.g. 1 = near month-start, 15 = mid-month, 28 = month-end). 0 = no day-of-month filter -- enter whenever the other conditions are met, same as before this existed.'
                ),
            ),
        ],
        scan_fn=_bull_put_spread_scan,
        backtest_fn=backtest_bull_put_spread,
    ),
    Strategy(
        id="cash_secured_put",
        label="Cash-Secured Put (options)",
        description=(
            "Sell a single out-of-the-money put and set aside the cash to buy the "
            "stock if assigned, rather than buying a further-OTM put to cap the "
            "downside (see Bull Put Spread for that version). Only sells puts "
            "while price is above its own trend SMA, and closes the position once "
            "it has captured the target share of the max profit -- 'sell at 50% "
            "max' by default. Same Black-Scholes modeling caveats as Bull Put "
            "Spread apply -- see the engine.options_pricing module docstring."
        ),
        params=[
            NumberParam(
                "dte_entry", "Entry DTE (days)", 45, 5, 90, step=1, is_int=True,
                help="Days to expiration at entry.",
            ),
            NumberParam(
                "short_delta", "Put delta", 0.30, 0.05, 0.50, step=0.05, is_int=False,
                help="Target delta magnitude of the short put -- higher = closer to the money.",
            ),
            NumberParam(
                "profit_target_pct", "Profit target (% of credit)", 50.0, 10.0, 90.0, step=5.0, is_int=False,
                help="Close the position once this much of the credit received has been captured.",
            ),
            NumberParam(
                "stop_loss_pct", "Stop loss (% of credit)", 200.0, 25.0, 300.0, step=25.0, is_int=False,
                help="Close the position if its cost to close grows to this much *beyond* the credit received.",
            ),
            NumberParam(
                "trend_sma_window", "Trend filter SMA window", 200, 20, 400, step=10, is_int=True,
                is_sma_window=True,
                help="Only sell new puts while price is above this SMA (avoid selling into a downtrend).",
            ),
            NumberParam(
                "entry_day_of_month", "Entry day of month (0 = any)", 0, 0, 28, step=1, is_int=True,
                help=(
                    'Only open a new position within 5 calendar days of this day of the month (e.g. 1 = near month-start, 15 = mid-month, 28 = month-end). 0 = no day-of-month filter -- enter whenever the other conditions are met, same as before this existed.'
                ),
            ),
        ],
        scan_fn=_cash_secured_put_scan,
        backtest_fn=backtest_cash_secured_put,
    ),
    Strategy(
        id="wheel",
        label="The Wheel (options)",
        description=(
            "A three-step cycle: sell cash-secured puts while flat (target "
            "delta ~0.20, for roughly an 80% chance of expiring worthless); if "
            "assigned, take the shares; then sell covered calls against them "
            "until they're called away, and start over. No trend filter -- the "
            "whole premise is being happy to own the stock through a dip, not "
            "timing entries. Same Black-Scholes modeling caveats as the other "
            "options strategies apply -- see the engine.options_pricing module "
            "docstring."
        ),
        params=[
            NumberParam(
                "put_delta", "Put delta", 0.20, 0.05, 0.50, step=0.05, is_int=False,
                help="Target delta magnitude of the cash-secured put sold while flat.",
            ),
            NumberParam(
                "call_delta", "Call delta", 0.20, 0.05, 0.50, step=0.05, is_int=False,
                help="Target delta of the covered call sold while holding the shares.",
            ),
            NumberParam(
                "dte_days", "Days to expiration", 30, 5, 90, step=1, is_int=True,
                help="Days to expiration at entry, for both the puts and the calls.",
            ),
            NumberParam(
                "entry_day_of_month", "Entry day of month (0 = any)", 0, 0, 28, step=1, is_int=True,
                help=(
                    'Only open a new position within 5 calendar days of this day of the month (e.g. 1 = near month-start, 15 = mid-month, 28 = month-end). 0 = no day-of-month filter -- enter whenever the other conditions are met, same as before this existed.'
                ),
            ),
        ],
        scan_fn=_wheel_scan,
        backtest_fn=backtest_wheel_strategy,
    ),
]


def get_strategy(strategy_id: str) -> Strategy:
    for s in STRATEGIES:
        if s.id == strategy_id:
            return s
    raise KeyError(f"No such strategy: {strategy_id!r}")


def default_grid_values(p: NumberParam, n: int = 4) -> List[float]:
    """A handful of evenly-spaced candidate values across [min, max] for
    `p`, always including its default -- used to seed the Sweep page's
    per-parameter multiselect so there's a sane starting grid instead of
    an empty one. Not exhaustive by design: a full sweep of every
    strategy's full min/max range would be millions of combinations, this
    just gives a reasonable few points to start from (the user can add or
    remove values in the UI).

    NOTE: this is *not* guaranteed to return a subset of what a different
    `n` would return (two different linspaces over the same range don't
    generally share points), so don't call this twice with different `n`
    and feed one result in as another Streamlit widget's `default` against
    the other as its `options` -- Streamlit requires every default to
    literally be one of the options, or it raises. Use
    `default_grid_selection` for that "candidate pool + a subset of it as
    the default" pattern instead.
    """
    lo, hi = float(p.min_value), float(p.max_value)
    if n <= 1 or hi <= lo:
        vals = [float(p.default)]
    else:
        step = (hi - lo) / (n - 1)
        vals = [lo + i * step for i in range(n)]

    if p.is_int:
        vals = sorted({int(round(v)) for v in vals} | {int(p.default)})
    else:
        vals = sorted({round(v, 2) for v in vals} | {round(float(p.default), 2)})
    return vals


def default_grid_selection(p: NumberParam, pool_n: int = 6, default_n: int = 4):
    """Like `default_grid_values`, but returns (candidate_pool, default_subset)
    where default_subset is guaranteed to be a literal subset of
    candidate_pool -- safe to pass straight through as a Streamlit
    `st.multiselect(options=candidate_pool, default=default_subset)` call,
    which raises if `default` contains anything not in `options`."""
    candidates = default_grid_values(p, n=pool_n)
    if len(candidates) <= default_n:
        return candidates, candidates

    # Evenly-spaced *indices* into the candidate pool (not a fresh
    # linspace over the value range), so every chosen default value is by
    # construction one of the candidates -- always keep the endpoints, and
    # always keep the pool's default value.
    default_value = int(p.default) if p.is_int else round(float(p.default), 2)
    idx_set = {round(i * (len(candidates) - 1) / (default_n - 1)) for i in range(default_n)}
    if default_value in candidates:
        idx_set.add(candidates.index(default_value))
    defaults = sorted({candidates[i] for i in idx_set})
    return candidates, defaults
