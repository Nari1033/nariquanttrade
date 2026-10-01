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
from engine.rsi_strategy import backtest_rsi_momentum, rsi_breakout_recent
from engine.rsi_ema_wma_strategy import (
    backtest_rsi_ema_wma,
    rsi_ema_wma_bullish_recent,
    rsi_ema_wma_lines,
)
from engine.luxalgo_strategy import (
    backtest_market_flow_full,
    backtest_trendline_breakout_core,
    market_flow_full_recent,
    trendline_breakout_recent,
)
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
    oscillator_fn: Optional[Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series, pd.Series]]] = None
    """Optional: given (df, typed_params), returns three Series to plot on
    their own panel below the price chart (not overlaid on it -- for a
    strategy whose signal lines live on a different scale than price,
    e.g. the RSI(9)+EMA(3)+WMA(21) strategy's 0-100 RSI-derived lines).
    None for every strategy that doesn't have one."""


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


def _rsi_momentum_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    return rsi_breakout_recent(
        bars,
        period=int(kwargs.get("period", 14)),
        buy_threshold=float(kwargs.get("buy_threshold", 70.0)),
        overbought_threshold=float(kwargs.get("overbought_threshold", 80.0)),
        exit_threshold=float(kwargs.get("exit_threshold", 60.0)),
        lookback_days=lookback_days,
    )


def _rsi_ema_wma_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    return rsi_ema_wma_bullish_recent(
        bars,
        rsi_period=int(kwargs.get("rsi_period", 9)),
        ema_period=int(kwargs.get("ema_period", 3)),
        wma_period=int(kwargs.get("wma_period", 21)),
        min_gap=float(kwargs.get("min_gap", 5.0)),
        buy_rsi_level=float(kwargs.get("buy_rsi_level", 50.0)),
        sell_rsi_level=float(kwargs.get("sell_rsi_level", 50.0)),
        lookback_days=lookback_days,
    )


def _rsi_ema_wma_oscillator_fn(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return rsi_ema_wma_lines(
        df,
        rsi_period=int(params.get("rsi_period", 9)),
        ema_period=int(params.get("ema_period", 3)),
        wma_period=int(params.get("wma_period", 21)),
    )


def _trendline_breakout_core_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    return trendline_breakout_recent(
        bars,
        pivot_lookback=int(kwargs.get("pivot_lookback", 5)),
        trendline_points=int(kwargs.get("trendline_points", 3)),
        atr_period=int(kwargs.get("atr_period", 14)),
        lookback_days=lookback_days,
    )


def _market_flow_full_scan(bars, lookback_days: int = 3, **kwargs) -> bool:
    return market_flow_full_recent(
        bars,
        pivot_lookback=int(kwargs.get("pivot_lookback", 5)),
        trendline_points=int(kwargs.get("trendline_points", 3)),
        atr_period=int(kwargs.get("atr_period", 14)),
        ema_period=int(kwargs.get("ema_period", 20)),
        opening_range_minutes=int(kwargs.get("opening_range_minutes", 15)),
        liquidity_lookback_bars=int(kwargs.get("liquidity_lookback_bars", 50)),
        liquidity_proximity_atr_mult=float(kwargs.get("liquidity_proximity_atr_mult", 1.0)),
        gap_filter_pct=float(kwargs.get("gap_filter_pct", 2.0)),
        lookback_days=lookback_days,
    )


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
    Strategy(
        id="rsi_momentum",
        label="RSI(14) Momentum Breakout",
        description=(
            "Not the textbook RSI strategy -- this one buys strength instead of "
            "fading it. Buy when RSI(14) rises up through the buy threshold (70 "
            "by default), touching or crossing it from below. Sell with a "
            "\"faded momentum\" exit: once RSI has reached the overbought "
            "threshold (80) at some point since entry, sell as soon as it falls "
            "back to the (lower) exit threshold (60). A fixed extreme sell "
            "level like 90 almost never gets hit on real daily data, so this "
            "overbought-then-pullback design is what actually produces trades. "
            "A stop loss is included in case price falls hard before RSI ever "
            "reaches the overbought threshold."
        ),
        params=[
            NumberParam(
                "period", "RSI period", 14, 2, 50, step=1, is_int=True, is_sma_window=True,
                help="Number of bars used to smooth average gains/losses (Wilder's method).",
            ),
            NumberParam(
                "buy_threshold", "Entry (RSI rising through)", 70.0, 50.0, 90.0, step=1.0, is_int=False,
                help="Buy when RSI rises up through this level.",
            ),
            NumberParam(
                "overbought_threshold", "Top (must reach this first)", 80.0, 60.0, 95.0, step=1.0, is_int=False,
                help="RSI must reach or cross this level at some point after entry before the pullback exit can arm. Must be greater than Entry.",
            ),
            NumberParam(
                "exit_threshold", "Exit (sell when RSI falls back to)", 60.0, 30.0, 75.0, step=1.0, is_int=False,
                help="Once Top has been reached, sell as soon as RSI falls back to or below this level. Must be less than Top.",
            ),
            NumberParam(
                "stop_loss_pct", "Stop loss (% below entry)", 15.0, 1.0, 50.0, step=1.0, is_int=False,
                help="Close the position if price falls this far below the entry price, even without RSI ever reaching the overbought threshold.",
            ),
        ],
        scan_fn=_rsi_momentum_scan,
        backtest_fn=backtest_rsi_momentum,
    ),
    Strategy(
        id="rsi9_ema3_wma21",
        label="RSI(9) + EMA(3) + WMA(21)",
        description=(
            "Three lines, all derived from the same RSI(9) series: \"Strength\" "
            "is the raw RSI(9) value. \"Price\" is a fast EMA(3) smoothing of "
            "Strength. \"Volume\" is a slower WMA(21) smoothing of Strength -- "
            "a lagging baseline. Buy when the three lines stack Volume < Price "
            "< Strength (WMA below EMA, EMA below RSI), Strength is above the "
            "buy RSI level, and Strength leads Volume by at least the minimum "
            "gap; sell when they stack the other way (Volume > Price > "
            "Strength), Strength is below the sell RSI level, and Volume leads "
            "Strength by at least the minimum gap. Buy/sell RSI levels default "
            "to the conventional 50 midline but are independently adjustable. "
            "A moving-average-crossover-style strategy (no stop loss, "
            "long-only, no shorting)."
        ),
        params=[
            NumberParam(
                "rsi_period", "RSI period (Strength)", 9, 2, 50, step=1, is_int=True,
                help="Number of bars used to smooth average gains/losses for the RSI (Wilder's method).",
            ),
            NumberParam(
                "ema_period", "EMA period (Price)", 3, 1, 20, step=1, is_int=True,
                help="Fast EMA smoothing period applied to the RSI series itself.",
            ),
            NumberParam(
                "wma_period", "WMA period (Volume)", 21, 2, 60, step=1, is_int=True, is_sma_window=True,
                help="Slow WMA smoothing period applied to the RSI series itself -- the baseline Strength/Price cross.",
            ),
            NumberParam(
                "min_gap", "Min RSI-WMA gap", 5.0, 0.0, 50.0, step=1.0, is_int=False,
                help="Minimum required gap between Strength (RSI) and Volume (WMA) -- buy needs Strength - Volume >= this, sell needs Volume - Strength >= this. 0 disables the gap filter.",
            ),
            NumberParam(
                "buy_rsi_level", "Buy RSI level", 50.0, 1.0, 99.0, step=1.0, is_int=False,
                help="Strength (RSI) must be above this level for a buy signal. Defaults to the conventional 50 midline; raise it to require a more decisively bullish RSI before buying.",
            ),
            NumberParam(
                "sell_rsi_level", "Sell RSI level", 50.0, 1.0, 99.0, step=1.0, is_int=False,
                help="Strength (RSI) must be below this level for a sell signal. Defaults to the conventional 50 midline; lower it to require a more decisively bearish RSI before selling.",
            ),
        ],
        scan_fn=_rsi_ema_wma_scan,
        backtest_fn=backtest_rsi_ema_wma,
        oscillator_fn=_rsi_ema_wma_oscillator_fn,
    ),
    Strategy(
        id="trendline_breakout_core",
        label="Trend Line Breakout (Core)",
        description=(
            "Inspired by LuxAlgo's 'Market Flow Trend Lines & Liquidity' indicator "
            "(a free but closed-source TradingView script -- this is our own "
            "from-scratch approximation of its published description, not a copy "
            "of its exact formula). The 'core piece': an auto trend line is fit "
            "through the last few confirmed swing highs and projected forward; "
            "buy when price closes up through that line. Exit at a single "
            "ATR-based take-profit, an ATR-based stop-loss, or period end -- "
            "whichever comes first. Meant for live intraday data (try the "
            "15-Minute interval with a date range of 60 days or less, matching "
            "yfinance's own limit on how far back intraday bars go), but also "
            "runs on daily bars."
        ),
        params=[
            NumberParam(
                "pivot_lookback", "Pivot lookback (bars each side)", 5, 2, 20, step=1, is_int=True,
                help="A bar must be the strict highest/lowest within this many bars on both sides to count as a confirmed swing point.",
            ),
            NumberParam(
                "trendline_points", "Trend line points", 3, 2, 8, step=1, is_int=True,
                help="Number of the most recent confirmed swing highs used to fit the resistance trend line.",
            ),
            NumberParam(
                "atr_period", "ATR period", 14, 2, 50, step=1, is_int=True, is_sma_window=True,
                help="Bars used to smooth the Average True Range that sizes the take-profit and stop-loss.",
            ),
            NumberParam(
                "take_profit_atr_mult", "Take profit (x ATR above entry)", 2.0, 0.5, 10.0, step=0.5, is_int=False,
                help="Exit target = entry price + this many ATRs (measured at entry).",
            ),
            NumberParam(
                "stop_loss_atr_mult", "Stop loss (x ATR below entry)", 1.5, 0.5, 10.0, step=0.5, is_int=False,
                help="Exit stop = entry price - this many ATRs (measured at entry). Checked before the take-profit on any bar that would hit both.",
            ),
        ],
        scan_fn=_trendline_breakout_core_scan,
        backtest_fn=backtest_trendline_breakout_core,
    ),
    Strategy(
        id="market_flow_full",
        label="LuxAlgo Market Flow (Full Package)",
        description=(
            "The 'full package' version of Trend Line Breakout (Core) -- same "
            "trend-line breakout entry, but only taken when it also: breaks above "
            "the day's Opening Range, is trading in a rising EMA with price above "
            "it, is happening near a recent liquidity zone (a cluster of prior "
            "swing highs/lows) rather than out in open air, and isn't on a day "
            "that gapped down hard at the open. On daily data the Opening Range "
            "can't be computed (it needs multiple bars per day), so that one "
            "filter passes through rather than blocking every trade -- the other "
            "three still apply. Position size is split into 3 equal legs at "
            "entry, each targeting its own ATR-based take-profit level (matching "
            "the real indicator's 'up to 3 ATR based take profits'), all sharing "
            "one ATR-based stop-loss. Best used with live intraday data (try the "
            "15-Minute interval, 60 days or less)."
        ),
        params=[
            NumberParam(
                "pivot_lookback", "Pivot lookback (bars each side)", 5, 2, 20, step=1, is_int=True,
                help="A bar must be the strict highest/lowest within this many bars on both sides to count as a confirmed swing point.",
            ),
            NumberParam(
                "trendline_points", "Trend line points", 3, 2, 8, step=1, is_int=True,
                help="Number of the most recent confirmed swing highs used to fit the resistance trend line.",
            ),
            NumberParam(
                "atr_period", "ATR period", 14, 2, 50, step=1, is_int=True, is_sma_window=True,
                help="Bars used to smooth the Average True Range that sizes the take-profits and stop-loss.",
            ),
            NumberParam(
                "ema_period", "EMA period (trend filter)", 20, 2, 100, step=1, is_int=True,
                help="Only take a breakout when price is above a rising EMA of this length.",
            ),
            NumberParam(
                "opening_range_minutes", "Opening range (minutes)", 15, 5, 60, step=5, is_int=True,
                help="Length of the first-of-the-day window whose high must be broken. No effect on daily bars (see description).",
            ),
            NumberParam(
                "liquidity_lookback_bars", "Liquidity zone lookback (bars)", 50, 10, 300, step=10, is_int=True,
                help="How far back to look for a prior swing high/low to count as a nearby liquidity zone.",
            ),
            NumberParam(
                "liquidity_proximity_atr_mult", "Liquidity zone proximity (x ATR)", 1.0, 0.1, 5.0, step=0.1, is_int=False,
                help="How close (in ATRs) price must be to a recent swing high/low to count as 'near a liquidity zone.'",
            ),
            NumberParam(
                "gap_filter_pct", "Skip entries after a gap-down of (%)", 2.0, 0.0, 10.0, step=0.5, is_int=False,
                help="No new entries for the rest of a day that opened down this much (as a %) from the prior close. 0 disables this filter.",
            ),
            NumberParam(
                "take_profit_1_atr_mult", "Take profit 1 (x ATR)", 1.0, 0.25, 10.0, step=0.25, is_int=False,
                help="Nearest of the 3 scaled exit targets (1/3 of the position).",
            ),
            NumberParam(
                "take_profit_2_atr_mult", "Take profit 2 (x ATR)", 2.0, 0.5, 15.0, step=0.25, is_int=False,
                help="Middle of the 3 scaled exit targets (1/3 of the position). Must be greater than take profit 1.",
            ),
            NumberParam(
                "take_profit_3_atr_mult", "Take profit 3 (x ATR)", 3.0, 0.75, 20.0, step=0.25, is_int=False,
                help="Furthest of the 3 scaled exit targets (1/3 of the position). Must be greater than take profit 2.",
            ),
            NumberParam(
                "stop_loss_atr_mult", "Stop loss (x ATR below entry)", 1.5, 0.5, 10.0, step=0.5, is_int=False,
                help="Shared exit stop for all 3 legs = entry price - this many ATRs (measured at entry).",
            ),
        ],
        scan_fn=_market_flow_full_scan,
        backtest_fn=backtest_market_flow_full,
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
