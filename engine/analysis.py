"""Underperformance analysis: for each completed trade, explain in plain
English why the strategy's *cumulative* return had fallen behind buy &
hold's cumulative return as of that trade's exit -- not why the individual
trade was good or bad in isolation.

Generic by design: works from `Trade` objects plus the two equity curves
(`engine.backtester.Trade`, `BacktestResult.equity_curve` /
`.buy_hold_curve`) and the underlying OHLCV `df` that every strategy in
this app already produces. It has no strategy-specific branching -- it
reads `Trade.meta.get("exit_reason")` generically (covers every value any
strategy in this app can set: "profit_target", "stop_loss",
"band_rejection", "expiration", "period_end", or absent entirely for the
plain SMA crossover strategies), so it applies automatically to every
current and future `Strategy` entry in app/strategies.py with no app.py
changes needed per strategy.

Two things can each pull cumulative performance behind buy & hold, and a
given trade can show either, both, or neither:
  1. The trade itself: a loss, or a win that captured only part of the
     underlying's move over the same entry-to-exit dates (e.g. an early
     profit-target/stop-loss/band-rejection exit ahead of a bigger
     continuation move).
  2. The gap before the trade: time spent flat (out of a position, in
     cash) between the previous exit and this entry, during which the
     underlying moved without the strategy participating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .backtester import Trade

_EXIT_REASON_LABELS = {
    "profit_target": "the position was closed at its profit target",
    "stop_loss": "the position was stopped out",
    "band_rejection": "the position was closed on the mean-reversion sell signal",
    "expiration": "the option reached expiration",
    "period_end": "the window ended while the position was still open",
}

# Below this, a price move is treated as noise rather than a cause worth
# naming -- avoids manufacturing an "explanation" out of a 0.1% wiggle.
_MATERIAL_MOVE_PCT = 0.5


def _pct_change(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[float]:
    """% change in `df["close"]` between the two dates, using the last
    known price at or before each date (so this tolerates dates that
    don't land exactly on a trading day/bar). None if either side can't
    be resolved (e.g. `start_date` before the data begins)."""
    if df.empty:
        return None
    try:
        start_px = df["close"].asof(start_date)
        end_px = df["close"].asof(end_date)
    except Exception:
        return None
    if start_px is None or end_px is None or pd.isna(start_px) or pd.isna(end_px) or start_px == 0:
        return None
    return (float(end_px) - float(start_px)) / float(start_px) * 100.0


def analyze_underperformance(
    trades: List[Trade],
    equity_curve: pd.Series,
    buy_hold_curve: pd.Series,
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """For each trade (in exit-date order), compare the strategy's
    cumulative return to buy & hold's cumulative return *as of that
    trade's exit*, both rebased to the start of `equity_curve` /
    `buy_hold_curve` (pass the already date-trimmed "visible window"
    curves so this lines up with whatever window the UI is showing).
    Returns one record per trade where the strategy was behind at that
    point, each with a plain-English `explanation`; trades where the
    strategy was even with or ahead of buy & hold are omitted.
    """
    if len(equity_curve) == 0 or len(buy_hold_curve) == 0:
        return []

    equity_base = float(equity_curve.iloc[0])
    bh_base = float(buy_hold_curve.iloc[0])
    window_start = equity_curve.index[0]
    equity_last_dt = equity_curve.index[-1]
    bh_last_dt = buy_hold_curve.index[-1]

    if equity_base == 0 or bh_base == 0:
        return []

    records: List[Dict[str, Any]] = []
    prior_exit_date = window_start

    for t in sorted(trades, key=lambda tr: tr.exit_date):
        lookup_dt = min(t.exit_date, equity_last_dt)
        bh_lookup_dt = min(t.exit_date, bh_last_dt)
        strat_val = equity_curve.asof(lookup_dt)
        bh_val = buy_hold_curve.asof(bh_lookup_dt)

        if pd.isna(strat_val) or pd.isna(bh_val):
            prior_exit_date = max(prior_exit_date, t.exit_date)
            continue

        strat_cum_pct = (float(strat_val) - equity_base) / equity_base * 100.0
        bh_cum_pct = (float(bh_val) - bh_base) / bh_base * 100.0
        gap_pct = strat_cum_pct - bh_cum_pct

        if gap_pct >= -1e-9:
            # Even with or ahead of buy & hold at this point -- nothing to explain.
            prior_exit_date = t.exit_date
            continue

        reasons: List[str] = []

        trade_underlying_pct = _pct_change(df, t.entry_date, t.exit_date)

        if not t.is_win:
            piece = f"this trade lost {t.return_pct:+.1f}%"
            if trade_underlying_pct is not None:
                piece += f" while the underlying moved {trade_underlying_pct:+.1f}% over the same dates"
            reasons.append(piece)
        elif (
            trade_underlying_pct is not None
            and trade_underlying_pct - t.return_pct > _MATERIAL_MOVE_PCT
        ):
            cause = _EXIT_REASON_LABELS.get(t.meta.get("exit_reason"), "the position was closed")
            reasons.append(
                f"the trade gained {t.return_pct:+.1f}% but the underlying rose "
                f"{trade_underlying_pct:+.1f}% over the same dates -- {cause} before the full move played out"
            )

        if t.entry_date > prior_exit_date:
            exposure_gap_pct = _pct_change(df, prior_exit_date, t.entry_date)
            if exposure_gap_pct is not None and exposure_gap_pct > _MATERIAL_MOVE_PCT:
                days_out = (t.entry_date - prior_exit_date).days
                reasons.append(
                    f"the underlying rose {exposure_gap_pct:+.1f}% over the {days_out} day(s) "
                    "before entry while the strategy was flat (out of the market)"
                )

        if not reasons:
            reasons.append(
                "cumulative return had already fallen behind buy & hold by this point, without one "
                "dominant, isolated cause in this trade or the gap before it -- likely the drag of "
                "several smaller trades/gaps adding up"
            )

        records.append(
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "trade_return_pct": t.return_pct,
                "strategy_cumulative_pct": strat_cum_pct,
                "buy_hold_cumulative_pct": bh_cum_pct,
                "gap_pct": gap_pct,
                "explanation": "; ".join(reasons) + ".",
            }
        )
        prior_exit_date = t.exit_date

    return records
