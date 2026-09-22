"""Parameter sweep utilities: run a strategy's backtest across many
parameter combinations and multiple date windows, to find configurations
that beat buy-and-hold -- and how consistently they do it.

Kept generic (not hardcoded to Bull Put Spread) so any strategy in
app/strategies.py with its own tunable NumberParams can reuse the same
sweep mechanics from app.py's Sweep page.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .backtester import BacktestResult, annualized_return_pct
from .data_utils import to_dataframe
from .models import PriceHistory


@dataclass
class SweepPeriod:
    """One date window to test every parameter combination over, e.g.
    "2020-09 -> 2022-09". `start`/`end` are anything pd.Timestamp accepts
    (date, datetime, or an ISO string)."""

    label: str
    start: object
    end: object


def _rebase_to_window(curve: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[float]:
    """% return of `curve` restricted to [start_ts, end_ts] -- same
    "rebase to the visible window" convention app.py's Backtest panel uses
    so a sweep's numbers match what you'd see running that combo by hand.
    None if fewer than 2 points fall in the window (nothing to compare)."""
    visible = curve.loc[start_ts:end_ts]
    if len(visible) < 2:
        return None
    return (visible.iloc[-1] - visible.iloc[0]) / visible.iloc[0] * 100.0


def generate_param_combos(
    param_grid: Dict[str, Sequence[float]],
    max_combos: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """All combinations from the cartesian product of param_grid's value
    lists, or a deterministic random sample of `max_combos` of them when
    the full product would be too large to run in a UI session. Sampling
    (rather than truncating) is what makes a coarse "explore the whole
    space a bit" sweep useful even when the full grid is in the millions.
    """
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    if not keys:
        return [{}]

    total = 1
    for vs in value_lists:
        total *= max(len(vs), 1)

    if max_combos is None or total <= max_combos:
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    rng = random.Random(seed)
    seen = set()
    combos: List[Dict[str, float]] = []
    max_attempts = max_combos * 30
    attempts = 0
    while len(combos) < max_combos and attempts < max_attempts:
        attempts += 1
        pick = tuple(rng.choice(vs) for vs in value_lists)
        if pick in seen:
            continue
        seen.add(pick)
        combos.append(dict(zip(keys, pick)))
    return combos


def sweep_strategy(
    bars: PriceHistory,
    backtest_fn: Callable[..., BacktestResult],
    param_grid: Dict[str, Sequence[float]],
    periods: List[SweepPeriod],
    buffer_days_fn: Callable[[Dict[str, float]], int],
    initial_capital: float = 10_000.0,
    ticker: Optional[str] = None,
    max_combos: Optional[int] = 150,
    seed: int = 42,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Run `backtest_fn` over every (param combo x period) pair and return
    a tidy DataFrame, one row per pair, with the strategy's and
    buy-and-hold's return *within that period's window* (not the whole
    backtest range), the edge (strategy - buy&hold), and basic trade
    stats. Rows where the strategy couldn't be evaluated (not enough data
    in that window, or the backtest raised, e.g. too little warm-up
    history) are skipped rather than erroring out the whole sweep.

    `buffer_days_fn(params) -> int` sizes how much warm-up history to
    fetch before each period's start (e.g. so the trend SMA is valid from
    day one of the window) -- mirrors the same buffer logic app.py's
    Backtest panel already uses for a single manual run.
    """
    full_df = to_dataframe(bars)
    combos = generate_param_combos(param_grid, max_combos=max_combos, seed=seed)

    rows = []
    total_runs = len(combos) * len(periods)
    done = 0
    for params in combos:
        buffer_days = buffer_days_fn(params)
        for period in periods:
            done += 1
            if progress_cb:
                progress_cb(done, total_runs)

            start_ts = pd.Timestamp(period.start)
            end_ts = pd.Timestamp(period.end)
            fetch_start_ts = start_ts - pd.Timedelta(days=buffer_days)
            window_df = full_df.loc[fetch_start_ts:end_ts]
            if len(window_df) < 10:
                continue

            try:
                result = backtest_fn(window_df, initial_capital=initial_capital, ticker=ticker, **params)
            except ValueError:
                continue

            strat_ret = _rebase_to_window(result.equity_curve, start_ts, end_ts)
            bh_ret = _rebase_to_window(result.buy_hold_curve, start_ts, end_ts)
            if strat_ret is None or bh_ret is None:
                continue

            visible_trades = [
                t for t in result.trades if t.exit_date >= start_ts and t.entry_date <= end_ts
            ]
            wins = sum(1 for t in visible_trades if t.is_win)
            win_rate = (wins / len(visible_trades) * 100) if visible_trades else 0.0
            window_days = (end_ts - start_ts).days
            strat_ann = annualized_return_pct(strat_ret, window_days)

            row = dict(params)
            row.update(
                {
                    "period": period.label,
                    "period_start": str(start_ts.date()),
                    "period_end": str(end_ts.date()),
                    "strategy_return_pct": round(strat_ret, 2),
                    "buy_hold_return_pct": round(bh_ret, 2),
                    "edge_pct": round(strat_ret - bh_ret, 2),
                    "beat_buy_hold": bool(strat_ret > bh_ret),
                    "strategy_annualized_pct": round(strat_ann, 2) if strat_ann is not None else None,
                    "total_trades": len(visible_trades),
                    "win_rate_pct": round(win_rate, 2),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_combos_across_periods(sweep_df: pd.DataFrame, param_keys: List[str]) -> pd.DataFrame:
    """Collapse the per-period sweep rows into one row per parameter
    combo, with how many of the tested periods it beat buy-and-hold in and
    its average/minimum edge -- so "find a combo that's robust across all
    4 windows" is a single sort instead of eyeballing the raw per-period
    table by hand."""
    if sweep_df.empty:
        return sweep_df
    grouped = sweep_df.groupby(param_keys, as_index=False).agg(
        periods_tested=("period", "count"),
        periods_beat_buy_hold=("beat_buy_hold", "sum"),
        avg_edge_pct=("edge_pct", "mean"),
        min_edge_pct=("edge_pct", "min"),
        avg_strategy_return_pct=("strategy_return_pct", "mean"),
        avg_win_rate_pct=("win_rate_pct", "mean"),
        total_trades=("total_trades", "sum"),
    )
    for col in ("avg_edge_pct", "min_edge_pct", "avg_strategy_return_pct", "avg_win_rate_pct"):
        grouped[col] = grouped[col].round(2)
    grouped["beats_buy_hold_every_period"] = grouped["periods_beat_buy_hold"] == grouped["periods_tested"]
    return grouped.sort_values(
        ["beats_buy_hold_every_period", "avg_edge_pct"], ascending=[False, False]
    ).reset_index(drop=True)
