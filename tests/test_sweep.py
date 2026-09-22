"""Unit tests for engine.sweep (parameter-grid sweeps across multiple
date windows). Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import math
import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import PriceBar
from engine.options_backtester import backtest_bull_put_spread
from engine.sweep import SweepPeriod, generate_param_combos, summarize_combos_across_periods, sweep_strategy


def make_bars(closes, start=date(2020, 1, 1)):
    bars = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=c, high=c, low=c, close=c, volume=1_000))
    return bars


def make_series(n, start_price, drift, seed):
    import random

    rng = random.Random(seed)
    price = start_price
    closes = []
    for _ in range(n):
        price *= math.exp(rng.gauss(drift, 0.014))
        closes.append(price)
    return closes


class TestGenerateParamCombos(unittest.TestCase):
    def test_small_grid_returns_full_cartesian_product(self):
        grid = {"a": [1, 2], "b": [10, 20, 30]}
        combos = generate_param_combos(grid, max_combos=100)
        self.assertEqual(len(combos), 6)
        seen = {(c["a"], c["b"]) for c in combos}
        self.assertEqual(len(seen), 6)

    def test_large_grid_is_randomly_sampled_to_max_combos(self):
        grid = {"a": list(range(10)), "b": list(range(10)), "c": list(range(10))}
        combos = generate_param_combos(grid, max_combos=25, seed=1)
        self.assertEqual(len(combos), 25)
        # No duplicate combos in the sample.
        seen = {tuple(sorted(c.items())) for c in combos}
        self.assertEqual(len(seen), 25)

    def test_same_seed_is_deterministic(self):
        grid = {"a": list(range(20)), "b": list(range(20))}
        c1 = generate_param_combos(grid, max_combos=15, seed=99)
        c2 = generate_param_combos(grid, max_combos=15, seed=99)
        self.assertEqual(c1, c2)

    def test_empty_grid_returns_one_empty_combo(self):
        self.assertEqual(generate_param_combos({}), [{}])


class TestSweepStrategy(unittest.TestCase):
    def _buffer_fn(self, p):
        return max(int(p["trend_sma_window"] * 1.3), 5) + 10

    def test_returns_expected_columns_and_edge_matches_returns(self):
        closes = make_series(n=700, start_price=100.0, drift=0.0004, seed=5)
        bars = make_bars(closes)
        periods = [
            SweepPeriod("early", closes and date(2020, 6, 1), date(2021, 1, 1)),
            SweepPeriod("late", date(2021, 6, 1), date(2021, 12, 1)),
        ]
        grid = {
            "dte_entry": [30],
            "short_delta": [0.30],
            "spread_width_pct": [3.0],
            "profit_target_pct": [50.0],
            "stop_loss_pct": [100.0],
            "trend_sma_window": [100],
        }
        df = sweep_strategy(bars, backtest_bull_put_spread, grid, periods, self._buffer_fn)
        expected_cols = {
            "dte_entry", "short_delta", "spread_width_pct", "profit_target_pct",
            "stop_loss_pct", "trend_sma_window", "period", "period_start", "period_end",
            "strategy_return_pct", "buy_hold_return_pct", "edge_pct", "beat_buy_hold",
            "strategy_annualized_pct", "total_trades", "win_rate_pct",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        for _, row in df.iterrows():
            # All three values are independently rounded to 2dp, so their
            # relationship can be off by a cent's worth of rounding error.
            self.assertAlmostEqual(
                row["edge_pct"], row["strategy_return_pct"] - row["buy_hold_return_pct"], places=1
            )
            self.assertEqual(row["beat_buy_hold"], row["strategy_return_pct"] > row["buy_hold_return_pct"])

    def test_declining_market_lets_capped_spread_beat_buy_and_hold(self):
        # A relentlessly declining underlying: buy-and-hold loses a lot,
        # while a bull-put-spread that only trades in the (nonexistent)
        # uptrend never enters a position at all -- flat 0% beats a large
        # negative buy-and-hold return every time. This is the mechanism
        # by which the sweep can find "beat buy & hold" combos even though
        # the strategy itself never trades.
        closes = [100.0 * math.exp(-0.0015 * i) for i in range(700)]
        bars = make_bars(closes)
        periods = [SweepPeriod("decline", date(2020, 6, 1), date(2021, 6, 1))]
        grid = {
            "dte_entry": [30], "short_delta": [0.30], "spread_width_pct": [3.0],
            "profit_target_pct": [50.0], "stop_loss_pct": [100.0], "trend_sma_window": [100],
        }
        df = sweep_strategy(bars, backtest_bull_put_spread, grid, periods, self._buffer_fn)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertLess(row["buy_hold_return_pct"], -10)
        self.assertAlmostEqual(row["strategy_return_pct"], 0.0, places=1)
        self.assertTrue(row["beat_buy_hold"])

    def test_too_little_data_in_window_is_skipped_not_raised(self):
        closes = make_series(n=50, start_price=100.0, drift=0.0, seed=3)
        bars = make_bars(closes)
        periods = [SweepPeriod("way too short", date(2020, 1, 1), date(2020, 1, 5))]
        grid = {
            "dte_entry": [30], "short_delta": [0.30], "spread_width_pct": [3.0],
            "profit_target_pct": [50.0], "stop_loss_pct": [100.0], "trend_sma_window": [200],
        }
        df = sweep_strategy(bars, backtest_bull_put_spread, grid, periods, self._buffer_fn)
        self.assertEqual(len(df), 0)


class TestSummarizeCombosAcrossPeriods(unittest.TestCase):
    def test_aggregates_and_flags_robust_combo(self):
        import pandas as pd

        rows = [
            # combo A beats buy & hold in both periods -> robust
            {"x": 1, "period": "p1", "edge_pct": 5.0, "beat_buy_hold": True,
             "strategy_return_pct": 5.0, "win_rate_pct": 60.0, "total_trades": 2},
            {"x": 1, "period": "p2", "edge_pct": 3.0, "beat_buy_hold": True,
             "strategy_return_pct": 3.0, "win_rate_pct": 60.0, "total_trades": 2},
            # combo B only beats it in one of the two periods -> not robust
            {"x": 2, "period": "p1", "edge_pct": 10.0, "beat_buy_hold": True,
             "strategy_return_pct": 10.0, "win_rate_pct": 80.0, "total_trades": 3},
            {"x": 2, "period": "p2", "edge_pct": -20.0, "beat_buy_hold": False,
             "strategy_return_pct": -20.0, "win_rate_pct": 20.0, "total_trades": 3},
        ]
        df = pd.DataFrame(rows)
        summary = summarize_combos_across_periods(df, ["x"])

        self.assertEqual(len(summary), 2)
        row_a = summary[summary["x"] == 1].iloc[0]
        row_b = summary[summary["x"] == 2].iloc[0]

        self.assertTrue(row_a["beats_buy_hold_every_period"])
        self.assertFalse(row_b["beats_buy_hold_every_period"])
        self.assertAlmostEqual(row_a["avg_edge_pct"], 4.0, places=6)
        self.assertAlmostEqual(row_b["avg_edge_pct"], -5.0, places=6)
        # Robust combos sort first regardless of raw average edge.
        self.assertEqual(summary.iloc[0]["x"], 1)

    def test_empty_input_returns_empty_output(self):
        import pandas as pd

        empty = pd.DataFrame(columns=["x", "period", "edge_pct", "beat_buy_hold"])
        summary = summarize_combos_across_periods(empty, ["x"])
        self.assertTrue(summary.empty)


if __name__ == "__main__":
    unittest.main()
