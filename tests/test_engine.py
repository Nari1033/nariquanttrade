"""Unit tests for the SMA crossover scanner + backtester engine.

Uses only the stdlib `unittest` (no pytest) since this environment has no
package-index access. Run with:

    python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtester import backtest_price_sma_crossover, backtest_sma_crossover
from engine.data_utils import to_dataframe
from engine.indicators import rsi, sma
from engine.models import PriceBar
from engine.scanner import (
    crossover_series,
    death_cross_recent,
    golden_cross_recent,
    price_cross_sma_recent,
    scan_universe,
)


def make_bars(closes, start=date(2024, 1, 1)):
    """Build a list of PriceBar from a list of closes, one per consecutive
    calendar day, with open=high=low=close and a fixed volume."""
    bars = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=c, high=c, low=c, close=c, volume=1_000))
    return bars


def golden_death_bars():
    # SMA1 (=close): [10, 10, 20, 10, 10]
    # SMA2 (2-day avg): [NaN, 10, 15, 15, 10]
    # diff = SMA1-SMA2:  [NaN, 0, 5, -5, 0]
    # -> golden cross on day 3 (diff turns >0 from <=0)
    # -> death cross on day 4 (diff turns <0 from >=0)
    return make_bars([10, 10, 20, 10, 10])


class TestDataUtils(unittest.TestCase):
    def test_to_dataframe_accepts_pricebars_and_dicts_identically(self):
        bars = make_bars([10, 11, 12])
        dicts = [
            {"date": "2024-01-01", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1000},
            {"date": "2024-01-02", "open": 11, "high": 11, "low": 11, "close": 11, "volume": 1000},
            {"date": "2024-01-03", "open": 12, "high": 12, "low": 12, "close": 12, "volume": 1000},
        ]
        df_bars = to_dataframe(bars)
        df_dicts = to_dataframe(dicts)
        pd.testing.assert_frame_equal(df_bars, df_dicts)
        self.assertEqual(list(df_bars["close"]), [10, 11, 12])
        self.assertTrue(df_bars.index.is_monotonic_increasing)

    def test_to_dataframe_sorts_out_of_order_input(self):
        bars = make_bars([10, 11, 12])
        shuffled = [bars[2], bars[0], bars[1]]
        df = to_dataframe(shuffled)
        self.assertEqual(list(df["close"]), [10, 11, 12])

    def test_to_dataframe_missing_column_raises(self):
        with self.assertRaises(ValueError):
            to_dataframe([{"date": "2024-01-01", "open": 1, "high": 1, "low": 1}])  # no close/volume

    def test_to_dataframe_drops_bars_with_missing_ohlc(self):
        # Mirrors a live source (e.g. yfinance) returning a NaN row for the
        # most recent/in-progress session -- that bar should be dropped
        # rather than silently propagating NaN into every downstream
        # calculation (see the "nan%" bug this guards against).
        import math

        dicts = [
            {"date": "2024-01-01", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1000},
            {"date": "2024-01-02", "open": 11, "high": 11, "low": 11, "close": 11, "volume": 1000},
            # Today's still-forming bar, as yfinance sometimes reports it.
            {"date": "2024-01-03", "open": math.nan, "high": math.nan, "low": math.nan, "close": math.nan, "volume": 0},
        ]
        df = to_dataframe(dicts)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["close"]), [10, 11])
        self.assertFalse(df["close"].isna().any())

    def test_to_dataframe_keeps_bar_with_only_missing_volume(self):
        import math

        dicts = [
            {"date": "2024-01-01", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1000},
            {"date": "2024-01-02", "open": 11, "high": 11, "low": 11, "close": 11.5, "volume": math.nan},
        ]
        df = to_dataframe(dicts)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["volume"].iloc[1], 0.0)
        self.assertAlmostEqual(df["close"].iloc[1], 11.5)


class TestSMA(unittest.TestCase):
    def test_sma_matches_manual_rolling_average(self):
        s = pd.Series([1, 2, 3, 4, 5, 6])
        result = sma(s, window=3)
        self.assertEqual(result.isna().tolist(), [True, True, False, False, False, False])
        self.assertAlmostEqual(result.iloc[2], (1 + 2 + 3) / 3)
        self.assertAlmostEqual(result.iloc[5], (4 + 5 + 6) / 3)

    def test_sma_no_partial_window_leakage(self):
        # A 5-window SMA must stay NaN for the first 4 bars, never averaging
        # over fewer than 5 observations.
        s = pd.Series([100.0] * 4 + [1_000_000.0])
        result = sma(s, window=5)
        self.assertTrue(result.iloc[:4].isna().all())
        self.assertAlmostEqual(result.iloc[4], (100 * 4 + 1_000_000) / 5)


class TestRSI(unittest.TestCase):
    def test_nan_during_warmup_then_valid(self):
        s = pd.Series(range(1, 31), dtype=float)
        result = rsi(s, period=14)
        # NaN for the first `period` bars (indices 0..13), valid from index
        # 14 on -- .diff() itself produces one leading NaN, and
        # min_periods=period on the smoothed gain/loss series holds the
        # NaN gate open until `period` valid observations have accumulated.
        self.assertTrue(result.iloc[:14].isna().all())
        self.assertFalse(result.iloc[14:].isna().any())

    def test_pure_uptrend_is_100(self):
        # No down bars at all -> average loss stays exactly 0 the whole
        # way -> RSI = 100 (not just "high"), deterministically.
        s = pd.Series(range(1, 31), dtype=float)
        result = rsi(s, period=14)
        self.assertTrue((result.iloc[14:] == 100.0).all())

    def test_pure_downtrend_is_0(self):
        s = pd.Series(range(30, 0, -1), dtype=float)
        result = rsi(s, period=14)
        self.assertTrue((result.iloc[14:] == 0.0).all())

    def test_flat_price_is_50(self):
        # Neither gains nor losses -> no directional information -> 50,
        # not a NaN/inf from a 0/0 division.
        s = pd.Series([100.0] * 30)
        result = rsi(s, period=14)
        self.assertTrue((result.iloc[14:] == 50.0).all())

    def test_raises_on_nonpositive_period(self):
        s = pd.Series(range(1, 31), dtype=float)
        with self.assertRaises(ValueError):
            rsi(s, period=0)

    def test_oscillating_series_stays_within_bounds(self):
        # A real (non-degenerate) mixed up/down series should never escape
        # [0, 100], and should show meaningful variation once warmed up.
        import math

        s = pd.Series([100 + 10 * math.sin(i / 3.0) for i in range(60)])
        result = rsi(s, period=14).dropna()
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 100).all())
        self.assertGreater(result.max() - result.min(), 10)


class TestCrossoverAndScanner(unittest.TestCase):
    def test_crossover_series_detects_golden_and_death_cross(self):
        df = to_dataframe(golden_death_bars())
        fast = sma(df["close"], 1)
        slow = sma(df["close"], 2)
        cross = crossover_series(fast, slow)
        self.assertEqual(cross.tolist(), [0, 0, 1, -1, 0])

    def test_golden_cross_recent_true_within_lookback(self):
        bars = golden_death_bars()
        # Golden cross happened on day 3 of 5 -> 2 trading days before the last bar.
        self.assertTrue(golden_cross_recent(bars, fast_window=1, slow_window=2, lookback_days=3))

    def test_golden_cross_recent_false_outside_lookback(self):
        bars = golden_death_bars()
        # Only the most recent day is in scope; golden cross (day 3) is not in it.
        self.assertFalse(golden_cross_recent(bars, fast_window=1, slow_window=2, lookback_days=1))

    def test_death_cross_recent_true_within_lookback(self):
        bars = golden_death_bars()
        self.assertTrue(death_cross_recent(bars, fast_window=1, slow_window=2, lookback_days=3))

    def test_price_cross_sma_recent_above_and_below(self):
        bars = golden_death_bars()
        self.assertTrue(
            price_cross_sma_recent(bars, sma_window=2, lookback_days=3, direction="above")
        )
        self.assertTrue(
            price_cross_sma_recent(bars, sma_window=2, lookback_days=3, direction="below")
        )

    def test_golden_cross_recent_false_when_no_cross_ever_happens(self):
        flat = make_bars([10] * 10)
        self.assertFalse(golden_cross_recent(flat, fast_window=1, slow_window=2, lookback_days=3))

    def test_scan_universe_returns_only_matching_tickers(self):
        golden = golden_death_bars()
        flat = make_bars([10] * 10)
        universe = {"GOLD": golden, "FLAT": flat}
        matches = scan_universe(
            universe,
            strategy_fn=golden_cross_recent,
            fast_window=1,
            slow_window=2,
            lookback_days=3,
        )
        self.assertEqual(matches, ["GOLD"])

    def test_scan_universe_skips_tickers_with_insufficient_data(self):
        too_short = make_bars([10, 11])
        universe = {"SHORT": too_short}
        # slow_window=200 can never be satisfied by 2 bars; should not raise.
        matches = scan_universe(universe, fast_window=50, slow_window=200)
        self.assertEqual(matches, [])


class TestBacktester(unittest.TestCase):
    def test_backtest_hand_verified_single_trade(self):
        bars = golden_death_bars()  # closes: 10, 10, 20, 10, 10
        result = backtest_sma_crossover(
            bars, fast_window=1, slow_window=2, initial_capital=10_000.0, ticker="TEST"
        )

        self.assertEqual(result.total_trades, 1)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_price, 20.0)  # golden cross day (day 3) close
        self.assertAlmostEqual(trade.exit_price, 10.0)  # death cross day (day 4) close
        self.assertAlmostEqual(trade.return_pct, -50.0)
        self.assertFalse(trade.is_win)
        self.assertFalse(trade.closed_at_period_end)

        # Strategy: 10,000 -> bought at 20 -> sold at 10 -> 5,000 (-50%).
        self.assertAlmostEqual(result.strategy_return_pct, -50.0)
        self.assertAlmostEqual(result.win_rate_pct, 0.0)

        # Buy & hold over the whole period: first close 10 -> last close 10 -> 0%.
        self.assertAlmostEqual(result.buy_hold_return_pct, 0.0)

        self.assertAlmostEqual(result.equity_curve.iloc[-1], 5_000.0)
        self.assertAlmostEqual(result.buy_hold_curve.iloc[-1], 10_000.0)

    def test_backtest_open_position_marked_to_market_at_period_end(self):
        # Golden cross with no subsequent death cross before data ends.
        bars = make_bars([10, 10, 20, 30, 40])
        # SMA1: 10,10,20,30,40 ; SMA2: NaN,10,15,25,35 ; diff: NaN,0,5,5,5
        # golden cross only on day 3 (diff 0 -> 5); never crosses back down.
        result = backtest_sma_crossover(bars, fast_window=1, slow_window=2, initial_capital=1_000.0)

        self.assertEqual(result.total_trades, 1)
        trade = result.trades[0]
        self.assertTrue(trade.closed_at_period_end)
        self.assertAlmostEqual(trade.entry_price, 20.0)
        self.assertAlmostEqual(trade.exit_price, 40.0)  # marked to market on last bar
        self.assertTrue(trade.is_win)
        self.assertAlmostEqual(result.strategy_return_pct, (40 - 20) / 20 * 100)

    def test_backtest_no_signals_means_no_trades_and_zero_strategy_return(self):
        flat = make_bars([10] * 10)
        result = backtest_sma_crossover(flat, fast_window=1, slow_window=2, initial_capital=1_000.0)
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.win_rate_pct, 0.0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0)
        self.assertAlmostEqual(result.buy_hold_return_pct, 0.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 1_000.0)

    def test_backtest_win_rate_across_multiple_trades(self):
        # Two full round trips: one loser, one flat (not a strict win).
        # Prices: 10,10,20,10,10,20,30,20,10,10
        # trade1: enter@20(day3) exit@10(day4)  -> loss
        # trade2: enter@20(day6) exit@20(day8) -> flat (0%, not a "win")
        closes = [10, 10, 20, 10, 10, 20, 30, 20, 10, 10]
        bars = make_bars(closes)
        result = backtest_sma_crossover(bars, fast_window=1, slow_window=2, initial_capital=1_000.0)
        self.assertEqual(result.total_trades, 2)
        self.assertAlmostEqual(result.trades[0].return_pct, -50.0)
        self.assertAlmostEqual(result.trades[1].return_pct, 0.0)
        self.assertAlmostEqual(result.win_rate_pct, 0.0)  # neither trade was strictly profitable

    def test_backtest_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_sma_crossover(make_bars([10]), fast_window=1, slow_window=2)

    def test_backtest_summary_is_json_friendly(self):
        bars = golden_death_bars()
        result = backtest_sma_crossover(bars, fast_window=1, slow_window=2, ticker="TEST")
        s = result.summary()
        self.assertEqual(s["ticker"], "TEST")
        self.assertEqual(s["strategy_name"], "golden_cross")
        self.assertIsInstance(s["strategy_return_pct"], float)
        self.assertIsInstance(s["total_trades"], int)


class TestPriceSmaBacktester(unittest.TestCase):
    """The 'price crosses its own SMA' backtester. Reuses golden_death_bars
    (closes: 10, 10, 20, 10, 10) with sma_window=2, which makes 'price'
    stand in for the same fast=1/slow=2 crossover already hand-verified in
    TestBacktester -- so these numbers should match exactly."""

    def test_matches_hand_verified_single_trade(self):
        bars = golden_death_bars()
        result = backtest_price_sma_crossover(
            bars, sma_window=2, initial_capital=10_000.0, ticker="TEST"
        )

        self.assertEqual(result.strategy_name, "price_cross_sma")
        self.assertIsNone(result.fast_window)
        self.assertEqual(result.slow_window, 2)

        self.assertEqual(result.total_trades, 1)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_price, 20.0)  # crosses above SMA2 on day 3
        self.assertAlmostEqual(trade.exit_price, 10.0)  # crosses below SMA2 on day 4
        self.assertAlmostEqual(trade.return_pct, -50.0)
        self.assertFalse(trade.is_win)

        self.assertAlmostEqual(result.strategy_return_pct, -50.0)
        self.assertAlmostEqual(result.buy_hold_return_pct, 0.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 5_000.0)

    def test_no_signals_means_no_trades(self):
        flat = make_bars([10] * 10)
        result = backtest_price_sma_crossover(flat, sma_window=2, initial_capital=1_000.0)
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 1_000.0)

    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_price_sma_crossover(make_bars([10]), sma_window=2)

    def test_summary_reports_null_fast_window(self):
        bars = golden_death_bars()
        result = backtest_price_sma_crossover(bars, sma_window=2, ticker="TEST")
        s = result.summary()
        self.assertIsNone(s["fast_window"])
        self.assertEqual(s["slow_window"], 2)
        self.assertEqual(s["strategy_name"], "price_cross_sma")


if __name__ == "__main__":
    unittest.main()
