"""Unit tests for the LuxAlgo-inspired Trend Line Breakout (Core) and
Market Flow (Full Package) strategies. Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_utils import to_dataframe
from engine.luxalgo_strategy import (
    _causal_trendline,
    _daily_gap_pct,
    _liquidity_zone_flags,
    _opening_range_levels,
    _pivot_mask,
    backtest_market_flow_full,
    backtest_trendline_breakout_core,
    market_flow_full_recent,
    market_flow_full_signals,
    trendline_breakout_recent,
    trendline_signals,
)
from engine.models import PriceBar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _intraday_df(rows):
    """rows: list of (timestamp, open, high, low, close) tuples. Returns a
    canonical DataFrame with a real intraday DatetimeIndex (unlike PriceBar,
    which only stores a date -- these strategies need time-of-day)."""
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1_000.0] * len(rows),
        },
        index=idx,
    )


def make_flat_intraday_bars(n_days=5, bars_per_day=10, price=100.0):
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2026-02-02") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * b)
            rows.append((ts, price, price, price, price))
    return _intraday_df(rows)


def make_descending_peaks_then_breakout_bars(
    n_flat_days=10, n_breakout_days=5, bars_per_day=26, base=100.0
):
    """n_flat_days of a triangular zigzag with a single, unambiguous daily
    peak (never a tie -- see the module docstring on _pivot_mask rejecting
    ties) that gets slightly *lower* each day -- a real descending
    resistance trend line forms from those peaks. Then n_breakout_days of a
    clean, monotonic rally that eventually closes back up through that
    trend line.

    Hand-verified once with the default pivot_lookback=3/trendline_points=3
    (see the strategy tests below): a resistance trend line forms from the
    zigzag's peaks, and the breakout happens partway through the rally
    phase -- tests below discover the actual trade dynamically rather than
    hardcoding which bar it's on."""
    rows = []
    for d in range(n_flat_days + n_breakout_days):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * b)
            if d < n_flat_days:
                daily_peak = base - d * 0.3
                wave = 1.5 - abs(b - 6) * 0.15 - (b > 6) * 0.01 * (b - 6)
                price = daily_peak + wave + b * 1e-4
            else:
                price = base + (d - (n_flat_days - 1)) * 2.0 + b * 0.05
            rows.append((ts, price, price + 0.3, price - 0.3, price))
    return _intraday_df(rows)


def make_daily_bars(n=60, start_price=100.0, daily_pct=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    prices = start_price * np.cumprod(1 + daily_pct + rng.normal(0, 0.001, n))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 1_000.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------


class TestPivotMask(unittest.TestCase):
    def test_flat_series_never_pivots(self):
        s = pd.Series([100.0] * 20)
        self.assertFalse(_pivot_mask(s, 3, 3, "high").any())
        self.assertFalse(_pivot_mask(s, 3, 3, "low").any())

    def test_single_unambiguous_peak_is_found(self):
        # 0..6 rising, 7 is the strict peak, 8..14 falling -- left/right=3
        # each confirms bar index 7.
        values = [0, 1, 2, 3, 4, 5, 6, 10, 6, 5, 4, 3, 2, 1, 0]
        s = pd.Series(values, dtype=float)
        mask = _pivot_mask(s, 3, 3, "high")
        self.assertEqual(list(mask[mask].index), [7])

    def test_single_unambiguous_trough_is_found(self):
        values = [10, 9, 8, 7, 6, 5, 4, 0, 4, 5, 6, 7, 8, 9, 10]
        s = pd.Series(values, dtype=float)
        mask = _pivot_mask(s, 3, 3, "low")
        self.assertEqual(list(mask[mask].index), [7])

    def test_tied_extreme_is_rejected(self):
        # Two equal maxima -- neither is a *unique* strict extreme within
        # its window, so this must find nothing (this is exactly the
        # symmetric-sine-sample pitfall that motivated the uniqueness
        # check in the first place).
        values = [0, 1, 2, 5, 5, 2, 1, 0]
        s = pd.Series(values, dtype=float)
        mask = _pivot_mask(s, 2, 2, "high")
        self.assertFalse(mask.any())

    def test_last_right_bars_never_confirmed(self):
        # A peak sitting in the last `right` bars can't be confirmed
        # (no bars after it to compare against) -- this is what keeps the
        # mask causal.
        values = [0, 1, 2, 3, 4, 5, 10]
        s = pd.Series(values, dtype=float)
        mask = _pivot_mask(s, 2, 2, "high")
        self.assertFalse(mask.any())


class TestCausalTrendline(unittest.TestCase):
    def test_nan_until_enough_confirmed_pivots(self):
        # Three pivots at positions 2, 6, 10 (mocked directly, bypassing
        # _pivot_mask, to isolate _causal_trendline's own logic); with
        # right=1 and n_points=2, the line should only start at bar 7
        # (the first bar at/after the *second* pivot's confirmation bar
        # 6+1=7).
        n = 15
        values = pd.Series(np.arange(n, dtype=float) * 2.0)
        mask = pd.Series([False] * n)
        mask.iloc[[2, 6, 10]] = True
        line = _causal_trendline(values, mask, right=1, n_points=2)
        self.assertTrue(line.iloc[:7].isna().all())
        self.assertTrue(line.iloc[7:].notna().all())

    def test_projects_a_simple_linear_relationship_exactly(self):
        # Pivot values lie exactly on y = 3x + 1 -- the least-squares fit
        # through them should reproduce that line exactly (no noise).
        n = 20
        pivot_positions = [2, 8, 14]
        values = pd.Series(np.nan, index=range(n))
        for p in pivot_positions:
            values.iloc[p] = 3 * p + 1
        # Fill non-pivot bars with something else entirely -- the fit must
        # only look at the flagged pivot values, not the raw series.
        values = values.fillna(-999.0)
        mask = pd.Series([False] * n)
        mask.iloc[pivot_positions] = True
        line = _causal_trendline(values, mask, right=0, n_points=3)
        last = n - 1
        self.assertAlmostEqual(line.iloc[last], 3 * last + 1, places=6)


class TestOpeningRangeLevels(unittest.TestCase):
    def test_unavailable_on_single_bar_days(self):
        df = make_daily_bars(n=30)
        or_high, or_low = _opening_range_levels(df, minutes=15)
        self.assertEqual(or_high.notna().sum(), 0)
        self.assertEqual(or_low.notna().sum(), 0)

    def test_confirmed_only_after_the_window_closes(self):
        day = pd.Timestamp("2026-03-02")
        rows = [
            (day + pd.Timedelta(minutes=0), 100, 101, 99, 100),
            (day + pd.Timedelta(minutes=15), 100, 105, 98, 101),  # last bar inside the 15-min window
            (day + pd.Timedelta(minutes=30), 101, 102, 100, 101.5),  # first bar after
            (day + pd.Timedelta(minutes=45), 101.5, 103, 101, 102),
        ]
        df = _intraday_df(rows)
        or_high, or_low = _opening_range_levels(df, minutes=20)
        self.assertTrue(pd.isna(or_high.iloc[0]))
        self.assertTrue(pd.isna(or_high.iloc[1]))
        self.assertEqual(or_high.iloc[2], 105.0)
        self.assertEqual(or_low.iloc[2], 98.0)
        self.assertEqual(or_high.iloc[3], 105.0)


class TestDailyGapPct(unittest.TestCase):
    def test_gap_matches_hand_computed_value(self):
        rows = [
            (pd.Timestamp("2026-01-05 09:30"), 100.0, 101, 99, 100.5),
            (pd.Timestamp("2026-01-05 09:45"), 100.5, 101, 100, 100.8),
            (pd.Timestamp("2026-01-06 09:30"), 95.0, 96, 94, 95.5),  # gapped down from 100.8
            (pd.Timestamp("2026-01-06 09:45"), 95.5, 96, 95, 95.8),
        ]
        df = _intraday_df(rows)
        gap = _daily_gap_pct(df)
        self.assertTrue(pd.isna(gap.iloc[0]))
        self.assertTrue(pd.isna(gap.iloc[1]))
        expected = (95.0 - 100.8) / 100.8 * 100.0
        self.assertAlmostEqual(gap.iloc[2], expected, places=6)
        self.assertAlmostEqual(gap.iloc[3], expected, places=6)


class TestLiquidityZoneFlags(unittest.TestCase):
    def test_true_only_near_a_confirmed_level_within_lookback(self):
        n = 12
        close = pd.Series([100.0] * n)
        close.iloc[8] = 110.0  # this bar is "near" the pivot level of 110
        high_mask = pd.Series([False] * n)
        high_mask.iloc[3] = True  # confirmed (with right=1) starting bar 4
        low_mask = pd.Series([False] * n)
        atr = pd.Series([1.0] * n)
        flags = _liquidity_zone_flags(
            close, high_mask, pd.Series([110.0] * n), low_mask, pd.Series([0.0] * n),
            right=1, lookback_bars=20, atr=atr, proximity_atr_mult=1.0,
        )
        self.assertFalse(flags.iloc[0])  # before the pivot is even confirmed
        self.assertFalse(flags.iloc[4])  # confirmed, but close (100) isn't within 1 ATR of 110
        self.assertTrue(flags.iloc[8])  # close is exactly at the level

    def test_expires_outside_the_lookback_window(self):
        n = 30
        close = pd.Series([110.0] * n)
        high_mask = pd.Series([False] * n)
        high_mask.iloc[2] = True
        low_mask = pd.Series([False] * n)
        atr = pd.Series([1.0] * n)
        flags = _liquidity_zone_flags(
            close, high_mask, pd.Series([110.0] * n), low_mask, pd.Series([0.0] * n),
            right=1, lookback_bars=5, atr=atr, proximity_atr_mult=1.0,
        )
        self.assertTrue(flags.iloc[4])  # still within the 5-bar lookback of position 2
        self.assertFalse(flags.iloc[10])  # long expired


# ---------------------------------------------------------------------------
# trendline_signals / trendline_breakout_recent (shared by both strategies)
# ---------------------------------------------------------------------------


class TestTrendlineSignals(unittest.TestCase):
    def test_raises_on_invalid_params(self):
        df = make_daily_bars(n=60)
        with self.assertRaises(ValueError):
            trendline_signals(df, pivot_lookback=0)
        with self.assertRaises(ValueError):
            trendline_signals(df, trendline_points=1)
        with self.assertRaises(ValueError):
            trendline_signals(df, atr_period=0)

    def test_flat_price_never_breaks_out(self):
        df = make_flat_intraday_bars()
        bullish_breakout, *_ = trendline_signals(df, pivot_lookback=2, trendline_points=2, atr_period=3)
        self.assertFalse(bullish_breakout.any())

    def test_breakout_fires_on_the_engineered_fixture(self):
        df = make_descending_peaks_then_breakout_bars()
        bullish_breakout, resistance_line, _support, atr, *_ = trendline_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14
        )
        self.assertTrue(bullish_breakout.any(), "expected the engineered rally to break the descending trend line")
        # Every breakout bar must actually satisfy its own definition:
        # close above the line, prior close at/below the (prior) line.
        prev_close = df["close"].shift(1)
        prev_line = resistance_line.shift(1)
        for i in range(len(df)):
            if bool(bullish_breakout.iloc[i]):
                self.assertGreater(df["close"].iloc[i], resistance_line.iloc[i])
                self.assertLessEqual(prev_close.iloc[i], prev_line.iloc[i])


class TestTrendlineBreakoutRecent(unittest.TestCase):
    def setUp(self):
        self.df = make_descending_peaks_then_breakout_bars()
        bullish_breakout, *_ = trendline_signals(self.df, pivot_lookback=3, trendline_points=3, atr_period=14)
        hits = bullish_breakout[bullish_breakout].index
        self.assertGreater(len(hits), 0, "fixture must produce at least one breakout")
        self.first_hit_pos = self.df.index.get_loc(hits[0])

    def test_true_when_signal_within_lookback(self):
        truncated = self.df.iloc[: self.first_hit_pos + 1]
        self.assertTrue(
            trendline_breakout_recent(truncated, pivot_lookback=3, trendline_points=3, atr_period=14, lookback_days=3)
        )

    def test_false_when_signal_outside_lookback(self):
        extended = self.df.iloc[: self.first_hit_pos + 40]
        self.assertFalse(
            trendline_breakout_recent(extended, pivot_lookback=3, trendline_points=3, atr_period=14, lookback_days=3)
        )

    def test_false_with_insufficient_data(self):
        self.assertFalse(trendline_breakout_recent(make_flat_intraday_bars(n_days=1, bars_per_day=5)))


# ---------------------------------------------------------------------------
# Core: backtest_trendline_breakout_core
# ---------------------------------------------------------------------------


class TestTrendlineBreakoutCoreBacktester(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_trendline_breakout_core(make_flat_intraday_bars(n_days=1, bars_per_day=5))

    def test_raises_on_invalid_take_profit_or_stop(self):
        df = make_descending_peaks_then_breakout_bars()
        with self.assertRaises(ValueError):
            backtest_trendline_breakout_core(df, take_profit_atr_mult=0)
        with self.assertRaises(ValueError):
            backtest_trendline_breakout_core(df, stop_loss_atr_mult=-1)

    def test_flat_price_never_trades(self):
        df = make_flat_intraday_bars(n_days=10, bars_per_day=20)
        result = backtest_trendline_breakout_core(df, pivot_lookback=2, trendline_points=2, atr_period=3, ticker="FLAT")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0, places=6)

    def test_breakout_produces_a_single_take_profit_exit(self):
        df = make_descending_peaks_then_breakout_bars()
        result = backtest_trendline_breakout_core(
            df, pivot_lookback=3, trendline_points=3, atr_period=14,
            take_profit_atr_mult=2.0, stop_loss_atr_mult=1.5, ticker="SYN",
        )
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta["exit_reason"], "take_profit_1")
        self.assertEqual(t.meta["of_legs"], 1)
        self.assertGreater(t.return_pct, 0)

    def test_stop_loss_bounds_the_loss_on_a_reversal(self):
        # Splice a sharp decline right after the breakout entry -- the
        # stop-loss must cap the loss near -stop_loss_atr_mult x ATR, not
        # ride it down to the take-profit that will now never come.
        df = make_descending_peaks_then_breakout_bars()
        bullish_breakout, _res, _sup, atr, *_ = trendline_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14
        )
        hits = bullish_breakout[bullish_breakout].index
        self.assertGreater(len(hits), 0)
        entry_pos = df.index.get_loc(hits[0])

        truncated = df.iloc[: entry_pos + 1].copy()
        entry_price = truncated["close"].iloc[-1]
        last_ts = truncated.index[-1]
        decline_rows = []
        price = entry_price
        for k in range(1, 21):
            price *= 0.97
            ts = last_ts + pd.Timedelta(minutes=15 * k)
            decline_rows.append((ts, price, price + 0.1, price - 0.1, price))
        decline_df = _intraday_df(decline_rows)
        full_df = pd.concat([truncated, decline_df])

        result = backtest_trendline_breakout_core(
            full_df, pivot_lookback=3, trendline_points=3, atr_period=14,
            take_profit_atr_mult=5.0, stop_loss_atr_mult=1.0, ticker="DROP",
        )
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta["exit_reason"], "stop_loss")
        self.assertLess(t.return_pct, 0)
        entry_atr = atr.iloc[entry_pos]
        max_loss_pct = stop_loss_pct = (1.0 * entry_atr) / entry_price * 100
        self.assertGreaterEqual(t.return_pct, -max_loss_pct - 1e-6)

    def test_never_hitting_target_or_stop_holds_to_period_end(self):
        df = make_descending_peaks_then_breakout_bars()
        result = backtest_trendline_breakout_core(
            df, pivot_lookback=3, trendline_points=3, atr_period=14,
            take_profit_atr_mult=100.0, stop_loss_atr_mult=100.0, ticker="HOLD",
        )
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].meta["exit_reason"], "period_end")
        self.assertTrue(result.trades[0].closed_at_period_end)

    def test_summary_reports_strategy_name(self):
        df = make_descending_peaks_then_breakout_bars()
        result = backtest_trendline_breakout_core(df, ticker="SYN")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "trendline_breakout_core")
        self.assertIsNone(s["fast_window"])
        self.assertIsNone(s["slow_window"])


# ---------------------------------------------------------------------------
# Full package: market_flow_full_signals / backtest_market_flow_full
# ---------------------------------------------------------------------------


class TestMarketFlowFullSignals(unittest.TestCase):
    def test_raises_on_invalid_params(self):
        df = make_descending_peaks_then_breakout_bars()
        with self.assertRaises(ValueError):
            market_flow_full_signals(df, ema_period=0)
        with self.assertRaises(ValueError):
            market_flow_full_signals(df, opening_range_minutes=0)
        with self.assertRaises(ValueError):
            market_flow_full_signals(df, liquidity_lookback_bars=0)
        with self.assertRaises(ValueError):
            market_flow_full_signals(df, liquidity_proximity_atr_mult=0)
        with self.assertRaises(ValueError):
            market_flow_full_signals(df, gap_filter_pct=-1)

    def test_full_entry_is_a_subset_of_the_core_breakout(self):
        df = make_descending_peaks_then_breakout_bars()
        core_breakout, *_ = trendline_signals(df, pivot_lookback=3, trendline_points=3, atr_period=14)
        full_entry, _atr = market_flow_full_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
            liquidity_proximity_atr_mult=3.0, gap_filter_pct=5.0,
        )
        # Every full-package entry must also be a raw trend-line breakout
        # (the extra filters can only narrow the set, never widen it).
        self.assertTrue((~full_entry | core_breakout).all())

    def test_opening_range_filter_blocks_full_but_not_core(self):
        # Day 0 has two clean, unique, descending peaks (a real resistance
        # trend line forms around ~101 and gently declines). Day 1 opens
        # with two *tied* high bars at 108 -- ties are never pivots (see
        # TestPivotMask.test_tied_extreme_is_rejected), so this elevates
        # the Opening Range high to 108 without distorting the trend line
        # -- then later bars close at 100-105: above the ~101 resistance
        # line (a real core breakout) but nowhere near the 108 Opening
        # Range high. Hand-verified against trendline_signals /
        # market_flow_full_signals directly before writing this test.
        day0 = pd.Timestamp("2026-04-06")
        day1 = day0 + pd.Timedelta(days=1)
        day0_close = [99.8, 100, 100.5, 101.0, 100.5, 100, 99.5, 99, 99.5, 100, 100.5, 100.9, 100.4, 99.9, 99.6]
        rows = []
        for b, price in enumerate(day0_close):
            ts = day0 + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * b)
            rows.append((ts, price, price + 0.1, price - 0.1, price))
        day1_bars = [
            (0, 100.0, 108.0, 99.5, 100.0),
            (15, 100.0, 108.0, 99.5, 100.2),
            (30, 100.5, 101.0, 100.0, 100.6),
            (45, 102.0, 102.5, 101.5, 102.2),
            (60, 103.5, 104.0, 103.0, 103.7),
            (75, 104.8, 105.3, 104.3, 105.0),
        ]
        for minutes, o, h, l, c in day1_bars:
            ts = day1 + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minutes)
            rows.append((ts, o, h, l, c))
        df = _intraday_df(rows)

        core_breakout, *_ = trendline_signals(df, pivot_lookback=3, trendline_points=2, atr_period=3)
        full_entry, _atr = market_flow_full_signals(
            df, pivot_lookback=3, trendline_points=2, atr_period=3, ema_period=2,
            opening_range_minutes=15, liquidity_proximity_atr_mult=10.0, gap_filter_pct=0.0,
        )
        self.assertTrue(core_breakout.any(), "expected the core trend-line breakout to fire on day 1's close climbing past ~101")
        self.assertFalse(full_entry.any(), "the Opening Range filter should block every one of those breakouts (108 OR high)")

    def test_gap_filter_blocks_entries_on_a_hard_gap_down_day(self):
        df = make_descending_peaks_then_breakout_bars()
        # Force the breakout day's open sharply below the prior day's
        # close, without disturbing the rest of that day's prices.
        breakout_day = df.index.normalize()[-1]
        day_mask = df.index.normalize() == breakout_day
        prior_close = df.loc[~day_mask, "close"].iloc[-1]
        df = df.copy()
        first_idx = df.index[day_mask][0]
        df.loc[first_idx, "open"] = prior_close * 0.90  # a 10% gap down

        blocked_entry, _atr = market_flow_full_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
            liquidity_proximity_atr_mult=3.0, gap_filter_pct=2.0,
        )
        allowed_entry, _atr2 = market_flow_full_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
            liquidity_proximity_atr_mult=3.0, gap_filter_pct=0.0,
        )
        self.assertFalse(blocked_entry.loc[day_mask].any(), "a 2% gap filter should block every entry on a 10%-gap-down day")
        self.assertEqual(
            list(allowed_entry.loc[day_mask]),
            list(trendline_signals(df, pivot_lookback=3, trendline_points=3, atr_period=14)[0].loc[day_mask]),
            "disabling the gap filter (0) should fall back to the plain trend-line breakout for that day",
        )


class TestMarketFlowFullBacktester(unittest.TestCase):
    def test_raises_on_bad_take_profit_ordering(self):
        df = make_descending_peaks_then_breakout_bars()
        with self.assertRaises(ValueError):
            backtest_market_flow_full(df, take_profit_1_atr_mult=2.0, take_profit_2_atr_mult=2.0, take_profit_3_atr_mult=3.0)
        with self.assertRaises(ValueError):
            backtest_market_flow_full(df, take_profit_1_atr_mult=3.0, take_profit_2_atr_mult=2.0, take_profit_3_atr_mult=1.0)

    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_market_flow_full(make_flat_intraday_bars(n_days=1, bars_per_day=5))

    def test_flat_price_never_trades(self):
        df = make_flat_intraday_bars(n_days=10, bars_per_day=20)
        result = backtest_market_flow_full(
            df, pivot_lookback=2, trendline_points=2, atr_period=3, ema_period=3,
            liquidity_lookback_bars=10, ticker="FLAT",
        )
        self.assertEqual(result.total_trades, 0)

    def test_breakout_produces_three_scaled_take_profit_legs(self):
        df = make_descending_peaks_then_breakout_bars()
        result = backtest_market_flow_full(
            df, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
            opening_range_minutes=15, liquidity_lookback_bars=50, liquidity_proximity_atr_mult=3.0,
            gap_filter_pct=5.0, take_profit_1_atr_mult=1.0, take_profit_2_atr_mult=2.0,
            take_profit_3_atr_mult=3.0, stop_loss_atr_mult=2.0, ticker="SYN",
        )
        self.assertEqual(result.total_trades, 3)
        reasons = sorted(t.meta["exit_reason"] for t in result.trades)
        self.assertEqual(reasons, ["take_profit_1", "take_profit_2", "take_profit_3"])
        for t in result.trades:
            self.assertEqual(t.meta["of_legs"], 3)
            self.assertEqual(t.entry_date, result.trades[0].entry_date)  # all 3 legs enter together
        # Further targets should be at least as profitable per-leg as
        # nearer ones, since ATR-at-entry is shared and mults are ordered.
        by_leg = {t.meta["leg"]: t.return_pct for t in result.trades}
        self.assertLessEqual(by_leg[1], by_leg[2])
        self.assertLessEqual(by_leg[2], by_leg[3])

    def test_runs_on_daily_bars_with_opening_range_unavailable(self):
        # A daily-granularity version of the same shape as
        # make_descending_peaks_then_breakout_bars (one bar per day: two
        # clean, unique, descending peaks, then a breakout rally) --
        # confirms the strategy still runs (and can still trade) when the
        # Opening Range simply can't be computed, per the module's
        # documented graceful degrade. Hand-verified against
        # trendline_signals/market_flow_full_signals directly.
        day0_close = [99.8, 100, 100.5, 101.0, 100.5, 100, 99.5, 99, 99.5, 100, 100.5, 100.9, 100.4, 99.9, 99.6]
        rally = [99.6 + k * 1.0 for k in range(1, 11)]
        rows = []
        for d, price in enumerate(day0_close + rally):
            day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
            rows.append((day, price, price + 0.5, price - 0.5, price))
        df = _intraday_df(rows)

        or_high, _ = _opening_range_levels(df, 15)
        self.assertEqual(or_high.notna().sum(), 0)

        result = backtest_market_flow_full(
            df, pivot_lookback=2, trendline_points=2, atr_period=3, ema_period=3,
            liquidity_lookback_bars=5, liquidity_proximity_atr_mult=5.0, gap_filter_pct=0.0,
            stop_loss_atr_mult=5.0, ticker="DAILY",
        )
        self.assertGreaterEqual(result.total_trades, 1)

    def test_summary_reports_strategy_name(self):
        df = make_descending_peaks_then_breakout_bars()
        result = backtest_market_flow_full(df, ticker="SYN")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "market_flow_full")


class TestMarketFlowFullRecent(unittest.TestCase):
    def test_true_when_signal_within_lookback_false_with_insufficient_data(self):
        df = make_descending_peaks_then_breakout_bars()
        full_entry, _atr = market_flow_full_signals(
            df, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
            liquidity_proximity_atr_mult=3.0, gap_filter_pct=5.0,
        )
        hits = full_entry[full_entry].index
        self.assertGreater(len(hits), 0)
        first_hit_pos = df.index.get_loc(hits[0])
        truncated = df.iloc[: first_hit_pos + 1]
        self.assertTrue(
            market_flow_full_recent(
                truncated, pivot_lookback=3, trendline_points=3, atr_period=14, ema_period=10,
                liquidity_proximity_atr_mult=3.0, gap_filter_pct=5.0, lookback_days=3,
            )
        )
        self.assertFalse(market_flow_full_recent(make_flat_intraday_bars(n_days=1, bars_per_day=5)))


if __name__ == "__main__":
    unittest.main()
