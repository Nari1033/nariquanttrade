"""Unit tests for the Modified Wheel Strategy: the plain Wheel plus a
down-market filter that skips opening new cash-secured puts (never
covered calls) while close is below its trend SMA AND RSI is weak.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import math
import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_utils import to_dataframe
from engine.models import PriceBar
from engine.modified_wheel import backtest_modified_wheel_strategy, bearish_filter_signal
from engine.wheel import backtest_wheel_strategy


def make_bars(closes, start=date(2020, 1, 1)):
    bars = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=c, high=c * 1.002, low=c * 0.998, close=c, volume=1_000))
    return bars


def make_mild_oscillation(n, mid=100.0, amp=3.0, period=25):
    return [mid + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def make_gentle_decline_closes():
    """60 flat/oscillating warm-up bars (enough for a short SMA/RSI/vol
    window to warm up and establish a flat state), then an 80-bar gentle,
    steady ~-21% decline -- shallow enough per-day that a very deep-OTM
    put mostly still expires worthless for a while, but persistent enough
    to eventually push price below even a deep strike (an assignment) and
    to keep the down-market filter flagged for a long stretch once it
    trips."""
    closes = make_mild_oscillation(60, mid=100.0, amp=1.0)
    price = closes[-1]
    for _ in range(80):
        price *= 0.997
        closes.append(price)
    return closes


class TestBearishFilterSignal(unittest.TestCase):
    def test_false_during_warmup(self):
        df = to_dataframe(make_bars(make_mild_oscillation(50)))
        signal = bearish_filter_signal(df["close"], trend_sma_window=20, rsi_period=14, rsi_threshold=40.0)
        # SMA(20)/RSI(14) are both warmed up well before bar 50, but the
        # first ~20 bars (before the SMA has enough history) must never
        # be flagged True regardless of price action.
        self.assertFalse(bool(signal.iloc[:19].any()))

    def test_true_only_when_both_conditions_hold(self):
        closes = make_gentle_decline_closes()
        df = to_dataframe(make_bars(closes))
        signal = bearish_filter_signal(df["close"], trend_sma_window=20, rsi_period=14, rsi_threshold=40.0)
        from engine.indicators import rsi as rsi_indicator, sma as sma_indicator

        sma20 = sma_indicator(df["close"], 20)
        rsi14 = rsi_indicator(df["close"], 14)
        expected = (df["close"] < sma20).fillna(False) & (rsi14 < 40.0).fillna(False)
        self.assertTrue((signal == expected).all())
        self.assertGreater(signal.sum(), 0, "the engineered decline should trip the filter at least once")

    def test_raises_on_invalid_params(self):
        df = to_dataframe(make_bars(make_mild_oscillation(50)))
        with self.assertRaises(ValueError):
            bearish_filter_signal(df["close"], trend_sma_window=1)
        with self.assertRaises(ValueError):
            bearish_filter_signal(df["close"], rsi_period=0)


class TestModifiedWheelBacktester(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_modified_wheel_strategy(make_bars([100.0] * 10))

    def test_rsi_threshold_zero_disables_filter_and_matches_plain_wheel(self):
        # RSI can never be negative, so rsi_threshold=0.0 makes the
        # bearish condition permanently False -- the modified wheel must
        # then be byte-for-byte identical to the plain wheel.
        bars = make_bars(make_gentle_decline_closes())
        baseline = backtest_wheel_strategy(bars, ticker="OSC", put_delta=0.05, dte_days=14)
        disabled = backtest_modified_wheel_strategy(
            bars, ticker="OSC", put_delta=0.05, dte_days=14,
            trend_sma_window=20, rsi_period=14, rsi_threshold=0.0,
        )
        self.assertEqual(
            [(t.entry_date, t.meta["leg"], t.meta["exit_reason"]) for t in baseline.trades],
            [(t.entry_date, t.meta["leg"], t.meta["exit_reason"]) for t in disabled.trades],
        )
        self.assertAlmostEqual(baseline.strategy_return_pct, disabled.strategy_return_pct, places=6)

    def test_filter_never_opens_a_new_put_on_a_flagged_bearish_day(self):
        bars = make_bars(make_gentle_decline_closes())
        df = to_dataframe(bars)
        signal = bearish_filter_signal(df["close"], trend_sma_window=20, rsi_period=14, rsi_threshold=40.0)
        result = backtest_modified_wheel_strategy(
            bars, ticker="DECLINE", put_delta=0.05, call_delta=0.20, dte_days=14, vol_window=20,
            trend_sma_window=20, rsi_period=14, rsi_threshold=40.0,
        )
        put_trades = [t for t in result.trades if t.meta["leg"] == "put"]
        self.assertGreater(len(put_trades), 0, "scenario should still sell at least one put")
        for t in put_trades:
            self.assertFalse(
                bool(signal.loc[t.entry_date]),
                f"a new put was opened on {t.entry_date.date()}, which the filter should have blocked",
            )

    def test_filter_does_not_block_covered_call_selling_on_a_bearish_day(self):
        # Once assigned, selling covered calls against shares already
        # held must continue even while the down-market filter is active
        # -- only *new put* entries are gated (see module docstring).
        bars = make_bars(make_gentle_decline_closes())
        df = to_dataframe(bars)
        signal = bearish_filter_signal(df["close"], trend_sma_window=20, rsi_period=14, rsi_threshold=40.0)
        result = backtest_modified_wheel_strategy(
            bars, ticker="DECLINE", put_delta=0.05, call_delta=0.20, dte_days=14, vol_window=20,
            trend_sma_window=20, rsi_period=14, rsi_threshold=40.0,
        )
        call_trades = [t for t in result.trades if t.meta["leg"] == "call"]
        self.assertGreater(len(call_trades), 0, "scenario should get assigned and then sell at least one call")
        bearish_calls = [t for t in call_trades if bool(signal.loc[t.entry_date])]
        self.assertGreater(
            len(bearish_calls), 0,
            "expected at least one covered call sold on a day the filter flagged bearish, "
            "proving calls are never gated by it",
        )

    def test_filter_can_only_delay_or_remove_put_entries_not_add_them(self):
        # Same invariant as the plain wheel's day-of-month filter test:
        # a gating filter can only reduce or defer opportunities.
        bars = make_bars(make_gentle_decline_closes())
        baseline = backtest_wheel_strategy(bars, ticker="DECLINE", put_delta=0.05, dte_days=14)
        filtered = backtest_modified_wheel_strategy(
            bars, ticker="DECLINE", put_delta=0.05, dte_days=14,
            trend_sma_window=20, rsi_period=14, rsi_threshold=40.0,
        )
        self.assertLessEqual(filtered.total_trades, baseline.total_trades)

    def test_entry_day_of_month_zero_matches_unfiltered_baseline(self):
        bars = make_bars(make_mild_oscillation(300))
        baseline = backtest_modified_wheel_strategy(bars, ticker="OSC")
        filtered = backtest_modified_wheel_strategy(bars, ticker="OSC", entry_day_of_month=0)
        self.assertEqual(
            [t.entry_date for t in baseline.trades], [t.entry_date for t in filtered.trades]
        )

    def test_summary_reports_modified_wheel_strategy_name_and_no_sma_windows(self):
        bars = make_bars(make_mild_oscillation(300))
        result = backtest_modified_wheel_strategy(bars, ticker="OSC")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "modified_wheel")
        self.assertIsNone(s["fast_window"])
        self.assertIsNone(s["slow_window"])

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        bars = make_bars(make_gentle_decline_closes())
        result = backtest_modified_wheel_strategy(
            bars, ticker="DECLINE", put_delta=0.05, dte_days=14,
            trend_sma_window=20, rsi_period=14, rsi_threshold=40.0,
        )
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
