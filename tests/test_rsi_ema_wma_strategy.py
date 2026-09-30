"""Unit tests for the RSI(9)+EMA(3)+WMA(21) strategy (scan + backtest).
Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_utils import to_dataframe
from engine.indicators import ema as ema_indicator
from engine.indicators import rsi as rsi_indicator
from engine.indicators import wma as wma_indicator
from engine.models import PriceBar
from engine.rsi_ema_wma_strategy import (
    backtest_rsi_ema_wma,
    rsi_ema_wma_bullish_recent,
    rsi_ema_wma_lines,
    rsi_ema_wma_signals,
)


def make_bars(closes, start=date(2020, 1, 1)):
    """One bar per consecutive calendar day; open/high/low all collapse to
    the bar's own close/prev-close range (fine for indicator math, which
    only reads `close`)."""
    bars = []
    prev = closes[0]
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(
            PriceBar(date=d, open=prev, high=max(prev, c), low=min(prev, c), close=c, volume=1_000)
        )
        prev = c
    return bars


def make_flat_bars(n=60, price=100.0, start=date(2020, 1, 1)):
    return make_bars([price] * n, start=start)


def make_wavy_bars(n=150, base=100.0, start=date(2020, 1, 1)):
    """A slow sine-wave-ish price path with small noise -- enough
    oscillation in RSI to reliably run through full bullish/bearish
    three-line stacks (and past the 50 line with a wide margin) multiple
    times, without depending on a specific random seed's exact values in
    assertions (tests below only check structural properties)."""
    import math

    closes = []
    price = base
    for i in range(n):
        price += math.sin(i / 8.0) * 1.2 + math.sin(i / 3.0) * 0.3
        closes.append(price)
    return make_bars(closes, start=start)


class TestRsiEmaWmaLines(unittest.TestCase):
    def test_lines_match_indicators_directly(self):
        bars = make_wavy_bars(120)
        df = to_dataframe(bars)
        strength, price_line, volume_line = rsi_ema_wma_lines(
            df, rsi_period=9, ema_period=3, wma_period=21
        )
        expected_strength = rsi_indicator(df["close"], period=9)
        expected_price_line = ema_indicator(expected_strength, period=3)
        expected_volume_line = wma_indicator(expected_strength, period=21)
        self.assertTrue(strength.equals(expected_strength))
        self.assertTrue(price_line.equals(expected_price_line))
        self.assertTrue(volume_line.equals(expected_volume_line))

    def test_lines_stay_in_rsi_range_once_valid(self):
        bars = make_wavy_bars(120)
        df = to_dataframe(bars)
        strength, price_line, volume_line = rsi_ema_wma_lines(df)
        for series in (strength, price_line, volume_line):
            valid = series.dropna()
            self.assertTrue((valid >= 0).all())
            self.assertTrue((valid <= 100).all())

    def test_flat_price_series_has_no_signals(self):
        # RSI is undefined-direction (50) the whole way on dead-flat
        # prices, so Strength/Price/Volume converge to the same value,
        # never form a strict stack, and never clear a positive min_gap
        # either -- no buy or sell signal should ever fire.
        bars = make_flat_bars(60)
        df = to_dataframe(bars)
        buy_signal, sell_signal, _, _, _ = rsi_ema_wma_signals(df)
        self.assertFalse(buy_signal.any())
        self.assertFalse(sell_signal.any())


class TestRsiEmaWmaSignals(unittest.TestCase):
    def test_buy_signal_requires_full_bullish_stack_above_50_with_gap(self):
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        buy_signal, _, strength, price_line, volume_line = rsi_ema_wma_signals(df, min_gap=5.0)
        for dt in df.index[buy_signal]:
            self.assertLess(volume_line.loc[dt], price_line.loc[dt])
            self.assertLess(price_line.loc[dt], strength.loc[dt])
            self.assertGreater(strength.loc[dt], 50.0)
            self.assertGreaterEqual(strength.loc[dt] - volume_line.loc[dt], 5.0)

    def test_sell_signal_requires_full_bearish_stack_below_50_with_gap(self):
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        _, sell_signal, strength, price_line, volume_line = rsi_ema_wma_signals(df, min_gap=5.0)
        for dt in df.index[sell_signal]:
            self.assertGreater(volume_line.loc[dt], price_line.loc[dt])
            self.assertGreater(price_line.loc[dt], strength.loc[dt])
            self.assertLess(strength.loc[dt], 50.0)
            self.assertGreaterEqual(volume_line.loc[dt] - strength.loc[dt], 5.0)

    def test_larger_min_gap_never_adds_signals(self):
        # Tightening min_gap can only drop signals, never add new ones --
        # every bar that qualifies at a wider gap must also qualify at a
        # narrower one.
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        buy_loose, sell_loose, _, _, _ = rsi_ema_wma_signals(df, min_gap=1.0)
        buy_tight, sell_tight, _, _, _ = rsi_ema_wma_signals(df, min_gap=15.0)
        self.assertTrue((buy_loose.sum() >= buy_tight.sum()))
        self.assertTrue((sell_loose.sum() >= sell_tight.sum()))
        self.assertFalse((buy_tight & ~buy_loose).any())
        self.assertFalse((sell_tight & ~sell_loose).any())

    def test_negative_min_gap_raises(self):
        bars = make_wavy_bars(60)
        df = to_dataframe(bars)
        with self.assertRaises(ValueError):
            rsi_ema_wma_signals(df, min_gap=-1.0)

    def test_signals_fire_once_per_transition_not_every_bar(self):
        # A signal bar's *previous* bar must not already have satisfied
        # every buy/sell condition -- otherwise every bar of a multi-day
        # run in that state would (wrongly) count as a signal.
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        buy_signal, sell_signal, strength, price_line, volume_line = rsi_ema_wma_signals(df, min_gap=5.0)
        bullish_full = (
            (volume_line < price_line).fillna(False)
            & (price_line < strength).fillna(False)
            & (strength > 50.0).fillna(False)
            & ((strength - volume_line) >= 5.0).fillna(False)
        )
        bearish_full = (
            (volume_line > price_line).fillna(False)
            & (price_line > strength).fillna(False)
            & (strength < 50.0).fillna(False)
            & ((volume_line - strength) >= 5.0).fillna(False)
        )
        bullish_prev = bullish_full.shift(1, fill_value=False)
        bearish_prev = bearish_full.shift(1, fill_value=False)
        self.assertFalse((buy_signal & bullish_prev).any())
        self.assertFalse((sell_signal & bearish_prev).any())

    def test_no_signal_during_warmup(self):
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        buy_signal, sell_signal, _, _, volume_line = rsi_ema_wma_signals(df, rsi_period=9, wma_period=21)
        warmup_end = volume_line.first_valid_index()
        self.assertIsNotNone(warmup_end)
        warmup_slice = df.index < warmup_end
        self.assertFalse(buy_signal[warmup_slice].any())
        self.assertFalse(sell_signal[warmup_slice].any())


class TestRsiEmaWmaBullishRecent(unittest.TestCase):
    def test_false_when_not_enough_bars(self):
        bars = make_wavy_bars(10)
        self.assertFalse(rsi_ema_wma_bullish_recent(bars))

    def test_matches_signals_lookback_window(self):
        bars = make_wavy_bars(150)
        df = to_dataframe(bars)
        buy_signal, _, _, _, _ = rsi_ema_wma_signals(df, min_gap=5.0)
        expected = bool(buy_signal.iloc[-3:].any())
        self.assertEqual(rsi_ema_wma_bullish_recent(bars, lookback_days=3, min_gap=5.0), expected)


class TestBacktestRsiEmaWma(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        bars = make_wavy_bars(15)
        with self.assertRaises(ValueError):
            backtest_rsi_ema_wma(bars)

    def test_trades_alternate_buy_then_sell_and_prices_match_bars(self):
        bars = make_wavy_bars(200)
        df = to_dataframe(bars)
        result = backtest_rsi_ema_wma(bars, ticker="TEST")
        self.assertGreater(result.total_trades, 0)
        for t in result.trades:
            self.assertLessEqual(t.entry_date, t.exit_date)
            self.assertAlmostEqual(t.entry_price, df.loc[t.entry_date, "close"])
            self.assertAlmostEqual(t.exit_price, df.loc[t.exit_date, "close"])
            self.assertIn(t.meta.get("exit_reason"), {"bearish_crossover", "period_end"})
            # Wins/losses derive from entry vs exit close price, long-only.
            is_win = t.exit_price > t.entry_price
            self.assertEqual(t.is_win, is_win)

    def test_trade_meta_carries_all_three_lines_at_entry_and_exit(self):
        bars = make_wavy_bars(200)
        result = backtest_rsi_ema_wma(bars, ticker="TEST")
        self.assertGreater(result.total_trades, 0)
        for t in result.trades:
            # Entry: full bullish stack, above 50, gapped by >= 5.
            self.assertLess(t.meta["entry_volume_line"], t.meta["entry_price_line"])
            self.assertLess(t.meta["entry_price_line"], t.meta["entry_strength"])
            self.assertGreater(t.meta["entry_strength"], 50.0)
            self.assertGreaterEqual(t.meta["entry_strength"] - t.meta["entry_volume_line"], 5.0)
            if t.meta["exit_reason"] == "bearish_crossover":
                # Exit on a real sell signal: full bearish stack, below 50, gapped.
                self.assertGreater(t.meta["exit_volume_line"], t.meta["exit_price_line"])
                self.assertGreater(t.meta["exit_price_line"], t.meta["exit_strength"])
                self.assertLess(t.meta["exit_strength"], 50.0)
                self.assertGreaterEqual(t.meta["exit_volume_line"] - t.meta["exit_strength"], 5.0)
            else:
                # period_end: just marked to market, no ordering guarantee.
                for key in ("exit_strength", "exit_price_line", "exit_volume_line"):
                    self.assertIsInstance(t.meta[key], float)

    def test_min_gap_is_a_real_parameter(self):
        # A much larger min_gap on the same data should never produce more
        # trades than a smaller one (each buy needs to clear a wider bar).
        bars = make_wavy_bars(200)
        loose = backtest_rsi_ema_wma(bars, min_gap=1.0, ticker="TEST")
        tight = backtest_rsi_ema_wma(bars, min_gap=25.0, ticker="TEST")
        self.assertGreaterEqual(loose.total_trades, tight.total_trades)

    def test_last_trade_marked_period_end_if_still_open(self):
        bars = make_wavy_bars(200)
        result = backtest_rsi_ema_wma(bars, ticker="TEST")
        if result.trades and result.trades[-1].meta.get("exit_reason") == "period_end":
            self.assertTrue(result.trades[-1].closed_at_period_end)

    def test_equity_curve_matches_buy_and_hold_when_never_in_a_position(self):
        # Flat prices never trigger a buy signal (see test_flat_price_series_has_no_signals),
        # so equity should sit at initial_capital for the whole run.
        bars = make_flat_bars(60)
        result = backtest_rsi_ema_wma(bars, initial_capital=10_000.0, ticker="FLAT")
        self.assertEqual(result.total_trades, 0)
        self.assertTrue((result.equity_curve == 10_000.0).all())

    def test_strategy_name_and_window_metadata(self):
        bars = make_wavy_bars(150)
        result = backtest_rsi_ema_wma(bars, wma_period=21, ticker="TEST")
        self.assertEqual(result.strategy_name, "rsi9_ema3_wma21")
        # Neither is a literal SMA-of-price window -- this strategy's
        # lines live on a separate oscillator panel, not as a price-chart
        # SMA overlay (see _rsi_ema_wma_oscillator_fn in app/strategies.py).
        self.assertIsNone(result.fast_window)
        self.assertIsNone(result.slow_window)


if __name__ == "__main__":
    unittest.main()
