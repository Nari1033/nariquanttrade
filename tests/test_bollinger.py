"""Unit tests for the Bollinger Band Mean Reversion strategy (scan +
backtest). Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bollinger import (
    backtest_bollinger_mean_reversion,
    bollinger_bands,
    bollinger_oversold_recent,
    bollinger_signals,
)
from engine.data_utils import to_dataframe
from engine.indicators import atr as atr_indicator
from engine.models import PriceBar


def make_bars(rows, start=date(2020, 1, 1)):
    """rows: list of (open, high, low, close) tuples, one per consecutive
    calendar day."""
    bars = []
    for i, (o, h, l, c) in enumerate(rows):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=o, high=h, low=l, close=c, volume=1_000))
    return bars


def make_oscillating_series(n=250, mid=100.0, amplitude=8.0, period=20, start=date(2020, 1, 1)):
    """A smooth sine-wave-like price path around `mid` -- oscillates
    between roughly mid-amplitude and mid+amplitude, giving repeated,
    predictable lower/upper band touches to test signal detection and the
    backtester's round-trip trades against."""
    import math

    rows = []
    for i in range(n):
        close = mid + amplitude * math.sin(2 * math.pi * i / period)
        # Give each bar a little intrabar range straddling the close, so
        # low/high can independently pierce a band even when close doesn't.
        o = close
        h = close + amplitude * 0.15
        l = close - amplitude * 0.15
        rows.append((o, h, l, close))
    return make_bars(rows, start=start)


def make_flat_bars(n=40, price=100.0, start=date(2020, 1, 1)):
    return make_bars([(price, price, price, price)] * n, start=start)


class TestBollingerBands(unittest.TestCase):
    def test_bands_are_nan_during_warmup(self):
        bars = make_flat_bars(n=30)
        df = to_dataframe(bars)
        middle, upper, lower = bollinger_bands(df["close"], window=20, num_std=2.0)
        self.assertTrue(middle.iloc[:19].isna().all())
        self.assertTrue(upper.iloc[:19].isna().all())
        self.assertTrue(lower.iloc[:19].isna().all())
        self.assertFalse(middle.iloc[19:].isna().any())

    def test_flat_price_gives_zero_width_bands(self):
        # Zero variance -> stdev is 0 -> upper == lower == middle == price.
        bars = make_flat_bars(n=30, price=100.0)
        df = to_dataframe(bars)
        middle, upper, lower = bollinger_bands(df["close"], window=20, num_std=2.0)
        self.assertAlmostEqual(middle.iloc[-1], 100.0, places=6)
        self.assertAlmostEqual(upper.iloc[-1], 100.0, places=6)
        self.assertAlmostEqual(lower.iloc[-1], 100.0, places=6)

    def test_wider_num_std_gives_wider_bands(self):
        bars = make_oscillating_series(n=100)
        df = to_dataframe(bars)
        _, upper_narrow, lower_narrow = bollinger_bands(df["close"], window=20, num_std=1.0)
        _, upper_wide, lower_wide = bollinger_bands(df["close"], window=20, num_std=3.0)
        self.assertGreater(upper_wide.iloc[-1], upper_narrow.iloc[-1])
        self.assertLess(lower_wide.iloc[-1], lower_narrow.iloc[-1])


class TestBollingerSignals(unittest.TestCase):
    # These signals are a two-bar reversal pattern at each band, not a
    # same-bar wick touch: the "signal candle" closes beyond a band, and
    # the signal only fires on the NEXT bar, if (and only if) that next
    # bar's close reverts back inside. Both bars' bands are evaluated at
    # their own index (a rolling window ending at that bar), matching how
    # the strategy itself evaluates it bar by bar.

    def setUp(self):
        self.bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=20)
        self.df = to_dataframe(self.bars)
        self.buy_signal, self.sell_signal = bollinger_signals(self.df, window=20, num_std=1.0)
        _, self.upper, self.lower = bollinger_bands(self.df["close"], window=20, num_std=1.0)

    def test_signals_fire_on_an_oscillating_series(self):
        self.assertTrue(self.buy_signal.any(), "expected at least one lower-band reversal")
        self.assertTrue(self.sell_signal.any(), "expected at least one upper-band reversal")

    def test_every_buy_signal_satisfies_its_own_definition(self):
        for i in range(len(self.df)):
            if bool(self.buy_signal.iloc[i]):
                # The PREVIOUS bar (the signal candle) closed below ITS OWN
                # band, and THIS bar's close reverts back above its own band.
                self.assertLess(self.df["close"].iloc[i - 1], self.lower.iloc[i - 1])
                self.assertGreater(self.df["close"].iloc[i], self.lower.iloc[i])

    def test_every_sell_signal_satisfies_its_own_definition(self):
        for i in range(len(self.df)):
            if bool(self.sell_signal.iloc[i]):
                self.assertGreater(self.df["close"].iloc[i - 1], self.upper.iloc[i - 1])
                self.assertLess(self.df["close"].iloc[i], self.upper.iloc[i])

    def test_close_beyond_band_does_not_trigger_reversion_signal(self):
        # Price pierces AND closes below the lower band, then keeps sliding
        # -- it never closes back inside, so the final bar should NOT count
        # as a buy signal even though the prior bar closed below the band.
        rows = [(c, c, c, c) for c in [self.df["close"].iloc[i] for i in range(30)]]
        price = rows[-1][0]
        for _ in range(10):
            price *= 0.85
            rows.append((price, price, price, price))
        bars = make_bars(rows)
        df = to_dataframe(bars)
        buy_signal, _ = bollinger_signals(df, window=20, num_std=1.0)
        self.assertFalse(bool(buy_signal.iloc[-1]))


def _first_buy_signal_index(df, window=20, num_std=1.0):
    buy_signal, _ = bollinger_signals(df, window=window, num_std=num_std)
    hits = buy_signal[buy_signal].index
    return None if len(hits) == 0 else df.index.get_loc(hits[0])


class TestBollingerOversoldRecent(unittest.TestCase):
    def setUp(self):
        self.bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=20)
        self.df = to_dataframe(self.bars)
        first_hit = _first_buy_signal_index(self.df)
        self.assertIsNotNone(first_hit, "test setup requires at least one buy signal to exist")
        self.first_hit = first_hit

    def test_true_when_signal_within_lookback(self):
        # Truncate the series to end exactly on the first signal bar --
        # it's the most recent bar, so it's within any lookback >= 1.
        truncated = self.bars[: self.first_hit + 1]
        self.assertTrue(bollinger_oversold_recent(truncated, window=20, num_std=1.0, lookback_days=3))

    def test_false_when_signal_outside_lookback(self):
        # Extend well past the signal bar so it falls outside a short lookback.
        extended = self.bars[: self.first_hit + 11]
        self.assertFalse(bollinger_oversold_recent(extended, window=20, num_std=1.0, lookback_days=3))

    def test_false_with_insufficient_data(self):
        bars = make_flat_bars(n=10)
        self.assertFalse(bollinger_oversold_recent(bars, window=20))


class TestBollingerBacktester(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_bollinger_mean_reversion(make_flat_bars(n=10), window=20)

    def test_flat_price_never_trades(self):
        # Zero-width bands on constant price -> close never strictly closes
        # beyond them -> no signals -> no trades -> strategy flat at 0%.
        bars = make_flat_bars(n=60, price=100.0)
        result = backtest_bollinger_mean_reversion(bars, window=20, ticker="FLAT")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0, places=6)

    def test_oscillating_series_produces_round_trip_trades_with_wins(self):
        bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=20)
        result = backtest_bollinger_mean_reversion(
            bars, window=20, num_std=1.0, atr_period=14, atr_multiple=1.0, ticker="OSC"
        )
        self.assertGreater(result.total_trades, 0)
        self.assertGreater(sum(1 for t in result.trades if t.is_win), 0)
        for t in result.trades:
            self.assertIn(
                t.meta.get("exit_reason"),
                ("profit_target", "band_rejection", "stop_loss", "period_end"),
            )

    def test_profit_target_exit_matches_middle_band_not_close(self):
        # The profit target is a limit-style order against the middle band
        # -- a winning trade should exit AT the band level, not at
        # whatever the bar's close happened to be.
        bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=20)
        df = to_dataframe(bars)
        result = backtest_bollinger_mean_reversion(
            bars, window=20, num_std=1.0, atr_period=14, atr_multiple=1.0, ticker="OSC"
        )
        target_trades = [t for t in result.trades if t.meta.get("exit_reason") == "profit_target"]
        self.assertGreater(len(target_trades), 0, "expected at least one profit-target exit")
        middle, _, _ = bollinger_bands(df["close"], window=20, num_std=1.0)
        for t in target_trades:
            exit_i = df.index.get_loc(t.exit_date)
            self.assertAlmostEqual(t.exit_price, middle.iloc[exit_i], places=6)

    def test_stop_loss_is_anchored_to_the_signal_candle_not_the_entry_bar(self):
        # Splice a relentless decline right after the first real buy
        # signal in an oscillating series -- the stop-loss must fire and
        # must sit exactly `atr_multiple` ATRs below the *signal candle's*
        # low (the oversold bar one before entry), not the entry bar's own
        # low, and not let the loss ride down to the eventual period-end
        # mark.
        warmup_bars = make_oscillating_series(n=100, mid=100.0, amplitude=10.0, period=20)
        warmup_df = to_dataframe(warmup_bars)
        first_hit = _first_buy_signal_index(warmup_df, window=20, num_std=1.0)
        self.assertIsNotNone(first_hit)

        rows = [
            (b.open, b.high, b.low, b.close) for b in warmup_bars[: first_hit + 1]
        ]
        price = rows[-1][3]
        for _ in range(40):
            price *= 0.97
            rows.append((price, price, price, price))
        bars = make_bars(rows)
        df = to_dataframe(bars)

        atr_period, atr_multiple = 14, 1.0
        result = backtest_bollinger_mean_reversion(
            bars, window=20, num_std=1.0, atr_period=atr_period, atr_multiple=atr_multiple, ticker="DROP"
        )
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta.get("exit_reason"), "stop_loss")
        self.assertLess(t.return_pct, 0)

        atr_series = atr_indicator(df, period=atr_period)
        signal_candle_low = df["low"].iloc[first_hit - 1]
        signal_candle_atr = atr_series.iloc[first_hit - 1]
        entry_bar_low = df["low"].iloc[first_hit]
        # Sanity check the test actually discriminates between the two
        # anchors -- if these happened to coincide, matching the formula
        # below wouldn't prove which bar it was anchored to.
        self.assertNotAlmostEqual(signal_candle_low, entry_bar_low, places=2)

        expected_stop = signal_candle_low - atr_multiple * signal_candle_atr
        self.assertAlmostEqual(t.exit_price, expected_stop, places=6)
        self.assertAlmostEqual(t.meta.get("stop_price"), round(expected_stop, 2), places=2)

    def test_sell_signal_while_flat_does_not_open_a_short(self):
        # This app is long-only: the upper-band reversal ("short entry" in
        # the original rule set) must be a no-op while there's no open
        # position, never a new (short) trade. Truncate the series to end
        # right after the first sell_signal, but before any buy_signal has
        # ever fired, and confirm nothing was ever opened.
        bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=20)
        df = to_dataframe(bars)
        buy_signal, sell_signal = bollinger_signals(df, window=20, num_std=1.0)
        sell_hits = sell_signal[sell_signal].index
        self.assertGreater(len(sell_hits), 0)
        first_sell_i = df.index.get_loc(sell_hits[0])
        buy_hits = buy_signal[buy_signal].index
        first_buy_i = df.index.get_loc(buy_hits[0]) if len(buy_hits) else None
        self.assertTrue(
            first_buy_i is None or first_sell_i < first_buy_i,
            "test setup requires a sell signal to occur before any buy signal",
        )

        truncated = bars[: first_sell_i + 1]
        result = backtest_bollinger_mean_reversion(
            truncated, window=20, num_std=1.0, atr_period=14, atr_multiple=1.0, ticker="FLATSHORT"
        )
        self.assertEqual(result.total_trades, 0)

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        bars = make_oscillating_series(n=200, mid=100.0, amplitude=10.0, period=25)
        result = backtest_bollinger_mean_reversion(bars, window=20, num_std=1.0, ticker="OSC")
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)

    def test_summary_reports_bollinger_strategy_name(self):
        bars = make_oscillating_series(n=150)
        result = backtest_bollinger_mean_reversion(bars, window=20, ticker="OSC")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "bollinger_mean_reversion")
        self.assertIsNone(s["fast_window"])
        self.assertEqual(s["slow_window"], 20)


if __name__ == "__main__":
    unittest.main()
