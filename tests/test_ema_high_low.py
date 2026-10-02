"""Unit tests for the 34 EMA High/Low strategy (and its EMA34+DAY-1
variant): signals, scan, and backtest. Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from engine.data_utils import to_dataframe
from engine.ema_high_low import (
    backtest_ema_high_low,
    ema_high_low_lines,
    ema_high_low_recent,
    ema_high_low_signals,
)
from engine.indicators import ema as ema_indicator
from engine.models import PriceBar


def make_bars(rows, start=date(2020, 1, 1)):
    """rows: list of (open, high, low, close) tuples, one per consecutive
    calendar day."""
    bars = []
    for i, (o, h, l, c) in enumerate(rows):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=o, high=h, low=l, close=c, volume=1_000))
    return bars


def make_flat_bars(n=40, price=100.0, start=date(2020, 1, 1)):
    return make_bars([(price, price, price, price)] * n, start=start)


def make_rally_then_decline_bars(
    n_warmup=40, n_rally=60, n_decline=60, base=100.0, rally_pct=0.01, decline_pct=0.015,
    start=date(2020, 1, 1),
):
    """A flat warm-up (long enough for EMA(34) to be valid), then a
    sustained rally (should trigger a long breakout and hold it), then a
    sustained decline (should flip the position to short on the way
    down). Every rally bar is green (close > open) and every decline bar
    is red (close < open) -- deliberately, so the EMA34+DAY-1 variant
    doesn't block the round trip either, making this fixture usable by
    both the plain and DAY-1 tests."""
    rows = [(base, base * 1.01, base * 0.99, base)] * n_warmup
    price = base
    for _ in range(n_rally):
        o = price
        price *= 1 + rally_pct
        rows.append((o, price * 1.002, o * 0.998, price))
    for _ in range(n_decline):
        o = price
        price *= 1 - decline_pct
        rows.append((o, o * 1.002, price * 0.998, price))
    return make_bars(rows, start=start)


class TestEmaHighLowLines(unittest.TestCase):
    def test_matches_the_ema_indicator_directly_per_column(self):
        bars = make_rally_then_decline_bars()
        df = to_dataframe(bars)
        ema_high, ema_low, ema_close = ema_high_low_lines(df, ema_period=34)
        self.assertTrue((ema_high.fillna(-1) == ema_indicator(df["high"], 34).fillna(-1)).all())
        self.assertTrue((ema_low.fillna(-1) == ema_indicator(df["low"], 34).fillna(-1)).all())
        self.assertTrue((ema_close.fillna(-1) == ema_indicator(df["close"], 34).fillna(-1)).all())

    def test_raises_on_too_short_period(self):
        bars = make_flat_bars(n=10)
        df = to_dataframe(bars)
        with self.assertRaises(ValueError):
            ema_high_low_lines(df, ema_period=1)


def _make_signal_fixture_df():
    """Hand-built 12-bar frame, small enough to reason about by hand with
    ema_period=5: a flat run to warm up both EMAs, a RED candle (day index
    7), a doji (day index 8), a big breakout UP on day index 9 (preceded
    by the doji), and a breakdown DOWN on day index 10 (preceded by a
    GREEN candle on day 9). This exercises both breakout directions and
    both "previous day" candle colors (plus the doji case) in one frame."""
    idx = pd.date_range("2020-01-01", periods=12, freq="D")
    data = {
        "open": [100] * 7 + [103, 100, 120, 100, 80],
        "high": [101] * 7 + [104, 101, 121, 101, 81],
        "low": [99] * 7 + [99, 99, 119, 99, 79],
        "close": [100] * 7 + [99, 100, 120, 95, 80],
    }
    return pd.DataFrame(data, index=idx)


class TestEmaHighLowSignals(unittest.TestCase):
    def setUp(self):
        self.df = _make_signal_fixture_df()

    def test_plain_signals_fire_on_genuine_breakouts(self):
        long_signal, short_signal, ema_high, ema_low, _ema_close = ema_high_low_signals(
            self.df, ema_period=5, day_minus_1_filter=False
        )
        # Day index 9 (2020-01-10): close=120 breaks up through emaHigh.
        self.assertTrue(bool(long_signal.iloc[9]))
        # Day index 10 (2020-01-11): close=95 breaks down through emaLow.
        self.assertTrue(bool(short_signal.iloc[10]))
        # Nowhere else should either fire in this fixture.
        self.assertEqual(int(long_signal.sum()), 1)
        self.assertEqual(int(short_signal.sum()), 1)

    def test_signal_bars_are_genuinely_outside_the_channel(self):
        long_signal, short_signal, ema_high, ema_low, _ = ema_high_low_signals(
            self.df, ema_period=5, day_minus_1_filter=False
        )
        close = self.df["close"]
        for i in range(len(self.df)):
            if bool(long_signal.iloc[i]):
                self.assertGreater(close.iloc[i], ema_high.iloc[i])
            if bool(short_signal.iloc[i]):
                self.assertLess(close.iloc[i], ema_low.iloc[i])

    def test_day_minus_1_filter_blocks_long_after_a_non_green_previous_day(self):
        # Day 9's previous day (day 8) is a doji (open == close == 100) --
        # neither green nor red -- so the long breakout on day 9 must be
        # masked out under the DAY-1 filter even though it fires in the
        # unfiltered signals.
        long_signal, _short_signal, *_ = ema_high_low_signals(
            self.df, ema_period=5, day_minus_1_filter=True
        )
        self.assertFalse(bool(long_signal.iloc[9]))

    def test_day_minus_1_filter_blocks_short_after_a_green_previous_day(self):
        # Day 10's previous day (day 9) is green (close=120 > open=100),
        # so the short breakdown on day 10 must be masked out under the
        # DAY-1 filter even though it fires in the unfiltered signals.
        _long_signal, short_signal, *_ = ema_high_low_signals(
            self.df, ema_period=5, day_minus_1_filter=True
        )
        self.assertFalse(bool(short_signal.iloc[10]))

    def test_flat_price_never_signals(self):
        bars = make_flat_bars(n=40)
        df = to_dataframe(bars)
        long_signal, short_signal, *_ = ema_high_low_signals(df, ema_period=34)
        self.assertFalse(long_signal.any())
        self.assertFalse(short_signal.any())


class TestEmaHighLowRecent(unittest.TestCase):
    def test_true_when_a_breakout_is_within_lookback(self):
        bars = make_rally_then_decline_bars()
        df = to_dataframe(bars)
        long_signal, _short_signal, *_ = ema_high_low_signals(df, ema_period=34)
        first_hit = df.index.get_loc(long_signal[long_signal].index[0])
        truncated = bars[: first_hit + 1]
        self.assertTrue(ema_high_low_recent(truncated, ema_period=34, lookback_days=3))

    def test_false_when_breakout_is_outside_lookback(self):
        bars = make_rally_then_decline_bars()
        df = to_dataframe(bars)
        long_signal, _short_signal, *_ = ema_high_low_signals(df, ema_period=34)
        first_hit = df.index.get_loc(long_signal[long_signal].index[0])
        extended = bars[: first_hit + 11]
        self.assertFalse(ema_high_low_recent(extended, ema_period=34, lookback_days=3))

    def test_false_with_insufficient_data(self):
        bars = make_flat_bars(n=10)
        self.assertFalse(ema_high_low_recent(bars, ema_period=34))

    def test_false_on_flat_price(self):
        bars = make_flat_bars(n=40)
        self.assertFalse(ema_high_low_recent(bars, ema_period=34))


class TestBacktestEmaHighLow(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_ema_high_low(make_flat_bars(n=10), ema_period=34)

    def test_flat_price_never_trades(self):
        bars = make_flat_bars(n=60, price=100.0)
        result = backtest_ema_high_low(bars, ema_period=34, ticker="FLAT")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0, places=6)

    def test_stays_flat_until_the_first_breakout(self):
        bars = make_rally_then_decline_bars()
        df = to_dataframe(bars)
        result = backtest_ema_high_low(bars, ema_period=34, initial_capital=10_000.0)
        self.assertGreaterEqual(result.total_trades, 1)
        first_trade = result.trades[0]
        # Equity must sit exactly at initial_capital for every bar before
        # the first trade's entry date (flat = parked in cash, not
        # compounding on price moves it was never exposed to).
        before = df.index[df.index < first_trade.entry_date]
        for dt in before:
            self.assertAlmostEqual(result.equity_curve.loc[dt], 10_000.0, places=6)

    def test_rally_then_decline_goes_long_then_flips_short(self):
        bars = make_rally_then_decline_bars()
        result = backtest_ema_high_low(bars, ema_period=34, initial_capital=10_000.0, ticker="ROUNDTRIP")
        self.assertGreaterEqual(result.total_trades, 2)
        legs = [t.meta.get("leg") for t in result.trades]
        self.assertEqual(legs[0], "long")
        self.assertEqual(legs[1], "short")
        # The long leg should be profitable (entered on the way up, exited
        # on the way back down after a sustained rally); ditto the short
        # leg being profitable on the way down.
        self.assertGreater(result.trades[0].return_pct, 0)
        self.assertTrue(result.trades[0].is_win)
        self.assertGreater(result.trades[1].return_pct, 0)
        self.assertTrue(result.trades[1].is_win)
        # A reversal's exit_date must exactly equal the next leg's entry_date
        # (same bar, same close price -- stop-and-reverse, not a gap).
        self.assertEqual(result.trades[0].exit_date, result.trades[1].entry_date)
        self.assertAlmostEqual(result.trades[0].exit_price, result.trades[1].entry_price, places=6)

    def test_final_open_position_is_marked_at_period_end(self):
        # A pure, never-reversing rally: the strategy goes long and should
        # still be long (open) when the data runs out.
        rows = [(100, 101, 99, 100)] * 40
        price = 100.0
        for _ in range(60):
            o = price
            price *= 1.01
            rows.append((o, price * 1.002, o * 0.998, price))
        bars = make_bars(rows)
        result = backtest_ema_high_low(bars, ema_period=34, initial_capital=10_000.0)
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertTrue(t.closed_at_period_end)
        self.assertEqual(t.meta.get("leg"), "long")
        self.assertEqual(t.exit_date, to_dataframe(bars).index[-1])

    def test_no_new_entry_is_taken_on_the_final_bar(self):
        # A flat run, then one single breakout bar right at the very end
        # of the data -- there's no bar left after it to manage a new
        # position, so the strategy should stay flat, not open-and-
        # immediately-close a same-day trade.
        rows = [(100, 101, 99, 100)] * 40 + [(100, 140, 100, 139)]
        bars = make_bars(rows)
        result = backtest_ema_high_low(bars, ema_period=34, initial_capital=10_000.0)
        self.assertEqual(result.total_trades, 0)

    def test_day_minus_1_variant_blocks_a_trade_the_plain_variant_takes(self):
        # Reuse the hand-built signal fixture (see TestEmaHighLowSignals):
        # its long breakout on day 9 is preceded by a doji, so the DAY-1
        # variant must take fewer (or different) trades than the plain one
        # over the same data -- a true end-to-end check, not just a check
        # on the signal series in isolation.
        df = _make_signal_fixture_df()
        bars = [
            PriceBar(
                date=row.Index.date(), open=row.open, high=row.high, low=row.low, close=row.close,
                volume=1_000,
            )
            for row in df.itertuples()
        ]
        plain = backtest_ema_high_low(bars, ema_period=5, initial_capital=10_000.0)
        filtered = backtest_ema_high_low(
            bars, ema_period=5, day_minus_1_filter=True, initial_capital=10_000.0
        )
        self.assertGreaterEqual(plain.total_trades, 1)
        self.assertLess(filtered.total_trades, plain.total_trades)

    def test_buy_hold_return_matches_first_to_last_close(self):
        bars = make_rally_then_decline_bars()
        df = to_dataframe(bars)
        result = backtest_ema_high_low(bars, ema_period=34)
        expected = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        self.assertAlmostEqual(result.buy_hold_return_pct, expected, places=6)

    def test_strategy_name_reflects_the_variant(self):
        bars = make_flat_bars(n=60)
        plain = backtest_ema_high_low(bars, ema_period=34)
        day1 = backtest_ema_high_low(bars, ema_period=34, day_minus_1_filter=True)
        self.assertEqual(plain.strategy_name, "ema_high_low")
        self.assertEqual(day1.strategy_name, "ema_high_low_day_minus_1")


if __name__ == "__main__":
    unittest.main()
