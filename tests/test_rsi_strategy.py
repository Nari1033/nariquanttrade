"""Unit tests for the RSI(14) Momentum Breakout strategy (scan + backtest).
Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_utils import to_dataframe
from engine.indicators import rsi as rsi_indicator
from engine.models import PriceBar
from engine.rsi_strategy import (
    backtest_rsi_momentum,
    rsi_breakout_recent,
    rsi_signals,
)


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


def make_warmup_then_rally_bars(
    n_warmup=20, n_rally=30, base=100.0, step=1.0, rally_pct=0.03, start=date(2020, 1, 1)
):
    """n_warmup bars of small alternating up/down moves around `base`
    (enough non-degenerate gain/loss history for RSI's smoothing to be
    meaningful -- avoids the all-flat "RSI=50" edge case), followed by
    n_rally consecutive up days of `rally_pct` each: a sustained,
    never-reversing rally strong enough to push RSI(14) up through 70 and
    keep climbing well past 80 -- and never back down, so with the
    overbought-then-pullback exit this fixture is exactly the "entered,
    RSI stays elevated, still open at period end" case."""
    rows = []
    price = base
    for i in range(n_warmup):
        price = base + (step if i % 2 == 0 else -step * 0.5)
        rows.append((price, price, price, price))
    for _ in range(n_rally):
        price *= 1 + rally_pct
        rows.append((price, price, price, price))
    return make_bars(rows, start=start)


def make_rally_then_pullback_bars(
    n_warmup=20,
    n_rally=15,
    n_decline=15,
    base=100.0,
    step=1.0,
    rally_pct=0.03,
    decline_pct=0.03,
    start=date(2020, 1, 1),
):
    """Same warm-up as make_warmup_then_rally_bars, then a rally that pushes
    RSI(14) up through 70 and on up past 80 (overbought), followed by a
    decline that pulls RSI back down to/through 60 -- the "faded momentum"
    round trip this strategy's exit is built around. Verified once by hand
    with the default parameters (buy signal ~day 26, overbought ~day 30,
    rollover exit ~day 39); the tests below discover the actual bars
    dynamically rather than hardcoding indices."""
    rows = []
    price = base
    for i in range(n_warmup):
        price = base + (step if i % 2 == 0 else -step * 0.5)
        rows.append((price, price, price, price))
    for _ in range(n_rally):
        price *= 1 + rally_pct
        rows.append((price, price, price, price))
    for _ in range(n_decline):
        price *= 1 - decline_pct
        rows.append((price, price, price, price))
    return make_bars(rows, start=start)


class TestRsiSignals(unittest.TestCase):
    def test_raises_when_overbought_threshold_not_above_buy_threshold(self):
        bars = make_warmup_then_rally_bars()
        df = to_dataframe(bars)
        with self.assertRaises(ValueError):
            rsi_signals(df, period=14, buy_threshold=70.0, overbought_threshold=70.0)
        with self.assertRaises(ValueError):
            rsi_signals(df, period=14, buy_threshold=70.0, overbought_threshold=50.0)

    def test_raises_when_exit_threshold_not_below_overbought_threshold(self):
        bars = make_warmup_then_rally_bars()
        df = to_dataframe(bars)
        with self.assertRaises(ValueError):
            rsi_signals(df, period=14, overbought_threshold=80.0, exit_threshold=80.0)
        with self.assertRaises(ValueError):
            rsi_signals(df, period=14, overbought_threshold=80.0, exit_threshold=85.0)

    def test_buy_signal_fires_on_a_sustained_rally(self):
        bars = make_warmup_then_rally_bars()
        df = to_dataframe(bars)
        buy_signal, r = rsi_signals(df, period=14, buy_threshold=70.0)
        self.assertTrue(buy_signal.any(), "expected at least one buy signal")
        self.assertEqual(len(r), len(df))

    def test_every_buy_signal_satisfies_its_own_definition(self):
        bars = make_warmup_then_rally_bars()
        df = to_dataframe(bars)
        r = rsi_indicator(df["close"], period=14)
        buy_signal, _ = rsi_signals(df, period=14, buy_threshold=70.0)
        for i in range(len(df)):
            if bool(buy_signal.iloc[i]):
                self.assertGreaterEqual(r.iloc[i], 70.0)
                self.assertLess(r.iloc[i - 1], 70.0)

    def test_rsi_series_matches_the_indicator_directly(self):
        bars = make_warmup_then_rally_bars()
        df = to_dataframe(bars)
        r_direct = rsi_indicator(df["close"], period=14)
        _, r_from_signals = rsi_signals(df, period=14)
        pd_testing_equal = (r_direct.fillna(-1) == r_from_signals.fillna(-1)).all()
        self.assertTrue(bool(pd_testing_equal))

    def test_flat_price_never_signals(self):
        bars = make_flat_bars(n=40)
        df = to_dataframe(bars)
        buy_signal, _ = rsi_signals(df, period=14)
        self.assertFalse(buy_signal.any())


def _first_buy_signal_index(df, period=14, buy_threshold=70.0):
    buy_signal, _ = rsi_signals(df, period=period, buy_threshold=buy_threshold)
    hits = buy_signal[buy_signal].index
    return None if len(hits) == 0 else df.index.get_loc(hits[0])


class TestRsiBreakoutRecent(unittest.TestCase):
    def setUp(self):
        self.bars = make_warmup_then_rally_bars()
        self.df = to_dataframe(self.bars)
        first_hit = _first_buy_signal_index(self.df)
        self.assertIsNotNone(first_hit, "test setup requires at least one buy signal to exist")
        self.first_hit = first_hit

    def test_true_when_signal_within_lookback(self):
        # Truncate the series to end exactly on the first signal bar --
        # it's the most recent bar, so it's within any lookback >= 1.
        truncated = self.bars[: self.first_hit + 1]
        self.assertTrue(rsi_breakout_recent(truncated, period=14, lookback_days=3))

    def test_false_when_signal_outside_lookback(self):
        # Extend well past the signal bar so it falls outside a short lookback.
        extended = self.bars[: self.first_hit + 11]
        self.assertFalse(rsi_breakout_recent(extended, period=14, lookback_days=3))

    def test_false_with_insufficient_data(self):
        bars = make_flat_bars(n=10)
        self.assertFalse(rsi_breakout_recent(bars, period=14))


class TestRsiMomentumBacktester(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_rsi_momentum(make_flat_bars(n=10), period=14)

    def test_raises_when_overbought_threshold_not_above_buy_threshold(self):
        bars = make_warmup_then_rally_bars()
        with self.assertRaises(ValueError):
            backtest_rsi_momentum(bars, period=14, buy_threshold=70.0, overbought_threshold=70.0)

    def test_raises_when_exit_threshold_not_below_overbought_threshold(self):
        bars = make_warmup_then_rally_bars()
        with self.assertRaises(ValueError):
            backtest_rsi_momentum(bars, period=14, overbought_threshold=80.0, exit_threshold=80.0)

    def test_flat_price_never_trades(self):
        bars = make_flat_bars(n=40, price=100.0)
        result = backtest_rsi_momentum(bars, period=14, ticker="FLAT")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0, places=6)

    def test_never_reaching_overbought_holds_to_period_end(self):
        # A rally that pushes RSI up through the buy threshold and keeps
        # climbing (never reaching overbought_threshold=95, deliberately
        # set out of reach here) never arms the rollover exit -- with a
        # loose stop loss, the position should just ride to period_end.
        bars = make_warmup_then_rally_bars()
        result = backtest_rsi_momentum(
            bars, period=14, overbought_threshold=95.0, exit_threshold=60.0,
            stop_loss_pct=50.0, ticker="RALLY",
        )
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta.get("exit_reason"), "period_end")

    def test_rally_then_pullback_produces_a_rollover_exit(self):
        bars = make_rally_then_pullback_bars()
        result = backtest_rsi_momentum(bars, period=14, stop_loss_pct=50.0, ticker="ROUNDTRIP")
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta.get("exit_reason"), "rsi_rollover")

    def test_rollover_exit_only_arms_after_overbought_is_reached(self):
        # Sanity check on the fixture itself: RSI must actually have
        # touched overbought_threshold before falling back to
        # exit_threshold, for the round trip in the test above to mean
        # what it claims to mean.
        bars = make_rally_then_pullback_bars()
        df = to_dataframe(bars)
        r = rsi_indicator(df["close"], period=14)
        buy_signal, _ = rsi_signals(df, period=14, buy_threshold=70.0)
        entry_idx = _first_buy_signal_index(df)
        self.assertIsNotNone(entry_idx)
        self.assertTrue(
            (r.iloc[entry_idx:] >= 80.0).any(),
            "fixture must reach the overbought threshold after entry",
        )
        self.assertTrue(
            (r.iloc[entry_idx:] <= 60.0).any(),
            "fixture must fall back to the exit threshold after entry",
        )

    def test_stop_loss_bounds_the_loss_when_price_reverses_before_overbought(self):
        # Splice a relentless decline right after the first real buy
        # signal in the rally -- the stop-loss must cap the loss near
        # -stop_loss_pct, not let it ride down waiting for RSI to ever
        # reach the (now unreachable) overbought threshold.
        warmup_bars = make_warmup_then_rally_bars()
        warmup_df = to_dataframe(warmup_bars)
        first_hit = _first_buy_signal_index(warmup_df)
        self.assertIsNotNone(first_hit)

        rows = [(b.open, b.high, b.low, b.close) for b in warmup_bars[: first_hit + 1]]
        price = rows[-1][3]
        for _ in range(20):
            price *= 0.95
            rows.append((price, price, price, price))
        bars = make_bars(rows)

        result = backtest_rsi_momentum(bars, period=14, stop_loss_pct=10.0, ticker="DROP")
        self.assertEqual(result.total_trades, 1)
        t = result.trades[0]
        self.assertEqual(t.meta.get("exit_reason"), "stop_loss")
        self.assertLess(t.return_pct, 0)
        self.assertGreaterEqual(t.return_pct, -10.0 - 1e-6)

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        bars = make_rally_then_pullback_bars()
        result = backtest_rsi_momentum(bars, period=14, ticker="ROUNDTRIP")
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)

    def test_summary_reports_rsi_strategy_name(self):
        bars = make_rally_then_pullback_bars()
        result = backtest_rsi_momentum(bars, period=14, ticker="ROUNDTRIP")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "rsi_momentum")
        self.assertIsNone(s["fast_window"])
        self.assertEqual(s["slow_window"], 14)


if __name__ == "__main__":
    unittest.main()
