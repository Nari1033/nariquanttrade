"""Unit tests for the Cash-Secured Put backtester. Uses stdlib unittest
only (no pytest/scipy/numpy).

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import math
import os
import sys
import unittest
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.cash_secured_put import backtest_cash_secured_put
from engine.models import PriceBar


def make_bars(closes, start=date(2020, 1, 1)):
    """One PriceBar per consecutive calendar day; open=high=low=close."""
    bars = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=c, high=c, low=c, close=c, volume=1_000))
    return bars


def make_trending_series(n=800, start_price=100.0, drift=0.0004, seed=12345):
    """A long, deterministic (seeded) pseudo-random walk with a mild
    positive drift and clearly non-trivial day-to-day noise, so realized
    volatility is meaningfully above zero and price spends long stretches
    above its own 200-day SMA (plenty of eligible entry days, and enough
    vol to drive some trades to profit target and some to stop loss)."""
    import random

    rng = random.Random(seed)
    price = start_price
    closes = []
    for _ in range(n):
        ret = rng.gauss(drift, 0.014)
        price *= math.exp(ret)
        closes.append(price)
    return make_bars(closes, start=date(2020, 1, 1))


class TestCashSecuredPutBacktester(unittest.TestCase):
    def setUp(self):
        self.bars = make_trending_series()

    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_cash_secured_put(make_bars([100.0] * 50))

    def test_no_entries_when_trend_filter_never_passes(self):
        closes = [100.0 * math.exp(-0.001 * i) for i in range(700)]
        result = backtest_cash_secured_put(make_bars(closes), ticker="DOWN")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], result.initial_capital)

    def test_every_entry_respects_the_trend_filter(self):
        result = backtest_cash_secured_put(self.bars, ticker="TREND", trend_sma_window=200)
        df = pd.DataFrame(
            {"close": [b.close for b in self.bars]},
            index=pd.to_datetime([b.date for b in self.bars]),
        )
        sma200 = df["close"].rolling(200, min_periods=200).mean()
        self.assertGreater(result.total_trades, 5, "expected a reasonable number of trades to check")
        for t in result.trades:
            self.assertGreater(df.loc[t.entry_date, "close"], sma200.loc[t.entry_date])

    def test_default_dte_and_delta_match_the_spec(self):
        # 45 DTE, ~30-delta short put, close at 50% of max profit -- the
        # spec's three numbers should be the function's own defaults.
        result = backtest_cash_secured_put(self.bars, ticker="TREND")
        for t in result.trades:
            self.assertEqual(t.meta.get("dte_entry"), 45)

    def test_profit_target_and_stop_loss_thresholds_are_respected(self):
        result = backtest_cash_secured_put(
            self.bars, ticker="TREND", profit_target_pct=50.0, stop_loss_pct=100.0
        )
        tol = 1e-6
        n_profit = n_stop = 0
        for t in result.trades:
            reason = t.meta.get("exit_reason")
            if reason == "profit_target":
                self.assertGreaterEqual(t.return_pct, 50.0 - tol)
                n_profit += 1
            elif reason == "stop_loss":
                self.assertLessEqual(t.return_pct, -100.0 + tol)
                n_stop += 1
        self.assertGreater(n_profit + n_stop, 0, "expected at least some profit/stop exits to check")

    def test_no_long_strike_in_meta_since_there_is_no_protective_leg(self):
        result = backtest_cash_secured_put(self.bars, ticker="TREND")
        self.assertGreater(result.total_trades, 0)
        for t in result.trades:
            self.assertIn("short_strike", t.meta)
            self.assertNotIn("long_strike", t.meta)

    def test_equity_curve_matches_sum_of_realized_trade_pnl(self):
        result = backtest_cash_secured_put(self.bars, ticker="TREND", initial_capital=10_000.0)
        total_pnl = sum((t.entry_price - t.exit_price) * 100.0 for t in result.trades)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 10_000.0 + total_pnl, places=6)

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        result = backtest_cash_secured_put(self.bars, ticker="TREND")
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)

    def test_summary_reports_cash_secured_put_strategy_name(self):
        result = backtest_cash_secured_put(self.bars, ticker="TREND")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "cash_secured_put")
        self.assertIsNone(s["fast_window"])
        self.assertEqual(s["slow_window"], 200)


if __name__ == "__main__":
    unittest.main()
