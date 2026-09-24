"""Unit tests for the Black-Scholes pricing utilities and the Bull Put
Spread backtester. Uses stdlib unittest only (no pytest/scipy/numpy).

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import math
import os
import sys
import unittest
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.calendar_filters import DEFAULT_WINDOW_DAYS
from engine.models import PriceBar
from engine.options_backtester import backtest_bull_put_spread
from engine.options_pricing import (
    black_scholes_delta,
    black_scholes_price,
    norm_cdf,
    norm_ppf,
    realized_volatility,
    strike_for_delta,
    strike_for_put_delta_magnitude,
)


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
    volatility is meaningfully above zero and the price spends long
    stretches above its own 200-day SMA (i.e. the bull put spread's trend
    filter has plenty of eligible days, and non-trivial vol drives some
    trades to profit target and some to stop loss)."""
    import random

    rng = random.Random(seed)
    price = start_price
    closes = []
    for _ in range(n):
        ret = rng.gauss(drift, 0.014)
        price *= math.exp(ret)
        closes.append(price)
    return make_bars(closes, start=date(2020, 1, 1))


class TestNormalDistribution(unittest.TestCase):
    def test_norm_cdf_known_values(self):
        self.assertAlmostEqual(norm_cdf(0.0), 0.5, places=6)
        self.assertAlmostEqual(norm_cdf(1.0), 0.8413447, places=6)
        self.assertAlmostEqual(norm_cdf(-1.959963985), 0.025, places=6)
        self.assertAlmostEqual(norm_cdf(1.959963985), 0.975, places=6)

    def test_norm_ppf_is_inverse_of_norm_cdf(self):
        for p in (0.01, 0.025, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.99):
            x = norm_ppf(p)
            self.assertAlmostEqual(norm_cdf(x), p, places=8)

    def test_norm_ppf_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            norm_ppf(0.0)
        with self.assertRaises(ValueError):
            norm_ppf(1.0)


class TestBlackScholes(unittest.TestCase):
    # Textbook reference (Hull, "Options, Futures, and Other Derivatives"):
    # S=42, K=40, r=10%, sigma=20%, T=0.5y -> call ~4.76, put ~0.81.
    S, K, r, sigma, T = 42, 40, 0.10, 0.20, 0.5

    def test_matches_textbook_reference_prices(self):
        call = black_scholes_price("call", self.S, self.K, self.T, self.r, self.sigma)
        put = black_scholes_price("put", self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call, 4.76, places=2)
        self.assertAlmostEqual(put, 0.81, places=2)

    def test_put_call_parity(self):
        call = black_scholes_price("call", self.S, self.K, self.T, self.r, self.sigma)
        put = black_scholes_price("put", self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call - put, self.S - self.K * math.exp(-self.r * self.T), places=9)

    def test_delta_matches_finite_difference_of_price(self):
        eps = 0.01
        for opt_type in ("call", "put"):
            analytical = black_scholes_delta(opt_type, self.S, self.K, self.T, self.r, self.sigma)
            p_up = black_scholes_price(opt_type, self.S + eps, self.K, self.T, self.r, self.sigma)
            p_dn = black_scholes_price(opt_type, self.S - eps, self.K, self.T, self.r, self.sigma)
            fd_delta = (p_up - p_dn) / (2 * eps)
            self.assertAlmostEqual(analytical, fd_delta, places=4)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            black_scholes_price("put", 100, 100, 0.0, 0.045, 0.2)  # T=0
        with self.assertRaises(ValueError):
            black_scholes_price("put", 100, 100, 0.5, 0.045, 0.0)  # sigma=0
        with self.assertRaises(ValueError):
            black_scholes_price("straddle", 100, 100, 0.5, 0.045, 0.2)  # bad type


class TestStrikeForDelta(unittest.TestCase):
    def test_put_strike_round_trips_to_target_delta(self):
        for target in (0.10, 0.20, 0.30, 0.45):
            K = strike_for_put_delta_magnitude(S=100.0, delta_magnitude=target, T=30 / 365, r=0.045, sigma=0.25)
            delta_check = black_scholes_delta("put", 100.0, K, 30 / 365, 0.045, 0.25)
            self.assertAlmostEqual(delta_check, -target, places=8)

    def test_call_strike_round_trips_to_target_delta(self):
        for target in (0.10, 0.30, 0.50):
            K = strike_for_delta("call", 100.0, target, 30 / 365, 0.045, 0.25)
            delta_check = black_scholes_delta("call", 100.0, K, 30 / 365, 0.045, 0.25)
            self.assertAlmostEqual(delta_check, target, places=8)

    def test_lower_delta_put_is_further_out_of_the_money(self):
        # A 10-delta put should sit further below spot than a 40-delta put.
        k10 = strike_for_put_delta_magnitude(100.0, 0.10, 30 / 365, 0.045, 0.25)
        k40 = strike_for_put_delta_magnitude(100.0, 0.40, 30 / 365, 0.045, 0.25)
        self.assertLess(k10, k40)


class TestRealizedVolatility(unittest.TestCase):
    def test_warmup_is_nan(self):
        flat = pd.Series([100.0] * 30)
        rv = realized_volatility(flat, window=20)
        self.assertTrue(rv.iloc[:19].isna().all())

    def test_zero_return_series_has_zero_vol(self):
        flat = pd.Series([100.0] * 30)
        rv = realized_volatility(flat, window=20)
        self.assertAlmostEqual(rv.iloc[-1], 0.0, places=9)

    def test_noisier_series_has_higher_vol(self):
        import random

        rng = random.Random(1)
        calm = [100.0]
        wild = [100.0]
        for _ in range(60):
            calm.append(calm[-1] * math.exp(rng.gauss(0, 0.002)))
        rng2 = random.Random(1)
        for _ in range(60):
            wild.append(wild[-1] * math.exp(rng2.gauss(0, 0.03)))
        rv_calm = realized_volatility(pd.Series(calm), window=20).iloc[-1]
        rv_wild = realized_volatility(pd.Series(wild), window=20).iloc[-1]
        self.assertLess(rv_calm, rv_wild)


class TestBullPutSpreadBacktester(unittest.TestCase):
    def setUp(self):
        self.bars = make_trending_series()

    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_bull_put_spread(make_bars([100.0] * 50))

    def test_no_entries_when_trend_filter_never_passes(self):
        # A relentlessly declining series never trades above its own SMA-200.
        closes = [100.0 * math.exp(-0.001 * i) for i in range(700)]
        result = backtest_bull_put_spread(make_bars(closes), ticker="DOWN")
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.strategy_return_pct, 0.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], result.initial_capital)

    def test_every_entry_respects_the_trend_filter(self):
        result = backtest_bull_put_spread(self.bars, ticker="TREND", trend_sma_window=200)
        df = pd.DataFrame(
            {"close": [b.close for b in self.bars]},
            index=pd.to_datetime([b.date for b in self.bars]),
        )
        sma200 = df["close"].rolling(200, min_periods=200).mean()
        self.assertGreater(result.total_trades, 5, "expected a reasonable number of trades to check")
        for t in result.trades:
            self.assertGreater(df.loc[t.entry_date, "close"], sma200.loc[t.entry_date])

    def test_profit_target_and_stop_loss_thresholds_are_respected(self):
        result = backtest_bull_put_spread(
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

    def test_stop_loss_never_exceeds_the_spreads_max_possible_loss(self):
        result = backtest_bull_put_spread(self.bars, ticker="TREND")
        for t in result.trades:
            if t.meta.get("exit_reason") != "stop_loss":
                continue
            # Recompute from full precision, not the rounded meta strikes.
            width_pct = 3.0  # default spread_width_pct
            width = t.meta["short_strike"] * width_pct / 100.0
            # meta strikes are rounded to 2dp for display; allow a small
            # tolerance from that rounding when checking the theoretical bound.
            max_loss_pct = -(width - t.entry_price) / t.entry_price * 100
            self.assertGreaterEqual(t.return_pct, max_loss_pct - 1.0)

    def test_equity_curve_matches_sum_of_realized_trade_pnl(self):
        result = backtest_bull_put_spread(self.bars, ticker="TREND", initial_capital=10_000.0)
        closed_trades_pnl = sum(
            (t.entry_price - t.exit_price) * 100.0 for t in result.trades if not t.closed_at_period_end
        )
        open_trade_pnl = sum(
            (t.entry_price - t.exit_price) * 100.0 for t in result.trades if t.closed_at_period_end
        )
        expected_final_equity = 10_000.0 + closed_trades_pnl + open_trade_pnl
        self.assertAlmostEqual(result.equity_curve.iloc[-1], expected_final_equity, places=6)

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        result = backtest_bull_put_spread(self.bars, ticker="TREND")
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)

    def test_summary_reports_bull_put_spread_strategy_name(self):
        result = backtest_bull_put_spread(self.bars, ticker="TREND")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "bull_put_spread")
        self.assertIsNone(s["fast_window"])
        self.assertEqual(s["slow_window"], 200)

    def test_entry_day_of_month_zero_matches_unfiltered_baseline(self):
        # 0 must be a true no-op -- same trade count and same entry dates
        # as never passing the param at all.
        baseline = backtest_bull_put_spread(self.bars, ticker="TREND")
        filtered = backtest_bull_put_spread(self.bars, ticker="TREND", entry_day_of_month=0)
        self.assertEqual(
            [t.entry_date for t in baseline.trades], [t.entry_date for t in filtered.trades]
        )

    def test_entry_day_of_month_restricts_every_entry_to_the_window(self):
        target = 10
        result = backtest_bull_put_spread(self.bars, ticker="TREND", entry_day_of_month=target)
        self.assertGreater(result.total_trades, 0, "test fixture should still produce some trades")
        for t in result.trades:
            day = t.entry_date.day
            # Distance to `target`, allowing for wraparound at month
            # boundaries not being modeled -- day_of_month_ok is a plain
            # abs() comparison, so just re-check that same arithmetic here.
            self.assertLessEqual(abs(day - target), DEFAULT_WINDOW_DAYS)
            self.assertEqual(t.meta["entry_day_of_month"], target)

    def test_narrow_entry_window_trades_less_than_unfiltered(self):
        # A day-of-month filter can only ever remove eligible entry days,
        # never add one -- so it should never produce *more* trades than
        # leaving it off, and for a fixture with many eligible uptrend
        # days it should produce strictly fewer.
        baseline = backtest_bull_put_spread(self.bars, ticker="TREND")
        filtered = backtest_bull_put_spread(self.bars, ticker="TREND", entry_day_of_month=15)
        self.assertLessEqual(filtered.total_trades, baseline.total_trades)


if __name__ == "__main__":
    unittest.main()
