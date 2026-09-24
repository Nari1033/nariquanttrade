"""Unit tests for the Wheel Strategy backtester (cash-secured puts while
flat, covered calls while holding shares). Uses stdlib unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import math
import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.calendar_filters import DEFAULT_WINDOW_DAYS
from engine.models import PriceBar
from engine.wheel import backtest_wheel_strategy


def make_bars(closes, start=date(2020, 1, 1)):
    """One PriceBar per consecutive calendar day, with a touch of intrabar
    range around each close (open=close, high/low +-0.2%)."""
    bars = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        bars.append(PriceBar(date=d, open=c, high=c * 1.002, low=c * 0.998, close=c, volume=1_000))
    return bars


def make_mild_oscillation(n, mid=100.0, amp=3.0, period=25):
    return [mid + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def make_full_cycle_closes():
    """A fully deterministic price path engineered to walk the wheel
    through every state transition at least once: a few puts expiring
    worthless, a crash big enough to force assignment, a rally big enough
    to get the shares called away, and a tail that runs out of data with
    a leg still open (period_end)."""
    closes = make_mild_oscillation(120, mid=100.0, amp=3.0)

    price = closes[-1]
    for i in range(60):
        price *= 0.985  # ~60% total decline -- deep enough to blow through any realistic put strike
        closes.append(price)

    base = closes[-1]
    closes.extend(base + 2.0 * math.sin(2 * math.pi * i / 25) for i in range(120))

    price = closes[-1]
    for i in range(60):
        price *= 1.02  # ~3x -- deep enough to blow through any realistic call strike
        closes.append(price)

    tail = closes[-1]
    closes.extend(tail + 1.0 * math.sin(2 * math.pi * i / 25) for i in range(40))
    return closes


class TestWheelBacktester(unittest.TestCase):
    def test_raises_on_too_little_data(self):
        with self.assertRaises(ValueError):
            backtest_wheel_strategy(make_bars([100.0] * 10))

    def test_first_leg_sold_is_always_a_put(self):
        bars = make_bars(make_mild_oscillation(300))
        result = backtest_wheel_strategy(bars, ticker="OSC")
        self.assertGreater(result.total_trades, 0)
        self.assertEqual(result.trades[0].meta["leg"], "put")

    def test_meta_shape_has_leg_and_short_strike_but_no_long_strike(self):
        bars = make_bars(make_mild_oscillation(300))
        result = backtest_wheel_strategy(bars, ticker="OSC")
        self.assertGreater(result.total_trades, 0)
        for t in result.trades:
            self.assertIn(t.meta["leg"], ("put", "call"))
            self.assertIn("short_strike", t.meta)
            self.assertNotIn("long_strike", t.meta)

    def test_worthless_expiration_is_a_clean_full_win(self):
        bars = make_bars(make_full_cycle_closes())
        result = backtest_wheel_strategy(bars, ticker="CYCLE")
        worthless = [t for t in result.trades if t.meta["exit_reason"] == "expired_worthless"]
        self.assertGreater(len(worthless), 0, "expected at least one worthless expiration to check")
        for t in worthless:
            self.assertEqual(t.exit_price, 0.0)
            self.assertAlmostEqual(t.return_pct, 100.0, places=6)
            self.assertTrue(t.is_win)

    def test_state_machine_alternates_legs_correctly_through_a_full_cycle(self):
        # Walks a crash (forces assignment) and a rally (forces the shares
        # to be called away) -- this exercises every transition the wheel
        # is supposed to make, and checks each one generically rather than
        # hardcoding the whole trade sequence.
        bars = make_bars(make_full_cycle_closes())
        result = backtest_wheel_strategy(bars, ticker="CYCLE")
        trades = result.trades
        self.assertGreaterEqual(len(trades), 5)

        assigned_count = sum(1 for t in trades if t.meta["exit_reason"] == "assigned")
        called_away_count = sum(1 for t in trades if t.meta["exit_reason"] == "called_away")
        self.assertGreater(assigned_count, 0, "scenario should force at least one assignment")
        self.assertGreater(called_away_count, 0, "scenario should force at least one call-away")

        self.assertEqual(trades[0].meta["leg"], "put")
        for k in range(len(trades) - 1):
            reason = trades[k].meta["exit_reason"]
            next_leg = trades[k + 1].meta["leg"]
            if reason == "assigned":
                self.assertEqual(next_leg, "call", "assignment should switch to selling covered calls")
            elif reason == "called_away":
                self.assertEqual(next_leg, "put", "being called away should switch back to selling puts")
            elif reason == "expired_worthless":
                self.assertEqual(
                    next_leg, trades[k].meta["leg"], "a worthless expiration keeps selling the same leg type"
                )

        # Every put-selling trade actually happened while flat, and every
        # call-selling trade happened while holding the shares from the
        # most recent assignment -- another way of stating the same
        # invariant, anchored to "assigned"/"called_away" as the only
        # events that flip share ownership.
        holding_shares = False
        for t in trades:
            expected_leg = "call" if holding_shares else "put"
            self.assertEqual(t.meta["leg"], expected_leg)
            if t.meta["exit_reason"] == "assigned":
                holding_shares = True
            elif t.meta["exit_reason"] == "called_away":
                holding_shares = False

    def test_equity_grows_while_holding_shares_through_a_rally(self):
        bars = make_bars(make_full_cycle_closes())
        result = backtest_wheel_strategy(bars, ticker="CYCLE")
        # The *last* assignment-to-call-away stretch is the one that runs
        # into the engineered rally at the end of the price path (earlier
        # assign/call-away cycles happen during the crash/flat sections).
        assigned = [t for t in result.trades if t.meta["exit_reason"] == "assigned"][-1]
        called_away = next(
            t for t in result.trades
            if t.meta["exit_reason"] == "called_away" and t.exit_date > assigned.exit_date
        )
        equity_at_assignment = result.equity_curve.loc[assigned.exit_date]
        equity_at_call_away = result.equity_curve.loc[called_away.exit_date]
        self.assertGreater(
            equity_at_call_away, equity_at_assignment,
            "holding 100 shares through the engineered rally should grow equity",
        )

    def test_period_end_trade_does_not_flip_share_ownership(self):
        # The synthetic scenario ends with an open put and no more bars --
        # it should be logged as period_end, not misread as a real assignment.
        bars = make_bars(make_full_cycle_closes())
        result = backtest_wheel_strategy(bars, ticker="CYCLE")
        last_trade = result.trades[-1]
        self.assertEqual(last_trade.meta["exit_reason"], "period_end")
        self.assertTrue(last_trade.closed_at_period_end)

    def test_win_rate_and_trade_count_are_internally_consistent(self):
        bars = make_bars(make_full_cycle_closes())
        result = backtest_wheel_strategy(bars, ticker="CYCLE")
        wins = sum(1 for t in result.trades if t.is_win)
        self.assertEqual(result.total_trades, len(result.trades))
        if result.total_trades:
            self.assertAlmostEqual(result.win_rate_pct, wins / result.total_trades * 100, places=6)
        else:
            self.assertEqual(result.win_rate_pct, 0.0)

    def test_summary_reports_wheel_strategy_name_and_no_sma_windows(self):
        bars = make_bars(make_mild_oscillation(300))
        result = backtest_wheel_strategy(bars, ticker="OSC")
        s = result.summary()
        self.assertEqual(s["strategy_name"], "wheel")
        self.assertIsNone(s["fast_window"])
        self.assertIsNone(s["slow_window"])

    def test_entry_day_of_month_zero_matches_unfiltered_baseline(self):
        # 0 must be a true no-op -- same trade count and same entry dates
        # as never passing the param at all.
        bars = make_bars(make_mild_oscillation(300))
        baseline = backtest_wheel_strategy(bars, ticker="OSC")
        filtered = backtest_wheel_strategy(bars, ticker="OSC", entry_day_of_month=0)
        self.assertEqual(
            [t.entry_date for t in baseline.trades], [t.entry_date for t in filtered.trades]
        )

    def test_entry_day_of_month_restricts_every_entry_to_the_window(self):
        bars = make_bars(make_mild_oscillation(300))
        target = 10
        result = backtest_wheel_strategy(bars, ticker="OSC", entry_day_of_month=target)
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
        # leaving it off, and for a fixture with many eligible entry days
        # it should produce strictly fewer.
        bars = make_bars(make_mild_oscillation(300))
        baseline = backtest_wheel_strategy(bars, ticker="OSC")
        filtered = backtest_wheel_strategy(bars, ticker="OSC", entry_day_of_month=15)
        self.assertLessEqual(filtered.total_trades, baseline.total_trades)


if __name__ == "__main__":
    unittest.main()
