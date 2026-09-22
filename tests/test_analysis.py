"""Unit tests for engine.analysis.analyze_underperformance. Uses stdlib
unittest only.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.analysis import analyze_underperformance
from engine.backtester import Trade


def _ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def _dates(n, start=date(2020, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _series(dates, values):
    return pd.Series(values, index=[_ts(d) for d in dates], dtype=float)


def _df(dates, closes):
    return pd.DataFrame({"close": closes}, index=[_ts(d) for d in dates])


class TestNoUnderperformance(unittest.TestCase):
    def test_empty_trades_returns_empty(self):
        dates = _dates(4)
        equity = _series(dates, [10000, 10500, 11000, 11500])
        buy_hold = _series(dates, [10000, 10200, 10400, 10600])
        df = _df(dates, [100, 102, 104, 106])
        self.assertEqual(analyze_underperformance([], equity, buy_hold, df), [])

    def test_strategy_always_ahead_produces_no_records(self):
        dates = _dates(4)
        equity = _series(dates, [10000, 10500, 11000, 11500])
        buy_hold = _series(dates, [10000, 10200, 10400, 10600])
        df = _df(dates, [100, 102, 104, 106])
        trade = Trade(
            entry_date=_ts(dates[0]),
            entry_price=100,
            exit_date=_ts(dates[3]),
            exit_price=115,
            return_pct=15.0,
            is_win=True,
        )
        self.assertEqual(analyze_underperformance([trade], equity, buy_hold, df), [])


class TestTradeLossCause(unittest.TestCase):
    def test_losing_trade_behind_buy_hold_is_explained_by_the_loss(self):
        dates = _dates(3)
        df = _df(dates, [100, 100, 90])  # -10% over the trade's own window
        equity = _series(dates, [10000, 10000, 9000])  # -10%
        buy_hold = _series(dates, [10000, 10500, 11000])  # +10%
        trade = Trade(
            entry_date=_ts(dates[0]),
            entry_price=100,
            exit_date=_ts(dates[2]),
            exit_price=90,
            return_pct=-10.0,
            is_win=False,
        )
        records = analyze_underperformance([trade], equity, buy_hold, df)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertLess(rec["gap_pct"], 0)
        self.assertAlmostEqual(rec["strategy_cumulative_pct"], -10.0, places=6)
        self.assertAlmostEqual(rec["buy_hold_cumulative_pct"], 10.0, places=6)
        self.assertIn("lost -10.0%", rec["explanation"])
        self.assertIn("underlying moved -10.0%", rec["explanation"])


class TestPartialCaptureCause(unittest.TestCase):
    def test_winning_trade_that_captures_only_part_of_the_move_is_explained(self):
        dates = _dates(3)
        df = _df(dates, [100, 105, 110])  # underlying +10% over the trade
        equity = _series(dates, [10000, 10000, 10300])  # strategy only +3%
        buy_hold = _series(dates, [10000, 10500, 11000])  # buy & hold +10%
        trade = Trade(
            entry_date=_ts(dates[0]),
            entry_price=1.0,
            exit_date=_ts(dates[2]),
            exit_price=0.5,
            return_pct=3.0,
            is_win=True,
            meta={"exit_reason": "profit_target"},
        )
        records = analyze_underperformance([trade], equity, buy_hold, df)
        self.assertEqual(len(records), 1)
        explanation = records[0]["explanation"]
        self.assertIn("gained +3.0%", explanation)
        self.assertIn("rose +10.0%", explanation)
        self.assertIn("profit target", explanation)

    def test_winning_trade_that_fully_captures_the_move_is_not_flagged_for_that_reason(self):
        # Trade return matches the underlying's move closely (within the
        # materiality threshold) -- shouldn't blame the trade itself.
        dates = _dates(3)
        df = _df(dates, [100, 103, 105])  # +5% over the trade
        equity = _series(dates, [10000, 10000, 10500])  # strategy +5%, matches
        buy_hold = _series(dates, [10000, 10800, 12000])  # buy & hold ends up further ahead
        trade = Trade(
            entry_date=_ts(dates[0]),
            entry_price=1.0,
            exit_date=_ts(dates[2]),
            exit_price=0.5,
            return_pct=5.0,
            is_win=True,
        )
        records = analyze_underperformance([trade], equity, buy_hold, df)
        self.assertEqual(len(records), 1)
        self.assertNotIn("gained", records[0]["explanation"])


class TestExposureGapCause(unittest.TestCase):
    def test_time_out_of_market_before_entry_is_explained(self):
        dates = _dates(8)
        # Underlying rises 15% over days 0-5 while flat, then the trade
        # (days 5-7) tracks the underlying almost exactly (+5%, no
        # trade-level cause to flag).
        closes = [100, 105, 108, 110, 112, 115, 118, 120.75]
        df = _df(dates, closes)
        equity = _series(dates, [10000] * 6 + [10000, 10500])  # jumps only at exit (day 7)
        buy_hold = _series(dates, [c / 100 * 10000 for c in closes])
        trade = Trade(
            entry_date=_ts(dates[5]),
            entry_price=1.0,
            exit_date=_ts(dates[7]),
            exit_price=0.5,
            return_pct=5.0,
            is_win=True,
        )
        records = analyze_underperformance([trade], equity, buy_hold, df)
        self.assertEqual(len(records), 1)
        explanation = records[0]["explanation"]
        self.assertIn("flat (out of the market)", explanation)
        self.assertIn("rose +15.0%", explanation)
        # The trade itself tracked the underlying almost exactly, so it
        # should not also be blamed for a partial capture.
        self.assertNotIn("gained", explanation)

    def test_no_gap_reason_when_entry_immediately_follows_prior_exit(self):
        dates = _dates(5)
        df = _df(dates, [100, 100, 100, 105, 110])
        equity = _series(dates, [10000, 9000, 9000, 9000, 9450])
        buy_hold = _series(dates, [10000, 10000, 10000, 10500, 11000])
        trades = [
            Trade(
                entry_date=_ts(dates[0]),
                entry_price=1.0,
                exit_date=_ts(dates[1]),
                exit_price=1.1,
                return_pct=-10.0,
                is_win=False,
            ),
            Trade(
                # Enters exactly when the first trade exited -- no flat gap.
                entry_date=_ts(dates[1]),
                entry_price=1.0,
                exit_date=_ts(dates[4]),
                exit_price=0.5,
                return_pct=5.0,
                is_win=True,
            ),
        ]
        records = analyze_underperformance(trades, equity, buy_hold, df)
        # Both trades leave the strategy behind buy & hold at their exit.
        self.assertEqual(len(records), 2)
        for rec in records:
            self.assertNotIn("flat (out of the market)", rec["explanation"])


class TestFallbackReason(unittest.TestCase):
    def test_falls_back_to_a_generic_explanation_when_no_single_cause_applies(self):
        dates = _dates(2)
        df = _df(dates, [100, 101])  # trade tracks the underlying closely
        equity = _series(dates, [10000, 10100])  # strategy +1%
        buy_hold = _series(dates, [10000, 10500])  # buy & hold +5%, decoupled from df on purpose
        trade = Trade(
            entry_date=_ts(dates[0]),  # == window start, so no exposure gap
            entry_price=1.0,
            exit_date=_ts(dates[1]),
            exit_price=0.99,
            return_pct=1.0,
            is_win=True,
        )
        records = analyze_underperformance([trade], equity, buy_hold, df)
        self.assertEqual(len(records), 1)
        self.assertIn("without one", records[0]["explanation"])


class TestOrderingAndMultipleTrades(unittest.TestCase):
    def test_records_are_returned_in_exit_date_order_regardless_of_input_order(self):
        dates = _dates(6)
        df = _df(dates, [100, 95, 90, 85, 80, 75])  # steady decline
        equity = _series(dates, [10000, 9500, 9000, 8500, 8000, 7500])
        buy_hold = _series(dates, [10000, 10500, 11000, 11500, 12000, 12500])
        trade_a = Trade(
            entry_date=_ts(dates[0]), entry_price=1, exit_date=_ts(dates[2]),
            exit_price=1, return_pct=-10.0, is_win=False,
        )
        trade_b = Trade(
            entry_date=_ts(dates[2]), entry_price=1, exit_date=_ts(dates[4]),
            exit_price=1, return_pct=-10.0, is_win=False,
        )
        records = analyze_underperformance([trade_b, trade_a], equity, buy_hold, df)
        self.assertEqual([r["exit_date"] for r in records], [_ts(dates[2]), _ts(dates[4])])


if __name__ == "__main__":
    unittest.main()
