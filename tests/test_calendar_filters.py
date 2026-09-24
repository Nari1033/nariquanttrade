"""Unit tests for engine.calendar_filters.day_of_month_ok -- pure logic,
no price data or backtester involved (see test_cash_secured_put.py,
test_options.py, and test_wheel.py for each engine actually gating
entries on this)."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.calendar_filters import DEFAULT_WINDOW_DAYS, day_of_month_ok


class TestDayOfMonthOk(unittest.TestCase):
    def test_zero_disables_filter_for_any_day(self):
        for day in (1, 10, 15, 20, 28, 31):
            dt = pd.Timestamp(2024, 1, day)
            self.assertTrue(day_of_month_ok(dt, entry_day_of_month=0))

    def test_exact_target_day_matches(self):
        dt = pd.Timestamp(2024, 1, 15)
        self.assertTrue(day_of_month_ok(dt, entry_day_of_month=15))

    def test_within_default_window_matches(self):
        target = 15
        for day in (target - DEFAULT_WINDOW_DAYS, target + DEFAULT_WINDOW_DAYS):
            dt = pd.Timestamp(2024, 1, day)
            self.assertTrue(day_of_month_ok(dt, entry_day_of_month=target))

    def test_outside_default_window_does_not_match(self):
        target = 15
        for day in (target - DEFAULT_WINDOW_DAYS - 1, target + DEFAULT_WINDOW_DAYS + 1):
            dt = pd.Timestamp(2024, 1, day)
            self.assertFalse(day_of_month_ok(dt, entry_day_of_month=target))

    def test_custom_window_is_respected(self):
        dt = pd.Timestamp(2024, 1, 20)  # 5 days from target 15
        self.assertFalse(day_of_month_ok(dt, entry_day_of_month=15, window=2))
        self.assertTrue(day_of_month_ok(dt, entry_day_of_month=15, window=5))

    def test_month_end_target_near_28th(self):
        # 28 is the only valid "month end" target (exists in every month,
        # including Feb), but should still match days a bit past it in a
        # 31-day month via the window.
        dt = pd.Timestamp(2024, 1, 31)
        self.assertTrue(day_of_month_ok(dt, entry_day_of_month=28))

    def test_invalid_target_raises(self):
        dt = pd.Timestamp(2024, 1, 15)
        for bad_target in (-1, 29, 30, 31, 100):
            with self.assertRaises(ValueError):
                day_of_month_ok(dt, entry_day_of_month=bad_target)


if __name__ == "__main__":
    unittest.main()
