"""Unit tests for app.data_provider's "offline" source -- list/load a
pre-fetched, on-disk dataset. No network, no yfinance; each test points
OFFLINE_DATA_DIR at a temp directory of hand-written fixture csvs (same
schema app.admin_fetch.save_ticker_csv writes: date,open,high,low,close,
volume) via monkeypatching, then restores it, following the stdlib
unittest convention used throughout this test suite."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import data_provider


def _write_fixture_csv(dir_path: Path, ticker: str, rows=5, start="2020-01-01"):
    import pandas as pd

    dates = pd.date_range(start, periods=rows, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [1.0 + i for i in range(rows)],
            "high": [2.0 + i for i in range(rows)],
            "low": [0.5 + i for i in range(rows)],
            "close": [1.5 + i for i in range(rows)],
            "volume": [1000 + i for i in range(rows)],
        }
    )
    df.to_csv(dir_path / f"{ticker.upper()}.csv", index=False)


class OfflineDataDirTestCase(unittest.TestCase):
    """Points data_provider.OFFLINE_DATA_DIR at a fresh temp dir for the
    duration of each test, then restores the real path."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_dir = data_provider.OFFLINE_DATA_DIR
        data_provider.OFFLINE_DATA_DIR = Path(self._tmpdir.name)

    def tearDown(self):
        data_provider.OFFLINE_DATA_DIR = self._orig_dir
        self._tmpdir.cleanup()


class TestListOfflineTickers(OfflineDataDirTestCase):
    def test_empty_dir_returns_empty_list(self):
        self.assertEqual(data_provider.list_offline_tickers(), [])

    def test_missing_dir_returns_empty_list_not_raise(self):
        data_provider.OFFLINE_DATA_DIR = Path(self._tmpdir.name) / "does_not_exist"
        self.assertEqual(data_provider.list_offline_tickers(), [])

    def test_lists_sorted_tickers(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "MSFT")
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL")
        self.assertEqual(data_provider.list_offline_tickers(), ["AAPL", "MSFT"])


class TestLoadOfflineTicker(OfflineDataDirTestCase):
    def test_loads_expected_shape(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL", rows=5)
        df = data_provider.load_offline_ticker("aapl")
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(df.index.name, "date")

    def test_missing_ticker_raises_with_count_not_full_list(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL")
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "MSFT")
        with self.assertRaises(ValueError) as ctx:
            data_provider.load_offline_ticker("NVDA")
        msg = str(ctx.exception)
        self.assertIn("2 ticker(s) available", msg)
        # Must not dump the whole ticker list into the message.
        self.assertNotIn("AAPL", msg)

    def test_trims_to_date_range(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL", rows=10)
        df = data_provider.load_offline_ticker("AAPL", start="2020-01-05", end="2020-01-07")
        self.assertEqual(len(df), 3)

    def test_resamples_to_weekly(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL", rows=14)
        daily = data_provider.load_offline_ticker("AAPL", interval="1d")
        weekly = data_provider.load_offline_ticker("AAPL", interval="1wk")
        self.assertLess(len(weekly), len(daily))


class TestLoadOfflineUniverse(OfflineDataDirTestCase):
    def test_loads_only_requested_tickers(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL")
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "MSFT")
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "NVDA")
        universe = data_provider.load_offline_universe(["aapl", "msft"])
        self.assertEqual(set(universe.keys()), {"AAPL", "MSFT"})

    def test_missing_tickers_are_silently_skipped(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL")
        universe = data_provider.load_offline_universe(["AAPL", "GHOST"])
        self.assertEqual(set(universe.keys()), {"AAPL"})

    def test_empty_request_returns_empty_dict(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL")
        self.assertEqual(data_provider.load_offline_universe([]), {})


class TestGetPriceHistoryOfflineDispatch(OfflineDataDirTestCase):
    def test_source_offline_routes_to_load_offline_ticker(self):
        _write_fixture_csv(data_provider.OFFLINE_DATA_DIR, "AAPL", rows=5)
        df = data_provider.get_price_history("AAPL", source="offline")
        self.assertEqual(len(df), 5)

    def test_unknown_source_error_mentions_offline(self):
        with self.assertRaises(ValueError) as ctx:
            data_provider.get_price_history("AAPL", source="bogus")
        self.assertIn("offline", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
