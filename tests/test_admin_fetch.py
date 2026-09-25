"""Unit tests for app.admin_fetch -- pure logic only (ticker-list parsing,
keep/skip decisions, csv writing, zipping). market_cap_fn/history_fn are
always injected as fakes so nothing here touches the network (same
convention as tests/test_news_provider.py: stdlib unittest, hand-built
fixtures, no mocking framework)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.admin_fetch import (
    fetch_batch,
    fetch_one_ticker,
    load_ticker_universe,
    save_ticker_csv,
    zip_offline_dataset,
)
from engine.data_utils import to_dataframe


def _fake_history(ticker: str, years: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return to_dataframe(
        pd.DataFrame(
            {
                "date": dates,
                "open": [1.0] * 5,
                "high": [2.0] * 5,
                "low": [0.5] * 5,
                "close": [1.5] * 5,
                "volume": [1000] * 5,
            }
        )
    )


def _failing_history(ticker: str, years: int) -> pd.DataFrame:
    raise ValueError(f"yfinance returned no data for '{ticker}'")


class TestLoadTickerUniverse(unittest.TestCase):
    def test_reads_seed_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tickers.csv"
            path.write_text("ticker\naapl\nMSFT\n nvda \n")
            result = load_ticker_universe(path)
            self.assertEqual(result, ["AAPL", "MSFT", "NVDA"])

    def test_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "does_not_exist.csv"
            self.assertEqual(load_ticker_universe(path), [])

    def test_missing_ticker_column_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tickers.csv"
            path.write_text("symbol\nAAPL\n")
            self.assertEqual(load_ticker_universe(path), [])

    def test_blank_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tickers.csv"
            path.write_text("ticker\nAAPL\n\nMSFT\n")
            self.assertEqual(load_ticker_universe(path), ["AAPL", "MSFT"])


class TestFetchOneTicker(unittest.TestCase):
    def test_saved_when_cap_ok(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            result = fetch_one_ticker(
                "aapl",
                min_market_cap=1_000_000_000.0,
                out_dir=out_dir,
                market_cap_fn=lambda t: 2_000_000_000.0,
                history_fn=_fake_history,
            )
            self.assertEqual(result["status"], "saved")
            self.assertEqual(result["ticker"], "AAPL")
            self.assertEqual(result["rows"], 5)
            self.assertTrue((out_dir / "AAPL.csv").exists())

    def test_skipped_below_min_market_cap(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            result = fetch_one_ticker(
                "smol",
                min_market_cap=1_000_000_000.0,
                out_dir=out_dir,
                market_cap_fn=lambda t: 500_000_000.0,
                history_fn=_fake_history,
            )
            self.assertEqual(result["status"], "skipped_small_cap")
            self.assertEqual(result["market_cap"], 500_000_000.0)
            # History must never have been fetched/written for a skip.
            self.assertFalse((out_dir / "SMOL.csv").exists())

    def test_unknown_market_cap_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            result = fetch_one_ticker(
                "unk",
                min_market_cap=1_000_000_000.0,
                out_dir=out_dir,
                market_cap_fn=lambda t: None,
                history_fn=_fake_history,
            )
            self.assertEqual(result["status"], "saved")

    def test_zero_min_market_cap_skips_the_lookup_entirely(self):
        calls = []

        def _tracking_cap_fn(t):
            calls.append(t)
            return 0.0

        with tempfile.TemporaryDirectory() as td:
            fetch_one_ticker(
                "any",
                min_market_cap=0,
                out_dir=Path(td),
                market_cap_fn=_tracking_cap_fn,
                history_fn=_fake_history,
            )
            self.assertEqual(calls, [])

    def test_failed_status_on_history_exception(self):
        with tempfile.TemporaryDirectory() as td:
            result = fetch_one_ticker(
                "bad",
                min_market_cap=0,
                out_dir=Path(td),
                market_cap_fn=lambda t: None,
                history_fn=_failing_history,
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("no data", result["error"])

    def test_failed_status_when_market_cap_fn_raises(self):
        def _raising_cap_fn(t):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as td:
            result = fetch_one_ticker(
                "bad",
                min_market_cap=1_000_000_000.0,
                out_dir=Path(td),
                market_cap_fn=_raising_cap_fn,
                history_fn=_fake_history,
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("boom", result["error"])


class TestFetchBatch(unittest.TestCase):
    def test_calls_progress_cb_for_each_ticker(self):
        calls = []

        with tempfile.TemporaryDirectory() as td:
            results = fetch_batch(
                ["AAA", "BBB", "CCC"],
                min_market_cap=0,
                out_dir=Path(td),
                market_cap_fn=lambda t: None,
                history_fn=_fake_history,
                progress_cb=lambda done, total: calls.append((done, total)),
            )
            self.assertEqual(len(results), 3)
            self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])

    def test_one_failure_does_not_stop_the_batch(self):
        def _history_fn(ticker, years):
            if ticker == "BBB":
                raise ValueError("boom")
            return _fake_history(ticker, years)

        with tempfile.TemporaryDirectory() as td:
            results = fetch_batch(
                ["AAA", "BBB", "CCC"],
                min_market_cap=0,
                out_dir=Path(td),
                market_cap_fn=lambda t: None,
                history_fn=_history_fn,
            )
            statuses = {r["ticker"]: r["status"] for r in results}
            self.assertEqual(statuses, {"AAA": "saved", "BBB": "failed", "CCC": "saved"})

    def test_no_progress_cb_is_fine(self):
        with tempfile.TemporaryDirectory() as td:
            results = fetch_batch(
                ["AAA"],
                min_market_cap=0,
                out_dir=Path(td),
                market_cap_fn=lambda t: None,
                history_fn=_fake_history,
            )
            self.assertEqual(len(results), 1)


class TestSaveTickerCsv(unittest.TestCase):
    def test_writes_expected_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            df = _fake_history("AAPL", 10)
            path = save_ticker_csv(df, "aapl", out_dir=out_dir)
            self.assertEqual(path, out_dir / "AAPL.csv")
            written = pd.read_csv(path)
            self.assertEqual(
                list(written.columns), ["date", "open", "high", "low", "close", "volume"]
            )
            self.assertEqual(written["date"].iloc[0], "2020-01-01")

    def test_round_trips_through_load_offline_ticker_schema(self):
        # save_ticker_csv's output must be readable by to_dataframe the same
        # way the bundled sample csvs are (data_provider.load_offline_ticker
        # relies on this).
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            df = _fake_history("AAPL", 10)
            path = save_ticker_csv(df, "AAPL", out_dir=out_dir)
            reloaded = to_dataframe(pd.read_csv(path))
            self.assertEqual(len(reloaded), len(df))
            self.assertEqual(list(reloaded.columns), list(df.columns))


class TestZipOfflineDataset(unittest.TestCase):
    def test_contains_saved_csvs(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            save_ticker_csv(_fake_history("AAA", 10), "AAA", out_dir=out_dir)
            save_ticker_csv(_fake_history("BBB", 10), "BBB", out_dir=out_dir)

            zip_bytes = zip_offline_dataset(out_dir=out_dir)
            with tempfile.TemporaryDirectory() as td2:
                zip_path = Path(td2) / "out.zip"
                zip_path.write_bytes(zip_bytes)
                with zipfile.ZipFile(zip_path) as zf:
                    names = sorted(zf.namelist())
            self.assertEqual(names, ["AAA.csv", "BBB.csv"])

    def test_empty_dir_produces_empty_zip(self):
        with tempfile.TemporaryDirectory() as td:
            zip_bytes = zip_offline_dataset(out_dir=Path(td))
            with tempfile.TemporaryDirectory() as td2:
                zip_path = Path(td2) / "out.zip"
                zip_path.write_bytes(zip_bytes)
                with zipfile.ZipFile(zip_path) as zf:
                    self.assertEqual(zf.namelist(), [])

    def test_nonexistent_dir_does_not_raise(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist"
            zip_bytes = zip_offline_dataset(out_dir=missing)
            self.assertIsInstance(zip_bytes, bytes)


if __name__ == "__main__":
    unittest.main()
