"""Unit tests for app.news_provider -- pure parsing/error-detection logic
only, no network calls (same convention as tests/test_engine.py: hand-built
fixtures, stdlib unittest, no mocking framework needed)."""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.news_provider import (
    NewsFetchError,
    SENTIMENT_ORDER,
    _check_for_api_error,
    parse_feed,
    sentiment_distribution,
)

SAMPLE_FEED = {
    "items": "2",
    "feed": [
        {
            "title": "Example Co beats on Q3 earnings",
            "url": "https://example.com/a",
            "time_published": "20260924T020801",
            "summary": "Example Co reported strong Q3 results.",
            "source": "Example Wire",
            "overall_sentiment_score": 0.43,
            "overall_sentiment_label": "Bullish",
            "topics": [{"topic": "earnings", "relevance_score": "0.9"}],
            "ticker_sentiment": [
                {
                    "ticker": "EX",
                    "relevance_score": "1.0",
                    "ticker_sentiment_score": "0.41",
                    "ticker_sentiment_label": "Bullish",
                },
                {
                    "ticker": "EXB",
                    "relevance_score": "0.3",
                    "ticker_sentiment_score": "0.10",
                    "ticker_sentiment_label": "Neutral",
                },
            ],
        },
        {
            # Missing "title" -- must be skipped, not raise.
            "url": "https://example.com/b",
            "time_published": "20260924T010000",
            "overall_sentiment_score": -0.2,
            "overall_sentiment_label": "Somewhat-Bearish",
        },
    ],
}


class TestParseFeed(unittest.TestCase):
    def test_parses_valid_rows_only(self):
        items = parse_feed(SAMPLE_FEED)
        # The second row (missing "title") is dropped; only the first parses.
        self.assertEqual(len(items), 1)

    def test_fields_round_trip(self):
        item = parse_feed(SAMPLE_FEED)[0]
        self.assertEqual(item.title, "Example Co beats on Q3 earnings")
        self.assertEqual(item.source, "Example Wire")
        self.assertEqual(item.overall_sentiment_label, "Bullish")
        self.assertAlmostEqual(item.overall_sentiment_score, 0.43)
        self.assertEqual(item.time_published, dt.datetime(2026, 9, 24, 2, 8, 1))
        self.assertEqual(item.topics, ["earnings"])
        self.assertEqual(len(item.tickers), 2)

    def test_primary_ticker_is_most_relevant(self):
        item = parse_feed(SAMPLE_FEED)[0]
        # EX has relevance 1.0 vs EXB's 0.3 -- EX must win even though it's
        # not first by any other ordering.
        self.assertEqual(item.primary_ticker.ticker, "EX")

    def test_empty_or_missing_feed(self):
        self.assertEqual(parse_feed({}), [])
        self.assertEqual(parse_feed({"feed": []}), [])
        self.assertEqual(parse_feed({"feed": None}), [])


class TestApiErrorDetection(unittest.TestCase):
    def test_normal_body_does_not_raise(self):
        _check_for_api_error(SAMPLE_FEED)  # should not raise

    def test_rate_limit_note_raises(self):
        with self.assertRaises(NewsFetchError):
            _check_for_api_error({"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."})

    def test_bad_key_error_message_raises(self):
        with self.assertRaises(NewsFetchError):
            _check_for_api_error({"Error Message": "the parameter apikey is invalid"})

    def test_information_key_raises(self):
        with self.assertRaises(NewsFetchError):
            _check_for_api_error({"Information": "the parameter apikey is invalid"})

    def test_non_dict_raises(self):
        with self.assertRaises(NewsFetchError):
            _check_for_api_error([])


class TestSentimentDistribution(unittest.TestCase):
    def test_counts_every_bucket_including_zero(self):
        items = parse_feed(SAMPLE_FEED)  # 1 Bullish item
        counts = sentiment_distribution(items)
        self.assertEqual(set(counts.keys()), set(SENTIMENT_ORDER))
        self.assertEqual(counts["Bullish"], 1)
        self.assertEqual(counts["Bearish"], 0)
        self.assertEqual(sum(counts.values()), 1)

    def test_empty_items(self):
        counts = sentiment_distribution([])
        self.assertEqual(sum(counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
