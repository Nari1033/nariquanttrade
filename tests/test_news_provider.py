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


def _item(
    title="Some routine headline",
    summary="",
    overall_sentiment_score=0.0,
    ticker="ABC",
    relevance=1.0,
):
    """Small builder for hand-crafted NewsItem fixtures below -- avoids
    repeating every field for each test case."""
    from app.news_provider import NewsItem, TickerSentiment

    return NewsItem(
        title=title,
        url=f"https://example.com/{hash(title) & 0xffffff}",
        source="Test Wire",
        time_published=dt.datetime(2026, 9, 24, 12, 0, 0),
        summary=summary,
        overall_sentiment_score=overall_sentiment_score,
        overall_sentiment_label="Neutral",
        tickers=[
            TickerSentiment(
                ticker=ticker,
                relevance_score=relevance,
                sentiment_score=overall_sentiment_score,
                sentiment_label="Neutral",
            )
        ]
        if ticker
        else [],
    )


class TestImpactScore(unittest.TestCase):
    def test_catalyst_keyword_raises_score(self):
        from app.news_provider import impact_score

        plain = _item(title="XYZ Corp stock dips slightly")
        catalyst = _item(title="XYZ Corp wins FDA approval for new drug")
        self.assertGreater(impact_score(catalyst), impact_score(plain))

    def test_extreme_sentiment_raises_score_either_direction(self):
        from app.news_provider import impact_score

        neutral = _item(overall_sentiment_score=0.0)
        very_bullish = _item(overall_sentiment_score=0.9)
        very_bearish = _item(overall_sentiment_score=-0.9)
        self.assertGreater(impact_score(very_bullish), impact_score(neutral))
        self.assertGreater(impact_score(very_bearish), impact_score(neutral))
        # A strong bearish surprise is exactly as "worth a look" as an
        # equally strong bullish one -- score should be symmetric.
        self.assertAlmostEqual(impact_score(very_bullish), impact_score(very_bearish))

    def test_low_relevance_lowers_score(self):
        from app.news_provider import impact_score

        squarely_about_it = _item(relevance=1.0)
        mentioned_in_passing = _item(relevance=0.1)
        self.assertGreater(impact_score(squarely_about_it), impact_score(mentioned_in_passing))

    def test_matched_catalysts_lists_hits(self):
        from app.news_provider import matched_catalysts

        item = _item(title="Company announces bankruptcy filing", summary="Chapter 11 planned")
        hits = matched_catalysts(item)
        self.assertIn("bankruptcy", hits)
        self.assertIn("chapter 11", hits)
        self.assertNotIn("fda approv", hits)


class TestTopMovers(unittest.TestCase):
    def test_ranks_by_impact_score_descending(self):
        from app.news_provider import top_movers

        low = _item(title="Routine update", ticker="AAA")
        high = _item(title="AAA wins FDA approval for flagship drug", ticker="AAA", overall_sentiment_score=0.6)
        result = top_movers([low, high], n=5)
        self.assertEqual(result[0], high)

    def test_caps_at_n(self):
        from app.news_provider import top_movers

        items = [_item(title=f"Ticker{i} wins FDA approval", ticker=f"T{i}") for i in range(10)]
        self.assertEqual(len(top_movers(items, n=5)), 5)

    def test_near_duplicate_same_ticker_collapsed_to_one(self):
        from app.news_provider import top_movers

        a = _item(title="Acme Corp wins FDA approval for new drug", ticker="ACME", overall_sentiment_score=0.5)
        b = _item(title="Acme Corp wins FDA approval for its new drug", ticker="ACME", overall_sentiment_score=0.4)
        filler = [_item(title=f"Other{i} routine update", ticker=f"O{i}") for i in range(5)]
        result = top_movers([a, b] + filler, n=5)
        acme_titles = [i for i in result if i.tickers and i.tickers[0].ticker == "ACME"]
        self.assertEqual(len(acme_titles), 1)
        # The higher-scoring (more bullish) of the pair is the one kept.
        self.assertEqual(acme_titles[0], a)

    def test_same_headline_different_ticker_not_deduped(self):
        from app.news_provider import top_movers

        a = _item(title="Company wins FDA approval for new drug", ticker="AAA", overall_sentiment_score=0.5)
        b = _item(title="Company wins FDA approval for new drug", ticker="BBB", overall_sentiment_score=0.5)
        result = top_movers([a, b], n=5)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        from app.news_provider import top_movers

        self.assertEqual(top_movers([], n=5), [])
