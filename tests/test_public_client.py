"""Unit tests for app/public_client.py (the Public.com brokerage API
client). All HTTP is mocked -- these never touch the network, so they run
the same offline as every other test in this suite.

Run with: python3 -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import public_client


def _resp(status_code=200, json_data=None, text=""):
    m = MagicMock()
    m.status_code = status_code
    m.ok = 200 <= status_code < 300
    m.json.return_value = json_data if json_data is not None else {}
    m.text = text
    return m


class PublicClientTestCase(unittest.TestCase):
    def setUp(self):
        # Module-level caches must not leak between tests.
        public_client._token_cache["access_token"] = None
        public_client._token_cache["expires_at"] = 0.0
        public_client._account_id_cache["account_id"] = None
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop("PUBLIC_API_SECRET", None)

    def tearDown(self):
        self._env_patch.stop()


class TestSecretResolution(PublicClientTestCase):
    def test_no_secret_configured_returns_none(self):
        self.assertIsNone(public_client.get_secret())
        self.assertFalse(public_client.has_secret())

    def test_env_var_secret_is_used(self):
        os.environ["PUBLIC_API_SECRET"] = "abc123"
        self.assertEqual(public_client.get_secret(), "abc123")
        self.assertTrue(public_client.has_secret())

    def test_require_secret_raises_clear_error_when_missing(self):
        with self.assertRaises(public_client.PublicApiError) as ctx:
            public_client._require_secret()
        self.assertIn("PUBLIC_API_SECRET", str(ctx.exception))


class TestAccessToken(PublicClientTestCase):
    def test_exchanges_secret_for_token_and_caches_it(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with patch("app.public_client.requests.post") as mock_post:
            mock_post.return_value = _resp(200, {"accessToken": "tok-1"})
            token = public_client._get_access_token()
        self.assertEqual(token, "tok-1")
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        self.assertIn("/userapiauthservice/personal/access-tokens", called_url)
        self.assertEqual(mock_post.call_args[1]["json"]["secret"], "my-secret")

        # Second call within validity window should NOT re-hit the network.
        with patch("app.public_client.requests.post") as mock_post2:
            token2 = public_client._get_access_token()
        mock_post2.assert_not_called()
        self.assertEqual(token2, "tok-1")

    def test_401_from_auth_raises_clear_error(self):
        os.environ["PUBLIC_API_SECRET"] = "bad-secret"
        with patch("app.public_client.requests.post") as mock_post:
            mock_post.return_value = _resp(401, {}, text="unauthorized")
            with self.assertRaises(public_client.PublicApiError) as ctx:
                public_client._get_access_token()
        self.assertIn("401", str(ctx.exception)) or self.assertIn(
            "rejected", str(ctx.exception).lower()
        )


class TestRequestRetryOn401(PublicClientTestCase):
    def test_expired_token_triggers_one_reauth_retry(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        public_client._token_cache["access_token"] = "stale-tok"
        public_client._token_cache["expires_at"] = 9_999_999_999.0  # looks fresh

        auth_resp = _resp(200, {"accessToken": "fresh-tok"})
        first_call_resp = _resp(401, {}, text="expired")
        second_call_resp = _resp(200, {"ok": True})

        with patch("app.public_client.requests.post", return_value=auth_resp), patch(
            "app.public_client.requests.request", side_effect=[first_call_resp, second_call_resp]
        ) as mock_request:
            resp = public_client._request("GET", "/some/path")

        self.assertTrue(resp.ok)
        self.assertEqual(mock_request.call_count, 2)
        # The retried call should carry the freshly re-exchanged token.
        second_headers = mock_request.call_args_list[1][1]["headers"]
        self.assertEqual(second_headers["Authorization"], "Bearer fresh-tok")


class TestGetAccountId(PublicClientTestCase):
    def test_parses_and_caches_first_account_id(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with patch(
            "app.public_client._request",
            return_value=_resp(200, {"accounts": [{"accountId": "acct-1"}, {"accountId": "acct-2"}]}),
        ) as mock_req:
            acct = public_client.get_account_id()
        self.assertEqual(acct, "acct-1")
        mock_req.assert_called_once()

        # Cached -- a second call shouldn't hit _request again.
        with patch("app.public_client._request") as mock_req2:
            acct2 = public_client.get_account_id()
        mock_req2.assert_not_called()
        self.assertEqual(acct2, "acct-1")

    def test_no_accounts_raises_clear_error(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with patch("app.public_client._request", return_value=_resp(200, {"accounts": []})):
            with self.assertRaises(public_client.PublicApiError):
                public_client.get_account_id()


class TestSmallestCoveringPeriod(unittest.TestCase):
    def test_none_start_defaults_to_five_year(self):
        self.assertEqual(public_client._smallest_covering_period(None, None), "FIVE_YEAR")

    def test_short_window_picks_small_period(self):
        # A 9-day span doesn't fit in WEEK (7 days), so it should escalate
        # to the next-smallest period that covers it.
        self.assertEqual(
            public_client._smallest_covering_period("2026-09-01", "2026-09-10"), "MONTH"
        )

    def test_multi_year_window_picks_five_year(self):
        self.assertEqual(
            public_client._smallest_covering_period("2022-01-01", "2026-09-01"), "FIVE_YEAR"
        )

    def test_very_long_window_falls_back_to_all(self):
        self.assertEqual(
            public_client._smallest_covering_period("1990-01-01", "2026-09-01"), "ALL"
        )


class TestFetchPublicBars(PublicClientTestCase):
    def test_builds_canonical_dataframe_trimmed_to_range(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        bars_payload = {
            "regularMarket": {
                "bars": [
                    {"timestamp": "2026-01-02", "open": "10", "high": "11", "low": "9", "close": "10.5", "volume": 100},
                    {"timestamp": "2026-01-03", "open": "10.5", "high": "12", "low": "10", "close": "11.5", "volume": 200},
                    {"timestamp": "2026-01-04", "open": "11.5", "high": "13", "low": "11", "close": "12.5", "volume": 300},
                ]
            }
        }
        with patch("app.public_client._request", return_value=_resp(200, bars_payload)) as mock_req:
            df = public_client.fetch_public_bars(
                "aapl", start="2026-01-03", end="2026-01-04", interval="1d"
            )
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]["close"], 11.5)
        self.assertAlmostEqual(df.iloc[-1]["close"], 12.5)
        # URL should carry the upper-cased ticker and ONE_DAY aggregation.
        called_path = mock_req.call_args[0][1]
        self.assertIn("AAPL", called_path)
        self.assertIn("ONE_DAY", called_path)

    def test_unsupported_interval_raises(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with self.assertRaises(public_client.PublicApiError):
            public_client.fetch_public_bars("AAPL", interval="5m")

    def test_empty_bars_raises_clear_error(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with patch(
            "app.public_client._request", return_value=_resp(200, {"regularMarket": {"bars": []}})
        ):
            with self.assertRaises(public_client.PublicApiError):
                public_client.fetch_public_bars("AAPL")


class TestOptionChain(PublicClientTestCase):
    def _mock_account(self):
        return patch("app.public_client.get_account_id", return_value="acct-1")

    def test_fetch_option_expirations_returns_list(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        with self._mock_account(), patch(
            "app.public_client._request",
            return_value=_resp(200, {"baseSymbol": "AAPL", "expirations": ["2026-10-16", "2026-11-20"]}),
        ):
            expirations = public_client.fetch_option_expirations("AAPL")
        self.assertEqual(expirations, ["2026-10-16", "2026-11-20"])

    def test_fetch_option_chain_flattens_and_sorts_by_strike(self):
        os.environ["PUBLIC_API_SECRET"] = "my-secret"
        payload = {
            "calls": [
                {
                    "instrument": {"symbol": "AAPL...C00160000"},
                    "bid": "2.40", "ask": "2.50", "last": "2.45",
                    "volume": 500, "openInterest": 1000,
                    "optionDetails": {
                        "strikePrice": "160.00",
                        "greeks": {"delta": "0.65", "impliedVolatility": "0.18"},
                    },
                },
                {
                    "instrument": {"symbol": "AAPL...C00150000"},
                    "bid": "5.40", "ask": "5.50", "last": "5.45",
                    "volume": 300, "openInterest": 900,
                    "optionDetails": {
                        "strikePrice": "150.00",
                        "greeks": {"delta": "0.80", "impliedVolatility": "0.20"},
                    },
                },
            ],
            "puts": [],
        }
        with self._mock_account(), patch("app.public_client._request", return_value=_resp(200, payload)):
            chain = public_client.fetch_option_chain("AAPL", "2026-11-20")

        self.assertEqual(len(chain["calls"]), 2)
        # Sorted ascending by strike: 150 before 160.
        self.assertEqual(chain["calls"][0]["strike"], 150.00)
        self.assertEqual(chain["calls"][1]["strike"], 160.00)
        self.assertEqual(chain["calls"][0]["delta"], 0.80)
        self.assertEqual(chain["puts"], [])


if __name__ == "__main__":
    unittest.main()
