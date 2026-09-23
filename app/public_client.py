"""Client for Public.com's brokerage API (https://public.com/api/docs),
used as this app's real market-data source -- replaces yfinance.

Two things this module provides:
  1. Real historical stock/ETF price bars (`fetch_public_bars`), which back
     the Scanner, Backtest, and Parameter Sweep tabs exactly like yfinance
     used to.
  2. A *live* current option chain with real bid/ask/greeks/open interest
     (`fetch_option_expirations` / `fetch_option_chain`), which backs the
     standalone "Live Option Chain" tab only.

Important limitation, by design: Public.com's API exposes a live/current
option chain snapshot, not historical options data (there's no endpoint for
"what was AAPL's option chain on 2023-01-05"). So the options *backtesters*
(Bull Put Spread, Cash-Secured Put, Wheel) keep using Black-Scholes pricing
from the underlying's own realized volatility, exactly as before -- this
module's option-chain functions are wired into their own informational tab
only, never into the backtest math.

Authentication flow (per the docs):
  - You generate a long-lived "secret" once, from your Public.com account
    settings (https://public.com/settings/security/api).
  - That secret is exchanged for a short-lived bearer access token via
    POST /userapiauthservice/personal/access-tokens -- this module does
    that exchange automatically and re-does it when the cached token is
    close to expiring, so callers never see a token directly.
  - Reading quotes/option-expirations/option-chain requires an `accountId`
    tied to your real Public.com brokerage account (this module fetches it
    once via GET /userapigateway/trading/account and caches it). This is a
    read-only market-data lookup -- nothing in this module ever calls an
    order/trading endpoint.

Getting the secret into this app -- NEVER hardcode it in source, since this
repo is public on GitHub:
  - Locally: put `PUBLIC_API_SECRET = "..."` in `.streamlit/secrets.toml`
    (already gitignored) or export it as the `PUBLIC_API_SECRET`
    environment variable before running `streamlit run app/app.py`.
  - On Streamlit Community Cloud: set `PUBLIC_API_SECRET` under the
    deployed app's Settings -> Secrets panel. Never commit it to the repo.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

API_BASE = "https://api.public.com"
_REQUEST_TIMEOUT_S = 20

# Re-auth this many seconds before the cached token's actual expiry, so a
# request never straddles the exact expiry instant.
_TOKEN_REFRESH_SLACK_S = 60
_TOKEN_VALIDITY_MINUTES = 30

# Smallest Public.com `period` enum value (in the "Get bars v2" family)
# whose calendar-day span covers a requested [start, end] window, in
# ascending order -- the app's date pickers let you ask for any arbitrary
# start/end, but Public.com's bars endpoint only offers these preset
# lookback windows, so we fetch the smallest one that covers the request
# and then trim to the exact [start, end] the caller asked for (same
# trim-after-fetch pattern the sample-data path already uses).
_PERIOD_DAYS = [
    ("DAY", 1),
    ("WEEK", 7),
    ("MONTH", 31),
    ("QUARTER", 93),
    ("HALF_YEAR", 183),
    ("YEAR", 365),
    ("FIVE_YEAR", 5 * 365),
    ("TEN_YEARS", 10 * 365),
    ("ALL", None),  # always covers everything
]

_INTERVAL_TO_AGGREGATION = {
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
    "1wk": "ONE_WEEK",
    "1mo": "ONE_MONTH",
}


class PublicApiError(RuntimeError):
    """Raised for anything that goes wrong talking to Public.com's API --
    missing secret, a non-2xx response, or a response shape this client
    doesn't recognize. Always carries a human-readable message suitable
    for showing directly in the Streamlit UI."""


# ---------------------------------------------------------------------------
# Secret / token / account plumbing
# ---------------------------------------------------------------------------

_token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}
_account_id_cache: Dict[str, Optional[str]] = {"account_id": None}


def get_secret() -> Optional[str]:
    """The long-lived Public.com secret, from (in order) the
    `PUBLIC_API_SECRET` environment variable, or Streamlit secrets if a
    `.streamlit/secrets.toml` / Streamlit Cloud "Secrets" entry defines it.
    None if neither is configured -- callers turn that into a clear error
    or a UI prompt rather than a confusing HTTP failure."""
    env_val = os.environ.get("PUBLIC_API_SECRET")
    if env_val:
        return env_val
    try:
        import streamlit as st

        val = st.secrets.get("PUBLIC_API_SECRET")
        if val:
            return str(val)
    except Exception:
        # No secrets.toml, streamlit not importable outside the app, or no
        # such key -- any of these just means "not configured this way".
        pass
    return None


def has_secret() -> bool:
    return bool(get_secret())


def _require_secret() -> str:
    secret = get_secret()
    if not secret:
        raise PublicApiError(
            "No Public.com API secret configured. Set the `PUBLIC_API_SECRET` "
            "environment variable, or add `PUBLIC_API_SECRET = \"...\"` to "
            ".streamlit/secrets.toml locally (or the app's Settings -> Secrets "
            "panel on Streamlit Community Cloud). Generate a secret at "
            "https://public.com/settings/security/api."
        )
    return secret


def _get_access_token(force_refresh: bool = False) -> str:
    """The cached bearer access token, refreshing it (by exchanging the
    secret again) if it's missing, close to expiry, or `force_refresh`."""
    now = time.time()
    if (
        not force_refresh
        and _token_cache["access_token"]
        and now < _token_cache["expires_at"] - _TOKEN_REFRESH_SLACK_S
    ):
        return _token_cache["access_token"]

    secret = _require_secret()
    resp = requests.post(
        f"{API_BASE}/userapiauthservice/personal/access-tokens",
        json={"secret": secret, "validityInMinutes": _TOKEN_VALIDITY_MINUTES},
        headers={"Content-Type": "application/json"},
        timeout=_REQUEST_TIMEOUT_S,
    )
    if resp.status_code == 401:
        raise PublicApiError(
            "Public.com rejected the configured PUBLIC_API_SECRET (401 Unauthorized). "
            "Double check the secret at https://public.com/settings/security/api."
        )
    if not resp.ok:
        raise PublicApiError(
            f"Public.com auth failed ({resp.status_code}): {resp.text[:300]}"
        )
    data = resp.json()
    token = data.get("accessToken")
    if not token:
        raise PublicApiError(f"Public.com auth response had no accessToken: {data!r}")

    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + _TOKEN_VALIDITY_MINUTES * 60
    return token


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}


def _request(method: str, path: str, retry_on_401: bool = True, **kwargs) -> requests.Response:
    """One authenticated request, transparently retrying once with a freshly
    re-exchanged token if the cached one turned out to be expired/invalid
    (a 401 mid-session, not just at the auth step itself)."""
    resp = requests.request(
        method, f"{API_BASE}{path}", headers=_auth_headers(), timeout=_REQUEST_TIMEOUT_S, **kwargs
    )
    if resp.status_code == 401 and retry_on_401:
        _get_access_token(force_refresh=True)
        return _request(method, path, retry_on_401=False, **kwargs)
    return resp


def get_account_id() -> str:
    """The first brokerage accountId on this Public.com account, cached
    after the first lookup. Read-only -- used only to scope market-data
    lookups (quotes, option chain, option expirations); never used for any
    order/trading endpoint."""
    if _account_id_cache["account_id"]:
        return _account_id_cache["account_id"]

    resp = _request("GET", "/userapigateway/trading/account")
    if not resp.ok:
        raise PublicApiError(
            f"Couldn't fetch your Public.com account ({resp.status_code}): {resp.text[:300]}"
        )
    accounts = resp.json().get("accounts") or []
    if not accounts:
        raise PublicApiError("Public.com returned no brokerage accounts for this API secret.")
    account_id = accounts[0].get("accountId")
    if not account_id:
        raise PublicApiError(f"Public.com account response had no accountId: {accounts[0]!r}")

    _account_id_cache["account_id"] = account_id
    return account_id


# ---------------------------------------------------------------------------
# Historical bars (stocks/ETFs) -- feeds the Scanner/Backtest/Sweep tabs
# ---------------------------------------------------------------------------


def _smallest_covering_period(start, end) -> str:
    if start is None:
        return "FIVE_YEAR"
    span_days = (pd.Timestamp(end or pd.Timestamp.today()) - pd.Timestamp(start)).days
    for period, days in _PERIOD_DAYS:
        if days is None or span_days <= days:
            return period
    return "ALL"


def fetch_public_bars(
    ticker: str,
    start=None,
    end=None,
    interval: str = "1d",
    instrument_type: str = "EQUITY",
    trading_session: str = "REGULAR_AND_EXTENDED_HOURS",
) -> pd.DataFrame:
    """Real historical OHLCV bars for `ticker` from Public.com, trimmed to
    [start, end] (either may be None for open-ended), in the same canonical
    shape `engine.data_utils.to_dataframe` produces for every other data
    source in this app.

    `interval` follows this app's existing convention ('1d'/'1wk'/'1mo'/'1h').
    Public.com only offers a fixed set of lookback windows (not an arbitrary
    start date) at the wire level, so this picks the smallest one that
    covers [start, end] and trims the extra afterwards.
    """
    from engine.data_utils import to_dataframe, trim_date_range

    aggregation = _INTERVAL_TO_AGGREGATION.get(interval)
    if aggregation is None:
        raise PublicApiError(f"Unsupported interval for Public.com bars: {interval!r}")
    period = _smallest_covering_period(start, end)

    resp = _request(
        "GET",
        f"/userapigateway/historicdata/{instrument_type}/{ticker.upper()}/{period}/{aggregation}",
        params={"tradingSessionToggle": trading_session},
    )
    if not resp.ok:
        raise PublicApiError(
            f"Public.com returned no bar data for '{ticker}' ({resp.status_code}): "
            f"{resp.text[:300]}"
        )
    payload = resp.json()
    bars = ((payload.get("regularMarket") or {}).get("bars")) or []
    if not bars:
        raise PublicApiError(f"Public.com returned no bars for '{ticker}'.")

    records = [
        {
            "date": b["timestamp"],
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": float(b.get("volume") or 0),
        }
        for b in bars
    ]
    df = to_dataframe(records)
    return trim_date_range(df, start=start, end=end)


# ---------------------------------------------------------------------------
# Live quotes + option chain -- feeds the standalone "Live Option Chain" tab
# ---------------------------------------------------------------------------


def fetch_quotes(symbols: List[str], instrument_type: str = "EQUITY") -> Dict[str, dict]:
    """Live quotes keyed by symbol. Returns only symbols Public.com resolved
    successfully; a bad/unknown symbol is silently omitted rather than
    failing the whole batch."""
    account_id = get_account_id()
    resp = _request(
        "POST",
        f"/userapigateway/marketdata/{account_id}/quotes",
        json={"instruments": [{"symbol": s.upper(), "type": instrument_type} for s in symbols]},
    )
    if not resp.ok:
        raise PublicApiError(f"Public.com quotes failed ({resp.status_code}): {resp.text[:300]}")
    out = {}
    for q in resp.json().get("quotes", []):
        if q.get("outcome") == "SUCCESS":
            sym = (q.get("instrument") or {}).get("symbol")
            if sym:
                out[sym] = q
    return out


def fetch_option_expirations(ticker: str) -> List[str]:
    account_id = get_account_id()
    resp = _request(
        "POST",
        f"/userapigateway/marketdata/{account_id}/option-expirations",
        json={"instrument": {"symbol": ticker.upper(), "type": "EQUITY"}},
    )
    if not resp.ok:
        raise PublicApiError(
            f"Couldn't fetch option expirations for '{ticker}' ({resp.status_code}): "
            f"{resp.text[:300]}"
        )
    return resp.json().get("expirations", [])


def _flatten_option_row(row: dict) -> dict:
    details = row.get("optionDetails") or {}
    greeks = details.get("greeks") or {}

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "symbol": (row.get("instrument") or {}).get("symbol"),
        "strike": _f(details.get("strikePrice")),
        "bid": _f(row.get("bid")),
        "ask": _f(row.get("ask")),
        "last": _f(row.get("last")),
        "volume": row.get("volume"),
        "open_interest": row.get("openInterest"),
        "delta": _f(greeks.get("delta")),
        "gamma": _f(greeks.get("gamma")),
        "theta": _f(greeks.get("theta")),
        "vega": _f(greeks.get("vega")),
        "implied_volatility": _f(greeks.get("impliedVolatility")),
    }


def fetch_option_chain(ticker: str, expiration: str) -> Dict[str, List[dict]]:
    """The live current option chain for `ticker` at `expiration`
    ('YYYY-MM-DD', from `fetch_option_expirations`), as
    {"calls": [...], "puts": [...]}, each row flattened to simple numeric
    fields (see `_flatten_option_row`), sorted by strike ascending."""
    account_id = get_account_id()
    resp = _request(
        "POST",
        f"/userapigateway/marketdata/{account_id}/option-chain",
        json={"instrument": {"symbol": ticker.upper(), "type": "EQUITY"}, "expirationDate": expiration},
    )
    if not resp.ok:
        raise PublicApiError(
            f"Couldn't fetch the option chain for '{ticker}' {expiration} "
            f"({resp.status_code}): {resp.text[:300]}"
        )
    payload = resp.json()
    calls = sorted((_flatten_option_row(r) for r in payload.get("calls", [])), key=lambda r: r["strike"] or 0)
    puts = sorted((_flatten_option_row(r) for r in payload.get("puts", [])), key=lambda r: r["strike"] or 0)
    return {"calls": calls, "puts": puts}
