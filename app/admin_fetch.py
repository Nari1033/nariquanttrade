"""Build the offline (real, pre-fetched) historical dataset used by
data_provider's "offline" source.

Why this lives in the app, not in a standalone script: the only place in
this whole setup that can actually reach Yahoo Finance is wherever this
Streamlit app is deployed (Streamlit Cloud) -- see the module docstring in
app/data_provider.py. So the fetch has to run as a page *inside* the app
itself, server-side, triggered by a person clicking a button while it's
deployed, rather than as an offline script someone runs on their laptop.

Design constraints this works around:
  - Yahoo Finance's free/unofficial API rate-limits aggressively. Fetching
    a broad market universe (~1,500 tickers here) in one shot is likely to
    get throttled or take a very long time. So this fetches in small
    user-controlled batches (a button click = one batch), not "the whole
    universe in one call".
  - Streamlit Cloud's filesystem is EPHEMERAL -- anything written to
    data/historical_prices/ during a session is gone on the next reboot/
    redeploy unless it's committed to git. So this also offers a zip
    download of whatever's been fetched so far, for the person to unzip
    into their local clone and commit/push -- that's what makes it
    "offline" data for every future deploy, not just this running process.
  - A ticker's market cap is checked via yfinance's lightweight fast_info
    (no full .info call, which is much slower) and tickers under the
    threshold are skipped rather than fetched, so time isn't spent
    downloading years of history for something that'll just be filtered
    out.

Nothing here is network-free, so (like data_provider's yfinance path) this
isn't covered by the automated test suite for its actual network calls --
tests cover the pure logic (parsing the ticker list, deciding keep/skip,
writing csvs) with fetch functions mocked out.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from app.data_provider import OFFLINE_DATA_DIR
from engine.data_utils import to_dataframe

DEFAULT_TICKER_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "tickers" / "broad_market.csv"
DEFAULT_MIN_MARKET_CAP = 1_000_000_000.0
DEFAULT_YEARS = 10


def load_ticker_universe(path: Path = DEFAULT_TICKER_LIST_PATH) -> List[str]:
    """The seed ticker list (S&P 500 + 400 + 600, ~1,500 tickers -- a
    broad-market stand-in for "Russell 3000"; the real Russell 3000 list
    isn't freely redistributable) that admin fetch batches are drawn from
    by default. Returns [] if the file doesn't exist rather than raising,
    so a fresh checkout without it just shows an empty universe instead of
    crashing the Admin tab."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    return [str(t).strip().upper() for t in df["ticker"] if str(t).strip()]


def get_market_cap(ticker: str) -> Optional[float]:
    """Market cap in dollars via yfinance's fast_info (much cheaper than a
    full .info call). Returns None if yfinance isn't installed, the ticker
    is bad, or the field isn't available -- callers treat None as "unknown,
    don't skip on cap alone" rather than "definitely under threshold"."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        fast_info = yf.Ticker(ticker).fast_info
        cap = fast_info.get("marketCap") if hasattr(fast_info, "get") else fast_info["marketCap"]
        return float(cap) if cap else None
    except Exception:
        return None


def _download_history(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=False, group_by="column"
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for '{ticker}'")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.rename(columns=str.lower)
    raw.index.name = "date"
    return to_dataframe(raw[["open", "high", "low", "close", "volume"]])


def save_ticker_csv(df: pd.DataFrame, ticker: str, out_dir: Path = OFFLINE_DATA_DIR) -> Path:
    """Write `df` (canonical to_dataframe shape) to out_dir/<TICKER>.csv in
    the same date,open,high,low,close,volume format the bundled sample
    data uses, so both sources are read back identically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.upper()}.csv"
    out = df.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def fetch_one_ticker(
    ticker: str,
    years: int = DEFAULT_YEARS,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    out_dir: Path = OFFLINE_DATA_DIR,
    market_cap_fn: Callable[[str], Optional[float]] = get_market_cap,
    history_fn: Callable[[str, int], pd.DataFrame] = _download_history,
) -> dict:
    """Fetch and save one ticker, or explain why it was skipped/failed.
    `market_cap_fn`/`history_fn` are injectable so this is unit-testable
    without any real network access -- tests pass in fakes.

    Returns a dict with at least a "status" key, one of:
      "saved"            -- market cap ok (or unknown+min_market_cap<=0),
                             history fetched and written to csv.
      "skipped_small_cap" -- market cap known and below min_market_cap;
                             history was never fetched (saves time/quota).
      "failed"            -- market cap lookup raised, or history fetch/
                             write raised; "error" holds the message.
    """
    ticker = ticker.strip().upper()
    result = {"ticker": ticker, "status": None, "market_cap": None, "rows": None, "error": None}

    if min_market_cap > 0:
        try:
            cap = market_cap_fn(ticker)
        except Exception as exc:  # pragma: no cover - defensive, market_cap_fn already catches
            result["status"] = "failed"
            result["error"] = f"market cap lookup: {exc}"
            return result
        result["market_cap"] = cap
        if cap is not None and cap < min_market_cap:
            result["status"] = "skipped_small_cap"
            return result

    try:
        df = history_fn(ticker, years)
        save_ticker_csv(df, ticker, out_dir=out_dir)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return result

    result["status"] = "saved"
    result["rows"] = len(df)
    return result


def fetch_batch(
    tickers: List[str],
    years: int = DEFAULT_YEARS,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    out_dir: Path = OFFLINE_DATA_DIR,
    market_cap_fn: Callable[[str], Optional[float]] = get_market_cap,
    history_fn: Callable[[str, int], pd.DataFrame] = _download_history,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """Fetch a list of tickers one at a time (no concurrency -- deliberately
    gentle on Yahoo's rate limiter), calling `progress_cb(done, total)`
    after each one if given. Never raises for an individual ticker failure;
    see fetch_one_ticker's per-ticker "status"."""
    results = []
    for i, t in enumerate(tickers, start=1):
        results.append(
            fetch_one_ticker(
                t,
                years=years,
                min_market_cap=min_market_cap,
                out_dir=out_dir,
                market_cap_fn=market_cap_fn,
                history_fn=history_fn,
            )
        )
        if progress_cb:
            progress_cb(i, len(tickers))
    return results


def zip_offline_dataset(out_dir: Path = OFFLINE_DATA_DIR) -> bytes:
    """Zip every csv currently in out_dir into an in-memory archive, for a
    Streamlit download button -- this is how fetched data escapes Streamlit
    Cloud's ephemeral filesystem: download the zip, unzip into your local
    clone's data/historical_prices/, commit, push."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if out_dir.exists():
            for path in sorted(out_dir.glob("*.csv")):
                zf.write(path, arcname=path.name)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Streamlit panel
# ---------------------------------------------------------------------------


def render_admin_fetch_panel() -> None:
    """The 'Build Dataset' tab: fetch real historical data server-side (the
    only place in this whole setup with working internet access to Yahoo
    Finance -- see the module docstring) in small batches, and offer a zip
    download of whatever's been fetched so far."""
    import streamlit as st

    st.caption(
        "Builds the REAL historical dataset used by the 'Offline dataset' data "
        "source (see the sidebar) -- a broad-market universe (S&P 500 + 400 + "
        "600, ~1,500 tickers by default) filtered to a minimum market cap, "
        "with N years of daily OHLCV fetched per ticker via yfinance."
    )
    st.warning(
        "⚠️ Yahoo Finance's free API rate-limits aggressively, so this fetches "
        "in **small batches you trigger** rather than all at once -- click "
        "'Fetch next batch' repeatedly (or come back and click it again "
        "later). Also: this app's filesystem is **ephemeral** -- anything "
        "fetched here is lost on the next reboot/redeploy unless you "
        "download the zip below and commit `data/historical_prices/` into "
        "the repo yourself.",
        icon="⚠️",
    )

    universe = load_ticker_universe()
    if not universe:
        st.error(
            f"No ticker seed list found at `{DEFAULT_TICKER_LIST_PATH}`. "
            "Nothing to fetch from."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        min_cap_billions = st.number_input(
            "Minimum market cap ($B)",
            min_value=0.0,
            value=DEFAULT_MIN_MARKET_CAP / 1e9,
            step=0.5,
            help="Tickers below this market cap are skipped (their history is never fetched, to save time/quota). 0 disables the filter.",
        )
    with col2:
        years = st.number_input(
            "Years of history", min_value=1, max_value=25, value=DEFAULT_YEARS, step=1
        )
    with col3:
        batch_size = st.number_input(
            "Batch size (tickers per click)", min_value=1, max_value=200, value=50, step=10
        )

    st.caption(f"Universe: **{len(universe)} ticker(s)** from `{DEFAULT_TICKER_LIST_PATH.name}`.")

    st.session_state.setdefault("admin_fetch_idx", 0)
    st.session_state.setdefault("admin_fetch_results", [])

    idx = st.session_state["admin_fetch_idx"]
    remaining = len(universe) - idx
    st.progress(min(idx / len(universe), 1.0) if universe else 1.0)
    st.caption(f"{idx} / {len(universe)} ticker(s) attempted so far this session ({remaining} remaining).")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        run_batch = st.button("Fetch next batch ▶", type="primary", disabled=remaining <= 0)
    with col_b:
        if st.button("Reset progress (keeps already-fetched files)"):
            st.session_state["admin_fetch_idx"] = 0
            st.session_state["admin_fetch_results"] = []
            st.rerun()
    with col_c:
        pass

    if run_batch:
        batch = universe[idx : idx + int(batch_size)]
        progress_bar = st.progress(0.0)
        status = st.empty()

        def _cb(done: int, total: int) -> None:
            progress_bar.progress(done / total if total else 1.0)
            status.caption(f"Fetching {done} / {total} in this batch…")

        with st.spinner(f"Fetching {len(batch)} ticker(s)…"):
            results = fetch_batch(
                batch,
                years=int(years),
                min_market_cap=float(min_cap_billions) * 1e9,
                progress_cb=_cb,
            )
        st.session_state["admin_fetch_results"].extend(results)
        st.session_state["admin_fetch_idx"] = idx + len(batch)
        st.rerun()

    all_results = st.session_state["admin_fetch_results"]
    if all_results:
        n_saved = sum(1 for r in all_results if r["status"] == "saved")
        n_skipped = sum(1 for r in all_results if r["status"] == "skipped_small_cap")
        n_failed = sum(1 for r in all_results if r["status"] == "failed")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fetched this session", len(all_results))
        m2.metric("Saved", n_saved)
        m3.metric("Skipped (< min cap)", n_skipped)
        m4.metric("Failed", n_failed)

        with st.expander(f"Per-ticker results ({len(all_results)})", expanded=False):
            st.dataframe(pd.DataFrame(all_results), use_container_width=True, hide_index=True)

    offline_count = len(list_offline_tickers())
    st.markdown(f"**{offline_count} ticker(s)** currently on disk in `data/historical_prices/` (this run).")

    if offline_count:
        zip_bytes = zip_offline_dataset()
        st.download_button(
            "⬇️ Download dataset so far (zip)",
            data=zip_bytes,
            file_name="historical_prices.zip",
            mime="application/zip",
            help=(
                "Unzip into your local clone's data/historical_prices/, then "
                "git add/commit/push -- that's what makes this data available "
                "to every future deploy, not just this running session."
            ),
        )
