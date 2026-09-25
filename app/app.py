"""QuantTrade -- Streamlit GUI: strategy scanner, backtester, and parameter sweep.

Run with:  streamlit run app/app.py   (from the project root)

Top-level layout: one "Scanner" tab (pick which strategy to scan with from
a dropdown -- it's one page, not duplicated per strategy) plus one
"Backtest" tab per strategy (see app/strategies.py -- add a new `Strategy`
entry there and it gets its own backtest tab automatically, no app.py
changes needed). Each panel uses that strategy's own scan_fn/backtest_fn
and parameters, so results always match whichever strategy is selected.

Data source is selectable in the sidebar:
  - "Sample data (offline demo)" - bundled SYNTHETIC csvs, works with zero
    setup. Clearly NOT real market data. Daily bars only on disk; weekly/
    monthly are resampled on the fly; hourly isn't available.
  - "Live (yfinance)" - real historical data via the free yfinance package.
    Requires `pip install yfinance` and internet access. Live fetches are
    cached to disk (see app/cache.py) so re-running a scan/backtest doesn't
    re-hit the API every time -- see the cache controls in the sidebar.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app import cache
from app.charts import plot_equity_curves, plot_price_with_signals
from app.data_provider import (
    fetch_yfinance_ticker,
    fetch_yfinance_universe,
    list_offline_tickers,
    list_sample_tickers,
    load_offline_ticker,
    load_offline_universe,
    load_sample_ticker,
    range_key as data_range_key,
)
from app.admin_fetch import render_admin_fetch_panel
from app.strategies import STRATEGIES, NumberParam, Strategy, default_grid_selection
from app.news_provider import (
    NEWS_TOPICS,
    NewsFetchError,
    SENTIMENT_ORDER,
    fetch_trending_news,
    matched_catalysts,
    sentiment_distribution,
    top_movers,
)
from engine.analysis import analyze_underperformance
from engine.backtester import annualized_return_pct
from engine.indicators import add_sma_columns
from engine.scanner import scan_universe
from engine.sweep import SweepPeriod, summarize_combos_across_periods, sweep_strategy

st.set_page_config(page_title="QuantTrade", layout="wide")

INTERVAL_CHOICES = {
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo",
    "Hourly (live only, recent history)": "1h",
}
# Rough calendar days per bar, used only to size the SMA warm-up buffer
# fetched before a custom backtest start date.
_BAR_TO_CALENDAR_DAYS = {"1d": 1.0, "1wk": 7.0, "1mo": 31.0, "1h": 1 / 24}

TODAY = dt.date.today()

# Fixed default windows for the Sweep page -- "test a strategy across
# several multi-year regimes, not just one lucky window." Anchored on
# Sept 20 (today, in the environment this was built in) so the most
# recent window is exactly the trailing two years; the user can uncheck
# any of these in the UI.
DEFAULT_SWEEP_PERIODS = [
    ("2020-09 → 2022-09", dt.date(2020, 9, 20), dt.date(2022, 9, 20)),
    ("2022-09 → 2024-09", dt.date(2022, 9, 20), dt.date(2024, 9, 20)),
    ("2024-09 → 2026-09", dt.date(2024, 9, 20), dt.date(2026, 9, 20)),
    ("2020-09 → 2026-09 (full)", dt.date(2020, 9, 20), dt.date(2026, 9, 20)),
]


# ---------------------------------------------------------------------------
# Sidebar: data source + cache controls
# ---------------------------------------------------------------------------

st.sidebar.title("Data source")
_offline_count = len(list_offline_tickers())
source_label = st.sidebar.radio(
    "Where should price data come from?",
    [
        "Sample data (offline demo)",
        f"Offline dataset ({_offline_count} tickers, cached)",
        "Live (yfinance)",
    ],
    index=0,
    help=(
        "Sample data is bundled synthetic price history so the app works "
        "immediately with no setup. Offline dataset is REAL historical data "
        "pre-fetched via the Build Dataset tab and committed to the repo -- "
        "still zero network at read time, but real tickers instead of "
        "fictional ones. Live fetches straight from yfinance every time "
        "(subject to its cache), for anything not in the offline dataset."
    ),
)
if source_label.startswith("Sample"):
    source = "sample"
elif source_label.startswith("Offline"):
    source = "offline"
else:
    source = "yfinance"

if source == "sample":
    st.sidebar.caption(
        "⚠️ Sample data is **synthetic** (fictional companies: ACME, GLOBEX, "
        "INITECH, WAYNE, CYBERDYNE, UMBRELLA) — not real market history. "
        "It's here so you can try the scanner and backtester with zero setup. "
        "Only daily bars exist on disk; Weekly/Monthly are resampled from "
        "them on the fly, and Hourly isn't available offline."
    )
    cache_max_age_hours = 12.0
    force_refresh = False
elif source == "offline":
    if _offline_count:
        st.sidebar.caption(
            f"✅ Real historical daily data for **{_offline_count} ticker(s)**, "
            "pre-fetched via the 🗄️ Build Dataset tab. Zero network at read "
            "time. Weekly/Monthly are resampled on the fly; Hourly isn't "
            "available (daily bars only were fetched)."
        )
    else:
        st.sidebar.warning(
            "No offline data yet. Go to the 🗄️ Build Dataset tab to fetch some, "
            "or switch source for now."
        )
    cache_max_age_hours = 12.0
    force_refresh = False
else:
    st.sidebar.caption(
        "Live mode uses the free `yfinance` package (no API key). Requires "
        "`pip install yfinance` and outbound internet access."
    )
    st.sidebar.subheader("Local cache")
    st.sidebar.caption(
        "Fetched data is saved to `data/cache/` and reused instead of "
        "calling the live API again, until it goes stale."
    )
    cache_max_age_hours = st.sidebar.number_input(
        "Treat cached data as fresh for (hours)", min_value=0.0, value=12.0, step=1.0
    )
    force_refresh = st.sidebar.checkbox(
        "Force refresh from live API (ignore cache)", value=False
    )
    with st.sidebar.expander("Cached tickers"):
        entries = cache.list_cache_entries()
        if entries:
            st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
            if st.button("🗑 Clear entire cache"):
                n = cache.clear_cache()
                st.success(f"Deleted {n} cached file(s). Re-run a scan/backtest to refetch.")
        else:
            st.caption("Nothing cached yet — run a scan or backtest to populate it.")

st.title("QuantTrade")
st.caption("Strategy scanner, backtester, and parameter sweep.")


# ---------------------------------------------------------------------------
# Welcome section -- a first-visit orientation for new users, collapsible
# so it doesn't get in the way once you know your way around. Expanded by
# default since a fresh Streamlit session is exactly when someone is
# seeing this for the first time.
# ---------------------------------------------------------------------------

with st.expander("👋 New here? Here's what QuantTrade does", expanded=True):
    st.markdown(
        "QuantTrade is a sandbox for testing trading strategies against historical "
        "data before risking real money on them. Screen tickers for a signal, "
        "backtest a strategy's return against simple buy & hold, and search across "
        "parameter combinations to see what actually holds up over time -- all from "
        "one place, using either free real market data or an offline sample dataset."
    )
    st.markdown("")

    thumb_cols = st.columns(5)
    thumbnails = [
        (
            "🔍",
            "Scanner",
            "Screen a list of tickers for a strategy signal -- a golden cross, a "
            "price/SMA break, or a bullish options setup -- over any date range "
            "and bar interval.",
        ),
        (
            "📈",
            "Multi-Strategy Backtest",
            "Run Golden Cross, Price-crosses-SMA, or Bull Put Spread against real "
            "or sample history. See return vs. buy & hold (total and annualized), "
            "win rate, equity curve, and a full trade log.",
        ),
        (
            "🎯",
            "Options Strategies",
            "Backtest a Bull Put Spread priced with Black-Scholes -- tune the "
            "delta, spread width, profit target, and stop loss, simulated day by "
            "day against the underlying.",
        ),
        (
            "🧪",
            "Parameter Sweep",
            "Grid-search a strategy's parameters across four market windows "
            "(2020-22, 2022-24, 2024-26, and the full span) to find combos that "
            "beat buy & hold consistently, not just once.",
        ),
        (
            "📰",
            "Trending News",
            "Live market news and sentiment via Alpha Vantage -- filter by ticker "
            "or topic, see each headline's sentiment score, and jump straight to "
            "the source.",
        ),
    ]
    for col, (icon, title, desc) in zip(thumb_cols, thumbnails):
        with col:
            with st.container(border=True):
                st.markdown(f"<div style='font-size:2.2rem'>{icon}</div>", unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(desc)

    st.markdown("")
    st.info(
        "**Getting started:** the sidebar defaults to bundled sample data, so "
        "everything works instantly with zero setup. Switch to **Live (yfinance)** "
        "there once you want to scan or backtest real tickers.",
        icon="💡",
    )


# ---------------------------------------------------------------------------
# Shared panel renderers -- called once per strategy, per sub-tab
# ---------------------------------------------------------------------------


def _render_param_widget(p: NumberParam, key: str):
    """Render one NumberParam as an st.number_input, using int or float
    widget semantics per its `is_int` flag. A strategy like Bull Put Spread
    has params (short_delta, spread_width_pct, ...) that are meaningless if
    blanket-cast to int, so this -- and every call site below -- always
    routes through p.is_int rather than assuming every param is a whole
    number of bars."""
    if p.is_int:
        return st.number_input(
            p.label,
            min_value=int(p.min_value),
            max_value=int(p.max_value),
            value=int(p.default),
            step=int(p.step),
            help=p.help,
            key=key,
        )
    return st.number_input(
        p.label,
        min_value=float(p.min_value),
        max_value=float(p.max_value),
        value=float(p.default),
        step=float(p.step),
        help=p.help,
        key=key,
    )


def _render_param_row(params: List[NumberParam], key_prefix: str, extra_cols: int = 0):
    """Render `params` as number_input widgets, wrapped into rows of at
    most 4 so strategies with many params (e.g. Bull Put Spread's 6) don't
    squeeze every widget into one unreadably narrow row. Returns
    (param_values dict, list of the trailing `extra_cols` st.columns
    objects from the last row, for a caller to put extra controls like
    "lookback days" or "initial capital" into)."""
    n = len(params)
    total_slots = n + extra_cols
    per_row = 4
    param_values: dict = {}
    extra_slots: list = []
    i = 0
    slot = 0
    while slot < total_slots:
        row_slots = min(per_row, total_slots - slot)
        cols = st.columns(row_slots)
        for c in cols:
            if i < n:
                p = params[i]
                with c:
                    param_values[p.key] = _render_param_widget(p, key=f"{key_prefix}_{p.key}")
                i += 1
            else:
                extra_slots.append(c)
            slot += 1
    return param_values, extra_slots


def _typed_params(strategy: Strategy, param_values: dict) -> dict:
    """Cast each param value per its own NumberParam.is_int, instead of
    blanket int(...)-ing every param -- required now that Bull Put Spread
    has float params (short_delta, spread_width_pct, profit_target_pct,
    stop_loss_pct) alongside the int ones."""
    by_key = {p.key: p for p in strategy.params}
    return {
        k: (int(v) if by_key[k].is_int else float(v))
        for k, v in param_values.items()
    }


def render_scanner_panel(strategy: Strategy, key_prefix: str) -> None:
    st.caption(strategy.description)

    param_values, extra = _render_param_row(strategy.params, key_prefix=f"{key_prefix}_scan", extra_cols=1)
    with extra[0]:
        lookback_days = st.number_input(
            "Signal within the last N bars",
            min_value=1,
            max_value=20,
            value=3,
            key=f"{key_prefix}_scan_lookback",
        )

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        scan_start = st.date_input(
            "Data start date", value=TODAY - dt.timedelta(days=730), key=f"{key_prefix}_scan_start"
        )
    with col_e:
        scan_end = st.date_input("Data end date", value=TODAY, key=f"{key_prefix}_scan_end")
    with col_f:
        scan_interval_label = st.selectbox(
            "Interval", list(INTERVAL_CHOICES.keys()), index=0, key=f"{key_prefix}_scan_interval"
        )
        scan_interval = INTERVAL_CHOICES[scan_interval_label]

    if source == "sample":
        default_tickers = ", ".join(list_sample_tickers())
        ticker_input = st.text_input(
            "Tickers to scan (comma-separated)", value=default_tickers, key=f"{key_prefix}_scan_tickers"
        )
    elif source == "offline":
        offline_tickers = list_offline_tickers()
        default_tickers = ", ".join(offline_tickers[:10])
        ticker_input = st.text_input(
            "Tickers to scan (comma-separated)",
            value=default_tickers,
            key=f"{key_prefix}_scan_tickers",
            help=f"{len(offline_tickers)} ticker(s) available offline -- showing the first 10 as a starting point.",
        )
    else:
        ticker_input = st.text_input(
            "Tickers to scan (comma-separated)",
            value="AAPL, MSFT, NVDA, GOOGL, AMZN",
            key=f"{key_prefix}_scan_tickers",
        )

    run_scan = st.button("Run scan", type="primary", key=f"{key_prefix}_scan_run")

    if run_scan:
        if scan_start >= scan_end:
            st.error("Data start date must be before the data end date.")
            return

        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        with st.spinner(f"Loading price history for {len(tickers)} ticker(s)..."):
            if source == "sample":
                universe, missing = {}, []
                for t in tickers:
                    try:
                        universe[t] = load_sample_ticker(
                            t, start=scan_start, end=scan_end, interval=scan_interval
                        )
                    except ValueError:
                        missing.append(t)
            elif source == "offline":
                universe = load_offline_universe(
                    tickers, start=scan_start, end=scan_end, interval=scan_interval
                )
                missing = [t for t in tickers if t not in universe]
            else:
                universe = fetch_yfinance_universe(
                    tickers,
                    start=scan_start,
                    end=scan_end,
                    interval=scan_interval,
                    max_age_hours=cache_max_age_hours,
                    force_refresh=force_refresh,
                )
                missing = [t for t in tickers if t not in universe]

        if missing:
            st.warning(f"Could not load data for: {', '.join(missing)}")

        if not universe:
            st.error("No price data available for the requested tickers.")
            return

        matches = scan_universe(
            universe,
            strategy_fn=strategy.scan_fn,
            lookback_days=int(lookback_days),
            **_typed_params(strategy, param_values),
        )

        scan_key = data_range_key(None, scan_start, scan_end)
        rows = []
        for ticker, df in universe.items():
            last_close = df["close"].iloc[-1] if len(df) else None
            last_date = df.index[-1].date() if len(df) else None
            cache_age = (
                cache.cache_age_hours(ticker, scan_key, scan_interval)
                if source == "yfinance"
                else None
            )
            rows.append(
                {
                    "Ticker": ticker,
                    "Signal triggered": "✅" if ticker in matches else "",
                    "Last close": round(float(last_close), 2) if last_close is not None else None,
                    "As of": str(last_date) if last_date else "",
                    "Bars available": len(df),
                    "Cache age (h)": round(cache_age, 1) if cache_age is not None else "",
                }
            )
        result_df = pd.DataFrame(rows).sort_values("Signal triggered", ascending=False).reset_index(
            drop=True
        )
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        if matches:
            st.success(f"{len(matches)} ticker(s) triggered: {', '.join(matches)}")
        else:
            st.info("No tickers triggered the signal in the requested window.")


def render_backtest_panel(strategy: Strategy, key_prefix: str) -> None:
    st.caption(strategy.description)

    row1c1, row1c2, row1c3, row1c4 = st.columns(4)
    with row1c1:
        if source == "sample":
            bt_ticker = st.selectbox("Ticker", list_sample_tickers(), key=f"{key_prefix}_bt_ticker")
        elif source == "offline":
            offline_tickers = list_offline_tickers()
            if offline_tickers:
                bt_ticker = st.selectbox("Ticker", offline_tickers, key=f"{key_prefix}_bt_ticker")
            else:
                st.warning("No offline data yet -- see the 🗄️ Build Dataset tab.")
                bt_ticker = None
        else:
            bt_ticker = st.text_input("Ticker", value="AAPL", key=f"{key_prefix}_bt_ticker").strip().upper()
    with row1c2:
        bt_start = st.date_input(
            "Start date", value=TODAY - dt.timedelta(days=5 * 365), key=f"{key_prefix}_bt_start"
        )
    with row1c3:
        bt_end = st.date_input("End date", value=TODAY, key=f"{key_prefix}_bt_end")
    with row1c4:
        bt_interval_label = st.selectbox(
            "Interval (tick size)", list(INTERVAL_CHOICES.keys()), index=0, key=f"{key_prefix}_bt_interval"
        )
        bt_interval = INTERVAL_CHOICES[bt_interval_label]

    param_values, extra = _render_param_row(strategy.params, key_prefix=f"{key_prefix}_bt", extra_cols=1)
    with extra[0]:
        initial_capital = st.number_input(
            "Initial capital ($)",
            min_value=100.0,
            value=10_000.0,
            step=1000.0,
            key=f"{key_prefix}_bt_capital",
        )

    # Largest *SMA-window* param drives how much warm-up history to fetch
    # before bt_start (see below) -- using is_sma_window rather than "every
    # param" matters now that a strategy like Bull Put Spread also has
    # non-window params (delta, spread width, profit/stop %) whose values
    # are nowhere near a sensible bar count.
    sma_window_params = [p for p in strategy.params if p.is_sma_window]
    max_window = max([p.default for p in sma_window_params] or [0])

    run_backtest = st.button("Run backtest", type="primary", key=f"{key_prefix}_bt_run")

    if not run_backtest:
        return

    if bt_start >= bt_end:
        st.error("Start date must be before end date.")
        return
    if bt_ticker is None:
        return
    if source in ("sample", "offline") and bt_interval == "1h":
        st.error(f"{source_label} is daily-only. Switch to Live (yfinance) for hourly bars.")
        return

    # Fetch extra history before bt_start so the slowest SMA in this
    # strategy is valid from the very first day of the window you asked
    # for, instead of ramping up from NaN. Everything shown below is still
    # trimmed/rebased back to exactly [bt_start, bt_end].
    buffer_days = max(int(max_window) * _BAR_TO_CALENDAR_DAYS.get(bt_interval, 1.0) * 1.3, 5)
    fetch_start = bt_start - dt.timedelta(days=int(buffer_days) + 5)

    try:
        with st.spinner(f"Loading {bt_ticker} price history..."):
            if source == "sample":
                df = load_sample_ticker(bt_ticker, start=fetch_start, end=bt_end, interval=bt_interval)
            elif source == "offline":
                df = load_offline_ticker(bt_ticker, start=fetch_start, end=bt_end, interval=bt_interval)
            else:
                df = fetch_yfinance_ticker(
                    bt_ticker,
                    start=fetch_start,
                    end=bt_end,
                    interval=bt_interval,
                    max_age_hours=cache_max_age_hours,
                    force_refresh=force_refresh,
                )
    except Exception as exc:
        st.error(f"Couldn't load data for '{bt_ticker}': {exc}")
        return

    typed_params = _typed_params(strategy, param_values)
    try:
        result = strategy.backtest_fn(
            df, initial_capital=float(initial_capital), ticker=bt_ticker, **typed_params
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    start_ts, end_ts = pd.Timestamp(bt_start), pd.Timestamp(bt_end)
    equity_visible = result.equity_curve.loc[start_ts:end_ts]
    buy_hold_visible = result.buy_hold_curve.loc[start_ts:end_ts]

    if len(equity_visible) < 2:
        st.error(
            "Not enough bars in the selected date range/interval to backtest. "
            "Try a wider date range or a finer interval."
        )
        return

    actual_data_start = df.index[0].date()
    if actual_data_start < bt_start:
        st.caption(
            f"ℹ️ Loaded data from **{actual_data_start}** — extra history before your "
            f"selected start so the strategy's SMA(s) are valid from day one. All figures "
            f"and charts below are for your selected **{bt_start} → {bt_end}** window."
        )

    # Returns rebased to the start of the visible window so strategy vs.
    # buy & hold are compared over exactly [bt_start, bt_end], even though
    # the underlying curves started earlier (for SMA warm-up).
    strategy_return_pct = (
        (equity_visible.iloc[-1] - equity_visible.iloc[0]) / equity_visible.iloc[0] * 100
    )
    buy_hold_return_pct = (
        (buy_hold_visible.iloc[-1] - buy_hold_visible.iloc[0]) / buy_hold_visible.iloc[0] * 100
    )
    visible_trades = [t for t in result.trades if t.exit_date >= start_ts and t.entry_date <= end_ts]
    total_trades = len(visible_trades)
    wins = sum(1 for t in visible_trades if t.is_win)
    win_rate_pct = (wins / total_trades * 100) if total_trades else 0.0

    window_days = (end_ts - start_ts).days
    strategy_annualized_pct = annualized_return_pct(strategy_return_pct, window_days)
    buy_hold_annualized_pct = annualized_return_pct(buy_hold_return_pct, window_days)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy return", f"{strategy_return_pct:+.1f}%")
    m2.metric(
        "Buy & Hold return",
        f"{buy_hold_return_pct:+.1f}%",
        delta=f"{strategy_return_pct - buy_hold_return_pct:+.1f} pp vs strategy",
        delta_color="off",
    )
    m3.metric("Total trades", total_trades)
    m4.metric("Win rate", f"{win_rate_pct:.0f}%")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric(
        "Strategy return (annualized)",
        f"{strategy_annualized_pct:+.1f}%" if strategy_annualized_pct is not None else "n/a",
        help="Compound annual growth rate (CAGR): what the strategy's return over your "
        "selected window would compound to over a full year. Short windows or a total "
        "return of -100% or worse make this undefined (shown as n/a).",
    )
    a2.metric(
        "Buy & Hold return (annualized)",
        f"{buy_hold_annualized_pct:+.1f}%" if buy_hold_annualized_pct is not None else "n/a",
        delta=(
            f"{strategy_annualized_pct - buy_hold_annualized_pct:+.1f} pp vs strategy"
            if strategy_annualized_pct is not None and buy_hold_annualized_pct is not None
            else None
        ),
        delta_color="off",
    )
    a3.metric("Window length", f"{window_days} days")
    a4.empty()

    sma_windows = tuple(sorted({int(v) for k, v in typed_params.items() if k in {p.key for p in sma_window_params}}))
    df_sma_visible = add_sma_columns(df, sma_windows).loc[start_ts:end_ts]

    upper_band = lower_band = None
    if strategy.band_fn is not None:
        # Bands are computed over the full fetched (warm-up-buffered) `df`
        # so they're valid from day one of the visible window, then
        # trimmed to match -- same convention as the SMA columns above.
        full_upper, full_lower = strategy.band_fn(df, typed_params)
        upper_band = full_upper.loc[start_ts:end_ts]
        lower_band = full_lower.loc[start_ts:end_ts]

    fig_price = plot_price_with_signals(
        df_sma_visible,
        result.fast_window if result.fast_window is not None else -1,
        result.slow_window if result.slow_window is not None else -1,
        trades=visible_trades,
        title=(
            f"{bt_ticker}: {strategy.label} ({bt_start} → {bt_end}, "
            f"{bt_interval_label.split(' (')[0]})"
        ),
        upper_band=upper_band,
        lower_band=lower_band,
    )
    st.pyplot(fig_price, use_container_width=True)

    # Rebase both curves to the same starting capital at bt_start for a
    # fair side-by-side chart.
    equity_rebased = equity_visible / equity_visible.iloc[0] * initial_capital
    buy_hold_rebased = buy_hold_visible / buy_hold_visible.iloc[0] * initial_capital
    fig_equity = plot_equity_curves(equity_rebased, buy_hold_rebased)
    st.pyplot(fig_equity, use_container_width=True)

    if visible_trades:
        # "Has strikes" (options-specific) is a narrower check than "has
        # any meta at all" -- Bollinger trades populate meta too (just
        # exit_reason, no strikes), so using `any(t.meta ...)` here would
        # mislabel Bollinger's stock entry/exit prices as "Credit/debit".
        has_strikes = any(t.meta.get("short_strike") is not None for t in visible_trades)
        has_exit_reason = any(t.meta.get("exit_reason") for t in visible_trades)
        # Generic, not wheel-specific -- any strategy whose trades carry a
        # meta["leg"] (currently just the Wheel, cycling between put and
        # call legs) gets this column; everything else doesn't.
        has_leg = any(t.meta.get("leg") for t in visible_trades)
        price_label = "Credit/debit" if has_strikes else "Price"
        trade_rows = [
            {
                "Entry date": t.entry_date.date(),
                f"Entry {price_label.lower()}": round(t.entry_price, 2),
                "Exit date": t.exit_date.date(),
                f"Exit {price_label.lower()}": round(t.exit_price, 2),
                "Return": f"{t.return_pct:+.1f}%",
                "Result": "Win" if t.is_win else "Loss",
                "Leg": t.meta.get("leg", "").capitalize(),
                "Exit reason": t.meta.get("exit_reason", ""),
                "Short strike": t.meta.get("short_strike", ""),
                "Long strike": t.meta.get("long_strike", ""),
                "Entered before window": "Yes" if t.entry_date < start_ts else "",
                "Open at period end": "Yes" if t.closed_at_period_end else "",
            }
            for t in visible_trades
        ]
        trade_df = pd.DataFrame(trade_rows)
        if not has_leg:
            trade_df = trade_df.drop(columns=["Leg"])
        if not has_exit_reason:
            trade_df = trade_df.drop(columns=["Exit reason"])
        if not has_strikes:
            # No strikes to show for non-options strategies -- drop the
            # options-only strike columns rather than showing a table full
            # of blank cells.
            trade_df = trade_df.drop(columns=["Short strike", "Long strike"])
        st.markdown("**Trade log**")
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

        # Generic for every strategy -- no strategy-specific logic here,
        # analyze_underperformance works from just the trades + the two
        # equity curves + the underlying price history.
        underperf_records = analyze_underperformance(visible_trades, equity_visible, buy_hold_visible, df)
        if underperf_records:
            with st.expander(
                f"🔍 Why did the strategy fall behind buy & hold? "
                f"({len(underperf_records)} of {total_trades} trade exit(s))"
            ):
                st.caption(
                    "At each trade exit below, the strategy's cumulative return (since "
                    f"{bt_start}) was lower than buy & hold's cumulative return over the same "
                    "stretch. Each entry explains what's driving that gap at that point in time -- "
                    "not just this trade, but the trade itself and/or the time spent out of the "
                    "market before it."
                )
                for rec in underperf_records:
                    st.markdown(
                        f"**{rec['exit_date'].date()}** — strategy {rec['strategy_cumulative_pct']:+.1f}% "
                        f"vs buy & hold {rec['buy_hold_cumulative_pct']:+.1f}% cumulative "
                        f"({rec['gap_pct']:+.1f} pp behind)"
                    )
                    st.caption(rec["explanation"])
        else:
            st.success(
                "The strategy was never behind buy & hold's cumulative return at any trade exit "
                "in this window."
            )
    else:
        st.info("No completed trades in this window — the signal never triggered.")


def render_sweep_panel() -> None:
    st.caption(
        "Try many parameter combinations for one strategy against one ticker, across "
        "several date windows, and see which combos beat plain buy & hold -- and how "
        "consistently they do it across periods, not just in one lucky window."
    )

    sweepable = [s for s in STRATEGIES if s.params]
    if not sweepable:
        st.info("No strategies with tunable parameters to sweep.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        default_idx = next(
            (i for i, s in enumerate(sweepable) if s.id == "bull_put_spread"), 0
        )
        strategy_label = st.selectbox(
            "Strategy", [s.label for s in sweepable], index=default_idx, key="sweep_strategy_choice"
        )
        strategy = next(s for s in sweepable if s.label == strategy_label)
    with col2:
        if source == "sample":
            sweep_ticker = st.selectbox("Ticker", list_sample_tickers(), key="sweep_ticker")
        elif source == "offline":
            offline_tickers = list_offline_tickers()
            if offline_tickers:
                sweep_ticker = st.selectbox("Ticker", offline_tickers, key="sweep_ticker")
            else:
                st.warning("No offline data yet -- see the 🗄️ Build Dataset tab.")
                sweep_ticker = None
        else:
            sweep_ticker = st.text_input("Ticker", value="AAPL", key="sweep_ticker").strip().upper()
    with col3:
        initial_capital = st.number_input(
            "Initial capital ($)", min_value=100.0, value=10_000.0, step=1000.0, key="sweep_capital"
        )

    st.markdown("**Date windows to test**")
    st.caption(
        "A combo only counts as 'beats buy & hold' if its return is higher over that "
        "exact window -- same rebasing convention as the Backtest tabs."
    )
    period_cols = st.columns(len(DEFAULT_SWEEP_PERIODS))
    chosen_periods: List[SweepPeriod] = []
    for (label, p_start, p_end), col in zip(DEFAULT_SWEEP_PERIODS, period_cols):
        with col:
            include = st.checkbox(label, value=True, key=f"sweep_period_{label}")
        if include:
            chosen_periods.append(SweepPeriod(label, p_start, p_end))

    st.markdown("**Parameter grid** — pick which values of each parameter to try")
    param_grid: Dict[str, list] = {}
    params = strategy.params
    for row_start in range(0, len(params), 3):
        row_params = params[row_start : row_start + 3]
        cols = st.columns(len(row_params))
        for p, c in zip(row_params, cols):
            with c:
                candidates, defaults = default_grid_selection(p, pool_n=6, default_n=4)
                chosen = st.multiselect(
                    p.label,
                    options=candidates,
                    default=defaults,
                    key=f"sweep_grid_{strategy.id}_{p.key}",
                    help=p.help,
                )
                param_grid[p.key] = chosen if chosen else [p.default]

    full_grid_size = 1
    for vals in param_grid.values():
        full_grid_size *= max(len(vals), 1)
    n_periods = max(len(chosen_periods), 1)

    max_combos = st.number_input(
        "Max parameter combinations to try (randomly sampled if the full grid is bigger)",
        min_value=1,
        max_value=1000,
        value=min(100, full_grid_size),
        step=10,
        key="sweep_max_combos",
        help=(
            f"Full grid from your selections above: {full_grid_size} combo(s). "
            "Sweeps run one backtest per (combo x period), so keep this modest -- "
            "a few hundred total backtests take well under a minute; a few thousand can be slow."
        ),
    )
    st.caption(
        f"Will run up to {min(max_combos, full_grid_size)} combo(s) × {n_periods} period(s) "
        f"= up to {min(max_combos, full_grid_size) * n_periods} backtests."
    )

    run_sweep = st.button("Run sweep", type="primary", key="sweep_run")

    if not run_sweep:
        return

    if not chosen_periods:
        st.error("Select at least one date window to test.")
        return

    sma_window_params = [p for p in params if p.is_sma_window]

    def buffer_days_fn(combo_params: dict) -> int:
        local_windows = [combo_params[p.key] for p in sma_window_params if p.key in combo_params]
        w = max(local_windows) if local_windows else 0
        return max(int(w * 1.3), 5) + 10

    max_sma_in_grid = max(
        [max(param_grid[p.key]) for p in sma_window_params if param_grid.get(p.key)] or [0]
    )
    fetch_buffer_days = max(int(max_sma_in_grid * 1.3), 5) + 10
    fetch_start = min(p.start for p in chosen_periods) - dt.timedelta(days=fetch_buffer_days)
    fetch_end = max(p.end for p in chosen_periods)

    if sweep_ticker is None:
        return

    try:
        with st.spinner(f"Loading {sweep_ticker} price history..."):
            if source == "sample":
                df = load_sample_ticker(sweep_ticker, start=fetch_start, end=fetch_end, interval="1d")
            elif source == "offline":
                df = load_offline_ticker(sweep_ticker, start=fetch_start, end=fetch_end, interval="1d")
            else:
                df = fetch_yfinance_ticker(
                    sweep_ticker,
                    start=fetch_start,
                    end=fetch_end,
                    interval="1d",
                    max_age_hours=cache_max_age_hours,
                    force_refresh=force_refresh,
                )
    except Exception as exc:
        st.error(f"Couldn't load data for '{sweep_ticker}': {exc}")
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def progress_cb(done: int, total: int) -> None:
        progress_bar.progress(done / total if total else 1.0)
        status.caption(f"Running backtest {done} / {total}…")

    with st.spinner("Running sweep…"):
        sweep_df = sweep_strategy(
            df,
            strategy.backtest_fn,
            param_grid,
            chosen_periods,
            buffer_days_fn,
            initial_capital=float(initial_capital),
            ticker=sweep_ticker,
            max_combos=int(max_combos),
            progress_cb=progress_cb,
        )
    progress_bar.empty()
    status.empty()

    if sweep_df.empty:
        st.warning(
            "No combo produced a valid backtest in the requested windows -- try a wider "
            "date range, a smaller trend-SMA window, or fewer/shorter periods."
        )
        return

    param_keys = [p.key for p in params]
    summary = summarize_combos_across_periods(sweep_df, param_keys)
    robust = summary[summary["beats_buy_hold_every_period"]]

    st.subheader("Results")
    if len(robust):
        st.success(
            f"{len(robust)} of {len(summary)} tested combo(s) beat buy & hold in "
            f"**all {len(chosen_periods)}** selected period(s) for {sweep_ticker}."
        )
    else:
        st.warning(
            f"None of the {len(summary)} tested combo(s) beat buy & hold in every one of the "
            f"{len(chosen_periods)} selected periods for {sweep_ticker}. This is a structural "
            f"tendency for a credit spread like Bull Put Spread: its max profit is capped at "
            f"the premium collected, so it can't keep pace with buy-and-hold through a strong, "
            f"sustained rally -- but it can still beat a flat or declining buy-and-hold. The "
            f"table below is ranked by how close each combo got (most periods beaten, then "
            f"average edge)."
        )

    st.markdown("**Combos ranked by robustness across periods, then average edge vs. buy & hold**")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if len(summary):
        top = summary.iloc[0]
        st.markdown("**Per-period detail for the top-ranked combo**")
        mask = pd.Series(True, index=sweep_df.index)
        for key in param_keys:
            mask &= sweep_df[key] == top[key]
        detail_cols = param_keys + [
            "period", "period_start", "period_end", "strategy_return_pct",
            "buy_hold_return_pct", "edge_pct", "strategy_annualized_pct",
            "total_trades", "win_rate_pct",
        ]
        st.dataframe(sweep_df.loc[mask, detail_cols], use_container_width=True, hide_index=True)

    csv_bytes = sweep_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full sweep results (CSV)",
        data=csv_bytes,
        file_name=f"{sweep_ticker}_{strategy.id}_sweep.csv",
        mime="text/csv",
        key="sweep_download",
    )



# ---------------------------------------------------------------------------
# Trending News panel -- live market news + sentiment via Alpha Vantage's
# NEWS_SENTIMENT API (app/news_provider.py does the fetch/parse; this just
# renders it). Free-tier Alpha Vantage keys are capped at 25 requests/day,
# so the actual HTTP call is wrapped in st.cache_data with a 15-minute TTL
# and only runs when a button below is pressed -- never on every rerun.
# ---------------------------------------------------------------------------

_TOPIC_LABEL_TO_SLUG = {slug.replace("_", " ").title(): slug for slug in NEWS_TOPICS}

_SENTIMENT_DOT = {
    "Bearish": "\U0001F534",
    "Somewhat-Bearish": "\U0001F7E0",
    "Neutral": "\u26AA",
    "Somewhat-Bullish": "\U0001F7E2",
    "Bullish": "\U0001F7E2",
}


@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_news(api_key: str, tickers: str, topics: str, sort: str, limit: int):
    """Thin cache boundary around news_provider.fetch_trending_news -- kept
    as a one-line wrapper (rather than decorating fetch_trending_news
    itself) so that module stays Streamlit-free and independently
    unit-testable (see tests/test_news_provider.py). Returns (items,
    fetched_at) rather than just items so the UI can show when this batch
    was actually pulled -- fetched_at is computed here, inside the cached
    function, so a cache HIT (same filters, within 15 min) keeps showing
    the original fetch time instead of the current time."""
    items = fetch_trending_news(
        api_key, tickers=tickers or None, topics=topics or None, sort=sort, limit=limit
    )
    return items, dt.datetime.now()


def render_news_panel() -> None:
    st.caption(
        "Latest market news from Alpha Vantage's `NEWS_SENTIMENT` API, optionally "
        "filtered by ticker or topic. Each headline is scored -1 (most negative) to "
        "+1 (most positive): \u2264-0.35 Bearish, -0.35 to -0.15 Somewhat-Bearish, "
        "-0.15 to 0.15 Neutral, 0.15 to 0.35 Somewhat-Bullish, \u22650.35 Bullish."
    )

    secrets_key = ""
    try:
        secrets_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        secrets_key = ""

    api_key = st.sidebar.text_input(
        "Alpha Vantage API key",
        value=secrets_key,
        type="password",
        help=(
            "Free key at alphavantage.co/support/#api-key. Free-tier keys are capped "
            "at 25 requests/day, so news results here are cached for 15 minutes."
        ),
        key="news_api_key",
    )
    if not secrets_key:
        st.sidebar.caption(
            "\U0001F4A1 To avoid re-entering this every time: add it to "
            "`.streamlit/secrets.toml` locally, or under this app's Settings -> "
            "Secrets on Streamlit Cloud, as `ALPHAVANTAGE_API_KEY = \"...\"`."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tickers_input = st.text_input(
            "Tickers (optional, comma-separated)",
            value="",
            placeholder="e.g. AAPL,TSLA",
            key="news_tickers",
        )
    with col2:
        topic_label = st.selectbox(
            "Topic (optional)",
            ["All"] + list(_TOPIC_LABEL_TO_SLUG.keys()),
            key="news_topic",
        )
        topic = _TOPIC_LABEL_TO_SLUG.get(topic_label, "")
    with col3:
        sort_label = st.selectbox("Sort", ["Latest", "Relevance"], key="news_sort")
        sort = "LATEST" if sort_label == "Latest" else "RELEVANCE"
    with col4:
        limit = st.number_input(
            "Max headlines", min_value=5, max_value=200, value=20, step=5, key="news_limit"
        )

    fetch_col, refresh_col = st.columns([1, 1])
    with fetch_col:
        fetch = st.button("Fetch trending news", type="primary", key="news_fetch")
    with refresh_col:
        refresh = st.button(
            "\U0001F504 Refresh",
            key="news_refresh",
            help=(
                "Skip the 15-minute cache and pull the latest headlines from Alpha "
                "Vantage right now, for whatever filters are set above."
            ),
        )
    if not (fetch or refresh):
        return

    if not api_key:
        st.error(
            "Enter an Alpha Vantage API key above (or set ALPHAVANTAGE_API_KEY in "
            "secrets) to fetch news."
        )
        return

    if refresh:
        # Drop every cached entry for this function (not just the current
        # filter combo) so Refresh always means "go live now", regardless
        # of which filters were used on the last fetch.
        _cached_fetch_news.clear()

    with st.spinner("Fetching news..."):
        try:
            items, fetched_at = _cached_fetch_news(
                api_key, tickers_input.strip().upper(), topic, sort, int(limit)
            )
        except NewsFetchError as exc:
            st.error(f"Couldn't fetch news: {exc.message}")
            return

    if not items:
        st.info("No headlines returned for these filters.")
        return

    # Pin up to 5 "potential movers" (FDA approvals, M&A, guidance cuts,
    # etc. -- see news_provider.impact_score) at the top, with near-dupe
    # coverage of the same story collapsed to one copy; everything else
    # follows, sorted most-bullish to most-bearish by overall_sentiment_score
    # (independent of the "Sort" control above, which only affects which
    # headlines Alpha Vantage returns in the first place).
    movers = top_movers(items, n=5)
    mover_ids = {id(m) for m in movers}
    rest = sorted(
        (i for i in items if id(i) not in mover_ids),
        key=lambda i: i.overall_sentiment_score,
        reverse=True,
    )
    items = movers + rest

    st.caption(
        f"{len(items)} headline(s) \u00b7 top {len(movers)} potential mover(s) pinned "
        f"first \u00b7 rest sorted bullish \u2192 bearish \u00b7 as of "
        f"{fetched_at.strftime('%H:%M:%S')} (auto-refreshes after 15 min, or press "
        "Refresh for a live pull now)"
    )

    counts = sentiment_distribution(items)
    dist_df = pd.DataFrame(
        {"Sentiment": SENTIMENT_ORDER, "Headlines": [counts[s] for s in SENTIMENT_ORDER]}
    ).set_index("Sentiment")
    st.bar_chart(dist_df)

    for item in items:
        primary = item.primary_ticker
        with st.container(border=True):
            if id(item) in mover_ids:
                hits = matched_catalysts(item)
                why = f" -- matched: {', '.join(hits)}" if hits else ""
                st.markdown(f"\U0001F680 **Potential mover**{why}")
            st.markdown(f"**[{item.title}]({item.url})**")
            meta_cols = st.columns([2, 2, 3, 3])
            meta_cols[0].caption(f"\U0001F553 {item.time_published.strftime('%b %d, %H:%M')} ET")
            meta_cols[1].caption(f"\U0001F4F0 {item.source}")
            dot = _SENTIMENT_DOT.get(item.overall_sentiment_label, "\u26AA")
            meta_cols[2].caption(
                f"{dot} {item.overall_sentiment_label} ({item.overall_sentiment_score:+.2f})"
            )
            if primary:
                p_dot = _SENTIMENT_DOT.get(primary.sentiment_label, "\u26AA")
                meta_cols[3].caption(
                    f"\U0001F3AF {primary.ticker}: {p_dot} {primary.sentiment_label} "
                    f"({primary.sentiment_score:+.2f})"
                )
            if item.summary:
                st.caption(item.summary)
            other_tickers = [t for t in item.tickers if primary is None or t.ticker != primary.ticker]
            if other_tickers:
                other = ", ".join(f"{t.ticker} ({t.sentiment_score:+.2f})" for t in other_tickers)
                st.caption(f"Also mentions: {other}")

    st.markdown("---")
    st.markdown("**Useful links**")
    st.markdown(
        "- [Alpha Vantage NEWS_SENTIMENT documentation]"
        "(https://www.alphavantage.co/documentation/#news-sentiment)\n"
        "- [Alpha Vantage full API documentation](https://www.alphavantage.co/documentation/)\n"
        "- [Get a free Alpha Vantage API key](https://www.alphavantage.co/support/#api-key)"
    )


# ---------------------------------------------------------------------------
# One top-level tab layout:
#   - "🔍 Scanner" -- a single page. Pick which strategy to scan with from
#     a dropdown, then it renders that strategy's own scan params/results
#     (each strategy still keeps its own remembered widget values, via
#     key_prefix=strategy.id, so switching back and forth doesn't reset
#     what you typed).
#   - One "📈 <strategy label>" tab per strategy for its backtest, so every
#     strategy's backtest results live on their own page instead of being
#     nested under a strategy-specific outer tab.
#   - "🧪 Parameter Sweep" and "📰 Trending News" always sit last, in that
#     order, after every per-strategy tab.
# Add a new Strategy to app/strategies.py and it shows up in both places
# automatically -- no other app.py changes needed.
# ---------------------------------------------------------------------------

main_tab_labels = (
    ["🔍 Scanner"]
    + [f"📈 {s.label}" for s in STRATEGIES]
    + ["🧪 Parameter Sweep", "📰 Trending News", "🗄️ Build Dataset"]
)
main_tabs = st.tabs(main_tab_labels)

with main_tabs[0]:
    strategy_by_label = {s.label: s for s in STRATEGIES}
    chosen_label = st.selectbox(
        "Strategy to scan with",
        list(strategy_by_label.keys()),
        key="scanner_strategy_choice",
        help="Each strategy has its own scan logic and parameters below.",
    )
    scanner_strategy = strategy_by_label[chosen_label]
    render_scanner_panel(scanner_strategy, key_prefix=scanner_strategy.id)

for strategy, backtest_tab in zip(STRATEGIES, main_tabs[1:-3]):
    with backtest_tab:
        render_backtest_panel(strategy, key_prefix=strategy.id)

with main_tabs[-3]:
    render_sweep_panel()

with main_tabs[-2]:
    render_news_panel()

with main_tabs[-1]:
    render_admin_fetch_panel()
