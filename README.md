# QuantTrade

*(Strategy scanner, backtester, and parameter sweep -- formerly "SMA Crossover Scanner & Backtester".)*

A small Python app with two pieces:

1. **Engine** (`engine/`) — pure, dependency-light functions that calculate
   SMA-50/SMA-200, detect a "Golden Cross" (or a price/SMA crossover in
   general), and backtest the SMA-crossover strategy against simple buy &
   hold. Fully unit tested (19 tests, `tests/test_engine.py`).
2. **GUI** (`app/`) — a Streamlit app with a **Scanner** tab (find tickers
   that just triggered a crossover) and a **Backtest** tab (pick a ticker +
   timeframe, see strategy vs. buy & hold, trade log, equity curve).

Data can come from the free `yfinance` package (real tickers, needs
internet), from bundled **synthetic** sample data (works instantly,
offline, no API key — see the disclaimer below), or from a **pre-fetched
offline dataset** of REAL historical prices for a broad market universe
(S&P 500+400+600, filtered to a minimum market cap) — see "Offline
dataset" below.

---

## Quickstart

```bash
pip install -r requirements.txt

# run the engine's test suite (stdlib unittest, no pytest needed)
python -m unittest discover -s tests -v

# launch the GUI
streamlit run app/app.py
```

The GUI defaults to "Sample data (offline demo)" so it works immediately.
Switch the sidebar to "Live (yfinance)" to scan/backtest real tickers once
you have internet access, or to "Offline dataset" once you've built one
via the Build Dataset tab (see below) — real historical prices with
zero network calls at read time.

> **Note on this build:** the sandbox this was built in has no access to
> PyPI beyond a small preinstalled set (pandas/numpy were present;
> `streamlit`, `yfinance`, `matplotlib`'s heavier cousins like `plotly`,
> and even `pytest` could not be installed or reached over the network).
> Concretely, that means: the engine (`engine/`) and the sample-data
> pipeline are fully tested and verified end-to-end, including rendering
> real chart images from the backtest output. `app/app.py`'s Streamlit
> wiring itself could not be executed here — it's straightforward,
> standard Streamlit code built on top of the tested `engine`/`app.charts`/
> `app.data_provider` modules, but you should give it a first run on your
> own machine to confirm the UI behaves as expected.

---

## 1. The engine (`engine/`)

### Input format

Every engine function accepts a **price history** in any of these shapes:

```python
from engine import PriceBar

# (a) list of PriceBar
bars = [PriceBar(date=..., open=..., high=..., low=..., close=..., volume=...), ...]

# (b) list of dicts
bars = [{"date": "2024-01-02", "open": 190.1, "high": 192.0, "low": 189.5,
         "close": 191.2, "volume": 52_000_000}, ...]

# (c) a pandas DataFrame with those columns (date as a column or as the index)
```

### SMA-50 / SMA-200

```python
from engine import add_sma_columns

df = add_sma_columns(bars, windows=(50, 200))
# df["sma_50"], df["sma_200"] — NaN until each window has enough history
```

### Golden Cross scanner

```python
from engine import golden_cross_recent

golden_cross_recent(bars, fast_window=50, slow_window=200, lookback_days=3)
# -> True if SMA-50 crossed above SMA-200 on any of the last 3 trading days
```

`death_cross_recent` is the mirror image, and `price_cross_sma_recent`
generalizes this to "price crosses above/below its own SMA-N" (e.g. price
crossing SMA-50). `scan_universe({ticker: bars, ...}, strategy_fn=...)` runs
any of these across many tickers and returns the ones that matched —
this is what powers the Scanner tab.

### Backtester

```python
from engine import backtest_sma_crossover

result = backtest_sma_crossover(bars, fast_window=50, slow_window=200,
                                 initial_capital=10_000.0, ticker="TEST")

result.strategy_return_pct   # total % return of the SMA-crossover strategy
result.buy_hold_return_pct   # total % return of buy & hold, same period
result.total_trades          # number of completed trades
result.win_rate_pct          # % of trades that were profitable
result.trades                # list of Trade(entry_date, entry_price, exit_date, exit_price, return_pct, is_win, ...)
result.equity_curve          # daily portfolio value while running the strategy
result.buy_hold_curve        # daily portfolio value of a buy & hold, same capital
```

**Modeled assumptions** (kept simple and stated explicitly, no hidden
magic):

- Buys 100% of capital at the **close** on the day SMA-50 crosses above
  SMA-200; sells 100% at the **close** on the day it crosses back below.
  No leverage, no shorting, no fees/slippage.
- If a position is still open when the data ends, it's **marked to market**
  on the final bar (closed at the last close) so the return is fully
  realized and counted as a trade.
- Buy & hold return is computed over the **entire supplied period** (first
  close to last close), so it's an apples-to-apples comparison against the
  strategy over the same timeframe you selected.
- SMAs use `min_periods = window` (no partial-window averaging), matching
  how SMA-50/SMA-200 are normally defined for trading signals.

All of this is exercised by hand-verified unit tests in
`tests/test_engine.py` (small windows like fast=1/slow=2 so the expected
crossover days and P/L can be computed by hand and checked exactly).

---

## 2. The GUI (`app/app.py`)

- **Scanner tab** — pick "Golden Cross" or "Price crosses SMA-50", a
  lookback window, and a comma-separated ticker list; get back a table of
  which tickers triggered.
- **Backtest tab** — pick one ticker, a timeframe (6mo/1y/2y/5y/10y/max),
  SMA windows, and starting capital; get back the strategy return, buy &
  hold return, trade count, win rate, a price+SMA+signals chart, an equity
  curve (strategy vs. buy & hold), and a trade log.
- **Trending News tab** — live market news + sentiment from Alpha
  Vantage's `NEWS_SENTIMENT` API (`app/news_provider.py`), optionally
  filtered by ticker or topic: each headline shows its source, an overall
  sentiment score/label, the sentiment for whichever ticker it's most
  about, a summary, and a link to the source article. A Refresh button
  bypasses the cache for a live pull. Requires a free Alpha Vantage API
  key (25 requests/day on the free tier -- results are cached for 15
  minutes to stay well under that). Set it either in
  `.streamlit/secrets.toml` as `ALPHAVANTAGE_API_KEY = "..."` (gitignored,
  local only) or paste it into the sidebar field at runtime; for the
  deployed app, add the same key under Streamlit Cloud's app Settings ->
  Secrets (separate from both the repo and your local secrets file).

- **Build Dataset tab** (`app/admin_fetch.py`) -- fetches the REAL
  historical dataset used by the "Offline dataset" data source, server-
  side, in small user-triggered batches (Yahoo Finance's free API rate-
  limits aggressively, so this is deliberately not a single bulk call).
  Default universe is a broad-market stand-in for "Russell 3000" (S&P
  500+400+600, ~1,500 tickers, `data/tickers/broad_market.csv`) filtered
  to a minimum market cap (checked cheaply via yfinance's `fast_info`
  before spending time on a full history fetch); adjustable years of
  history and batch size. Click "Fetch next batch" repeatedly (or return
  later) to work through the universe. Because Streamlit Cloud's
  filesystem is ephemeral, anything fetched is lost on the next
  reboot/redeploy unless you download the "dataset so far" zip and commit
  `data/historical_prices/` into the repo yourself -- that's what makes it
  available to every future deploy, not just the running session.

### Sample data disclaimer

`data/sample_prices/*.csv` (ACME, GLOBEX, INITECH, WAYNE, CYBERDYNE,
UMBRELLA) is **synthetic data generated by
`scripts/generate_sample_data.py`** — geometric Brownian motion with
engineered drift regimes (a clean uptrend, a choppy whipsaw name, a
straight bull run, a bear market, a flat/no-trend name, and one with a
fresh Golden Cross in the final days). The tickers are fictional company
names on purpose, so nobody mistakes this for real historical prices of an
actual stock. It exists purely so the app works with zero setup; switch to
"Live (yfinance)" for real data.

---

## 3. Choosing a data API for the iOS app

You'll eventually want a plain REST/JSON API an iOS app can call directly
with `URLSession` (`yfinance`, used for the Python prototype above, is an
unofficial Python scraper library, not something you'd call from Swift).
Here's what free/affordable REST APIs currently offer (checked September
2026):

| Provider | Free tier | Historical data on free tier? | Paid entry tier |
|---|---|---|---|
| **Financial Modeling Prep (FMP)** | 250 calls/day | Yes — end-of-day historical prices, company profiles, 150+ endpoints | **Starter: $22/mo** (billed annually) — 300 calls/min, up to 5 years of historical data |
| **Alpha Vantage** | 25 calls/day | Yes, but the daily cap is very low for scanning multiple tickers | $49.99/mo — 75 req/min, no daily cap |
| **Twelve Data** | 800 calls/day (8/min) | Unclear from their current pricing page — it reads as real-time-oriented; confirm historical/time-series coverage in their docs before relying on it | Paid tiers unlock guaranteed historical coverage |

**Recommendation: Financial Modeling Prep's free tier**, moving to their
$22/mo Starter tier if you outgrow 250 calls/day or need more than
end-of-day granularity. It's the only one of the three that explicitly
includes historical price data on the free plan — which is exactly what
a backtester needs — with a REST/JSON API that's a straightforward fit for
`URLSession` on iOS. Reserve Alpha Vantage for light, occasional lookups
(its 25/day cap makes it impractical for scanning a watchlist), and treat
Twelve Data as worth a second look if you mainly want real-time quotes with
a higher daily call budget, once you've confirmed their current historical
data terms directly.

Sources: [FMP pricing plans](https://site.financialmodelingprep.com/pricing-plans) ·
[Alpha Vantage premium pricing](https://www.alphavantage.co/premium/) ·
[Twelve Data pricing](https://twelvedata.com/pricing)

---

## Project layout

```
engine/                  Pure strategy/backtest logic (no UI, no I/O deps)
  models.py               PriceBar dataclass, accepted input types
  data_utils.py            Normalizes any input shape into a canonical DataFrame
  indicators.py            SMA calculation
  scanner.py                Crossover detection, golden/death cross, multi-ticker scan
  backtester.py               The backtest engine (Trade, BacktestResult)
tests/test_engine.py     19 unit tests (stdlib unittest, hand-verified math)
app/
  app.py                  Streamlit GUI (Scanner + Backtest tabs)
  data_provider.py          Sample/offline-data loaders + yfinance fetcher
  admin_fetch.py             Build Dataset tab -- fetches the real offline dataset
  charts.py                  Matplotlib chart builders
data/sample_prices/       Bundled synthetic CSVs (see disclaimer above)
data/tickers/              Seed ticker universe(s) for the offline dataset fetch
data/historical_prices/    REAL pre-fetched OHLCV (empty until you build it -- see above)
scripts/generate_sample_data.py   Regenerates the synthetic sample data
requirements.txt
```
