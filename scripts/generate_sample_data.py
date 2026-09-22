"""Generate bundled SYNTHETIC daily OHLCV data so the app works out of the
box with no API key and no network access.

This is NOT real market data. Tickers are fictional company names chosen
specifically so nobody mistakes this for real historical prices of an
actual stock. Each series is built from piecewise-drift geometric Brownian
motion so it exercises the scanner/backtester in a predictable way (clear
golden cross, a choppy whipsaw name, a straight bull run, a bear market,
and a fresh cross within the last few days).

Run with: python3 scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_prices"
END_DATE = pd.Timestamp("2026-09-18")
N_DAYS = 6 * 252  # ~6 trading years, enough for 1y/2y/5y backtests
DATES = pd.bdate_range(end=END_DATE, periods=N_DAYS)


def build_close_path(rng: np.random.Generator, start_price: float, segments) -> np.ndarray:
    """segments: list of (n_days, daily_drift, daily_vol). Concatenated and
    trimmed/padded to exactly N_DAYS."""
    chunks = [rng.normal(loc=drift, scale=vol, size=n) for n, drift, vol in segments]
    log_rets = np.concatenate(chunks)
    if len(log_rets) < N_DAYS:
        n, drift, vol = segments[-1]
        pad = N_DAYS - len(log_rets)
        log_rets = np.concatenate([log_rets, rng.normal(drift, vol, size=pad)])
    log_rets = log_rets[:N_DAYS]
    return start_price * np.exp(np.cumsum(log_rets))


def ohlcv_from_close(rng: np.random.Generator, closes: np.ndarray, base_volume: int):
    n = len(closes)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    prev_close = closes[0] * (1 - 0.001)
    for i, c in enumerate(closes):
        o = prev_close * (1 + rng.normal(0, 0.003))
        rng_size = abs(rng.normal(0, 0.01)) * c
        h = max(o, c) + rng_size * rng.uniform(0.2, 1.0)
        l = max(min(o, c) - rng_size * rng.uniform(0.2, 1.0), 0.01)
        opens[i] = round(o, 2)
        highs[i] = round(h, 2)
        lows[i] = round(l, 2)
        prev_close = c
    volumes = rng.integers(int(base_volume * 0.5), int(base_volume * 1.5), size=n)
    return opens, highs, lows, volumes


# (n_days, daily_drift, daily_vol) segments per ticker, roughly summing to N_DAYS.
TICKER_SEGMENTS = {
    # Long grind sideways, then a clean sustained uptrend -> a textbook,
    # profitable golden cross roughly 2 years before the end of the series.
    "ACME": [(760, 0.0000, 0.011), (760, 0.00085, 0.012), (232, 0.0004, 0.013)],
    # Choppy / mean-reverting the whole way -> whipsaws the SMA strategy
    # with several false starts.
    "GLOBEX": [(252, 0.0003, 0.017), (252, -0.0003, 0.017)] * 3,
    # Strong monotonic bull run almost from day one -> buy & hold should
    # beat the strategy since the 200-day SMA needs a long warm-up.
    "INITECH": [(1512, 0.00075, 0.013)],
    # Sustained bear market -> the strategy should sit in cash for most of
    # the decline and lose far less than buy & hold.
    "WAYNE": [(760, 0.0002, 0.011), (752, -0.00075, 0.016)],
    # Flat/no-trend chop with low volatility -> few or no signals at all.
    "CYBERDYNE": [(1512, 0.00002, 0.006)],
    # Long, gentle drift, then a short sharp rally engineered so SMA-50
    # crosses above SMA-200 within the last few trading days -> this is the
    # one the scanner should flag as a fresh Golden Cross.
    "UMBRELLA": [(1467, -0.00005, 0.010), (45, 0.0015, 0.012)],
}

START_PRICES = {
    "ACME": 40.0,
    "GLOBEX": 65.0,
    "INITECH": 22.0,
    "WAYNE": 120.0,
    "CYBERDYNE": 55.0,
    "UMBRELLA": 30.0,
}

BASE_VOLUME = {
    "ACME": 3_500_000,
    "GLOBEX": 1_800_000,
    "INITECH": 6_000_000,
    "WAYNE": 2_400_000,
    "CYBERDYNE": 900_000,
    "UMBRELLA": 4_200_000,
}


# Per-ticker RNG seed overrides. Most tickers just use a stable offset from
# their position in TICKER_SEGMENTS; UMBRELLA is pinned to a specific seed
# found by search so its engineered rally actually produces a Golden Cross
# in the final trading days (exact crossover timing is sensitive to the
# random draws, not just the drift/vol parameters).
SEED_OVERRIDES = {
    "UMBRELLA": 171,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (ticker, segments) in enumerate(TICKER_SEGMENTS.items()):
        seed = SEED_OVERRIDES.get(ticker, 1000 + i)
        rng = np.random.default_rng(seed=seed)
        closes = build_close_path(rng, START_PRICES[ticker], segments)
        opens, highs, lows, volumes = ohlcv_from_close(rng, closes, BASE_VOLUME[ticker])
        df = pd.DataFrame(
            {
                "date": DATES.strftime("%Y-%m-%d"),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": np.round(closes, 2),
                "volume": volumes,
            }
        )
        out_path = OUT_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(df)} rows, {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f})")


if __name__ == "__main__":
    main()
