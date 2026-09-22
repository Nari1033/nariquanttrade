"""Black-Scholes option pricing utilities.

Used to *model* option prices for backtesting since free, point-in-time
historical options chains / implied-volatility history aren't available.
Volatility is proxied by the underlying's own realized (historical)
volatility -- see `realized_volatility`. This will not match real
historical option market prices (no bid/ask spread, skew, or term
structure), but it lets a strategy's *rules* -- strike selection by delta,
DTE-based entry/exit, profit-target/stop-loss thresholds -- be backtested
consistently using only the OHLCV data this app already has.

No external dependencies (no scipy): the normal CDF uses math.erf, and the
inverse normal CDF uses Acklam's rational approximation (accurate to
~1e-9), since scipy isn't installed in every environment this runs in.
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd

OptionType = Literal["call", "put"]


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's algorithm coefficients for the inverse standard normal CDF.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function). Accurate to ~1e-9
    over the full (0, 1) range via Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p!r}")
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    )


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError(
            f"Invalid Black-Scholes inputs: S={S}, K={K}, T={T}, sigma={sigma} "
            "(S, K, sigma must be > 0 and T > 0)"
        )
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Theoretical Black-Scholes European option price. T is in years, r
    and sigma are decimals (0.045 = 4.5%, 0.20 = 20% annualized vol)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def black_scholes_delta(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes delta. Calls are in [0, 1]; puts are in [-1, 0]."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm_cdf(d1)
    if option_type == "put":
        return norm_cdf(d1) - 1.0
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def strike_for_delta(
    option_type: OptionType, S: float, target_delta: float, T: float, r: float, sigma: float
) -> float:
    """Solve for the strike K whose Black-Scholes delta equals
    `target_delta`, using the closed-form inverse of the delta formula
    (exact for Black-Scholes -- no iteration needed).

    `target_delta` is signed the way Black-Scholes reports it: positive for
    calls (e.g. 0.30 for a 30-delta call), negative for puts (e.g. -0.30
    for a 30-delta put). Pass the magnitude with the right sign for the
    option type, or use strike_for_put_delta_magnitude for the common
    "30-delta put" phrasing used in options trading.
    """
    if option_type == "call":
        if not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta for a call must be in (0, 1)")
        d1 = norm_ppf(target_delta)
    elif option_type == "put":
        if not -1.0 < target_delta < 0.0:
            raise ValueError("target_delta for a put must be in (-1, 0)")
        d1 = norm_ppf(target_delta + 1.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma * sigma) * T))
    return K


def strike_for_put_delta_magnitude(
    S: float, delta_magnitude: float, T: float, r: float, sigma: float
) -> float:
    """Convenience wrapper: strike of a put whose delta magnitude (a
    positive number, e.g. 0.30 for "30-delta put") matches `delta_magnitude`.
    """
    if not 0.0 < delta_magnitude < 1.0:
        raise ValueError("delta_magnitude must be in (0, 1)")
    return strike_for_delta("put", S, -delta_magnitude, T, r, sigma)


def realized_volatility(
    close: pd.Series, window: int = 20, trading_days_per_year: int = 252
) -> pd.Series:
    """Annualized realized (historical) volatility: rolling stdev of daily
    log returns, annualized by sqrt(trading_days_per_year). Used as the
    Black-Scholes sigma input in lieu of real implied volatility. NaN for
    the first `window` bars."""
    log_returns = (close / close.shift(1)).apply(math.log)
    return log_returns.rolling(window=window, min_periods=window).std() * math.sqrt(
        trading_days_per_year
    )
