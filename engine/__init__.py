"""Core strategy scanner and backtesting engine."""

from .models import PriceBar
from .data_utils import to_dataframe, trim_date_range, resample_ohlcv
from .indicators import sma, add_sma_columns
from .scanner import (
    crossover_series,
    golden_cross_recent,
    death_cross_recent,
    price_cross_sma_recent,
)
from .backtester import (
    backtest_sma_crossover,
    backtest_price_sma_crossover,
    annualized_return_pct,
    BacktestResult,
    Trade,
)
from .options_pricing import (
    black_scholes_price,
    black_scholes_delta,
    strike_for_delta,
    strike_for_put_delta_magnitude,
    realized_volatility,
)
from .options_backtester import backtest_bull_put_spread
from .cash_secured_put import backtest_cash_secured_put
from .bollinger import (
    bollinger_bands,
    bollinger_signals,
    bollinger_oversold_recent,
    backtest_bollinger_mean_reversion,
)
from .analysis import analyze_underperformance
from .sweep import SweepPeriod, generate_param_combos, sweep_strategy, summarize_combos_across_periods

__all__ = [
    "PriceBar",
    "to_dataframe",
    "trim_date_range",
    "resample_ohlcv",
    "sma",
    "add_sma_columns",
    "crossover_series",
    "golden_cross_recent",
    "death_cross_recent",
    "price_cross_sma_recent",
    "backtest_sma_crossover",
    "backtest_price_sma_crossover",
    "annualized_return_pct",
    "BacktestResult",
    "Trade",
    "black_scholes_price",
    "black_scholes_delta",
    "strike_for_delta",
    "strike_for_put_delta_magnitude",
    "realized_volatility",
    "backtest_bull_put_spread",
    "backtest_cash_secured_put",
    "bollinger_bands",
    "bollinger_signals",
    "bollinger_oversold_recent",
    "backtest_bollinger_mean_reversion",
    "analyze_underperformance",
    "SweepPeriod",
    "generate_param_combos",
    "sweep_strategy",
    "summarize_combos_across_periods",
]
