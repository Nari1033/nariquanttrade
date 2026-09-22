"""Data models for OHLCV price history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from typing import Any, Iterable, Union


@dataclass(frozen=True)
class PriceBar:
    """A single day's OHLCV bar."""

    date: Date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_dict(d: dict) -> "PriceBar":
        required = ("date", "open", "high", "low", "close")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Price bar dict is missing required keys: {missing}")
        raw_date = d["date"]
        if isinstance(raw_date, str):
            parsed_date = datetime.fromisoformat(raw_date[:10]).date()
        elif isinstance(raw_date, datetime):
            parsed_date = raw_date.date()
        else:
            parsed_date = raw_date
        return PriceBar(
            date=parsed_date,
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d.get("volume", 0.0)),
        )


PriceHistory = Union[Iterable[PriceBar], Iterable[dict], Any]
"""A price history can be a list of PriceBar, a list of dicts with the keys
date/open/high/low/close/volume, or a pandas DataFrame with those columns
(a DatetimeIndex is also accepted in place of a 'date' column)."""
