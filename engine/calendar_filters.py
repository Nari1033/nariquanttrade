"""Calendar-day-of-month entry filter, shared by the options strategies.

Answers a concrete day-of-month question -- "is it better to open new
short-option positions near the start, middle, or end of the month?" --
as a plain, sweepable numeric knob (`entry_day_of_month`) rather than a
fixed rule baked into any one strategy, so it drops straight into the
existing Parameter Sweep page like every other NumberParam.

`entry_day_of_month=0` (the default everywhere it's wired in) disables
the filter entirely -- every existing backtest's behavior is byte-for-byte
unchanged unless this is explicitly set. Any other value is a target
calendar day 1-28 (not 29-31, so the target exists in every month
including February); a fixed +/-DEFAULT_WINDOW_DAYS tolerance around it
keeps enough eligible entry days per month for a backtest to actually
trade, rather than gating to one exact date that might land on a weekend
or a day this engine's calendar-day (not trading-day) DTE model has no
bar for.
"""

from __future__ import annotations

import pandas as pd

#: +/- this many calendar days around `entry_day_of_month` still count as
#: a match. 5 days keeps a real window of eligible entry days per month
#: (day-of-month strategies gated to one exact date would rarely align
#: with an actual price bar) while still meaningfully separating "early
#: month" (~day 1-6ish), "mid-month" (~day 10-20), and "month end"
#: (~day 23-28) as distinct, non-overlapping choices.
DEFAULT_WINDOW_DAYS = 5

#: The only valid non-zero range for `entry_day_of_month` -- 29/30/31
#: aren't valid targets since they don't exist in every month.
MIN_TARGET_DAY = 1
MAX_TARGET_DAY = 28


def day_of_month_ok(dt: pd.Timestamp, entry_day_of_month: int, window: int = DEFAULT_WINDOW_DAYS) -> bool:
    """True if `entry_day_of_month` is 0 (filter disabled -- always True)
    or `dt`'s calendar day-of-month falls within +/-`window` days of it.

    Raises ValueError for an out-of-range non-zero target (matches this
    app's convention of raising on bad backtest inputs, e.g. "need at
    least N price bars", rather than silently clamping)."""
    if entry_day_of_month == 0:
        return True
    if not (MIN_TARGET_DAY <= entry_day_of_month <= MAX_TARGET_DAY):
        raise ValueError(
            f"entry_day_of_month must be 0 (disabled) or in "
            f"[{MIN_TARGET_DAY}, {MAX_TARGET_DAY}], got {entry_day_of_month}"
        )
    return abs(dt.day - entry_day_of_month) <= window
