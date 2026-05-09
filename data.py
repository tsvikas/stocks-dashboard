"""Price-fetch helpers.

Kept free of Streamlit imports so tests can call the same functions the
app does without spinning up a script context. ``app.py`` wraps these
with ``st.cache_data`` for runtime memoization.
"""

from __future__ import annotations

import pandas as pd
import yfinance_cache as yfc


def fetch_close(ticker: str, period: str) -> pd.Series:
    dat = yfc.Ticker(ticker)
    df = dat.history(
        period=period,
        max_age="6h",
        adjust_splits=True,
        adjust_divs=True,
    )
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"].dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.name = ticker
    return s
