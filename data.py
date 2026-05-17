"""Pure helpers used by the Streamlit dashboard.

Kept free of Streamlit imports so tests can call the same functions the
app does without spinning up a script context. ``app.py`` wraps the
fetch helpers with ``st.cache_data`` for runtime memoization.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import colorcet as cc
import numpy as np
import pandas as pd
import yfinance as yf
import yfinance_cache as yfc

TICKERS_FILE = Path(__file__).with_name("tickers.toml")

# Glasbey palette tuned for light backgrounds (excludes near-white hues):
# 256 maximally distinguishable colors, so each selected ticker keeps a
# visually distinct line on the cream-colored dashboard.
TICKER_PALETTE: tuple[str, ...] = tuple(cc.glasbey_light)


def assign_colors(
    tickers: Iterable[str],
    existing: dict[str, str],
    palette: Sequence[str] = TICKER_PALETTE,
) -> dict[str, str]:
    """Return a stable ticker→color mapping.

    Tickers already in ``existing`` keep their color; tickers no longer in
    the active set are dropped (freeing their palette slot); new tickers
    take the first palette color not currently in use. Cycles by position
    only after the entire palette is consumed.
    """
    active = list(tickers)
    keep = set(active)
    result = {t: c for t, c in existing.items() if t in keep}
    for t in active:
        if t in result:
            continue
        used = set(result.values())
        for c in palette:
            if c not in used:
                result[t] = c
                break
        else:
            result[t] = palette[len(result) % len(palette)]
    return result

# yfinance native periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
# For unsupported ranges we fetch "max" and slice client-side.
# Resample rule keeps the chart payload at ~250-1300 points per ticker so the
# Vega-Lite redraw on every widget toggle stays sub-second.
LOOKBACKS: dict[str, tuple[str, int | None, str | None]] = {
    "1Y": ("1y", None, None),
    "2Y": ("2y", None, None),
    "3Y": ("max", 3, None),
    "5Y": ("5y", None, None),
    "10Y": ("10y", None, "W-FRI"),
    "20Y": ("max", 20, "W-FRI"),
    "30Y": ("max", 30, "ME"),
    "50Y": ("max", 50, "ME"),
    "MAX": ("max", None, "ME"),
}


def load_quick_tickers(
    path: Path,
) -> list[tuple[str, bool, list[tuple[str, str, bool]]]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        (
            g["name"],
            g.get("expanded", False),
            [(t["symbol"], t["label"], t.get("default", False)) for t in g["tickers"]],
        )
        for g in data["group"]
    ]


QUICK_TICKERS = load_quick_tickers(TICKERS_FILE)


def fetch_close(ticker: str, period: str) -> pd.Series:
    try:
        df = yfc.Ticker(ticker).history(
            period=period,
            max_age="6h",
            adjust_splits=True,
            adjust_divs=True,
        )
    except Exception:  # noqa: BLE001
        # yfinance-cache requires Yahoo `info` metadata (exchange, timezone,
        # firstTradeDate) that crypto and several non-US indices don't expose,
        # so it raises an assortment of TypeError/AttributeError/ValueError/
        # Exception. Plain yfinance has no such requirement.
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"].dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.name = ticker
    return s


def load_prices(
    tickers: tuple[str, ...],
    lookback_key: str,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_close,
) -> tuple[pd.DataFrame, dict[str, str]]:
    period, slice_years, resample_rule = LOOKBACKS[lookback_key]
    cutoff = (
        pd.Timestamp.now("UTC").tz_localize(None) - pd.DateOffset(years=slice_years)
        if slice_years is not None
        else None
    )

    columns: list[pd.Series] = []
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            s = fetch_fn(t, period)
        except Exception as exc:  # noqa: BLE001 — surface any fetch failure inline
            errors[t] = type(exc).__name__
            continue
        if s.empty:
            errors[t] = "no data"
            continue
        if cutoff is not None:
            s = s[s.index >= cutoff]
            if s.empty:
                errors[t] = "no data in window"
                continue
        columns.append(s)

    if not columns:
        return pd.DataFrame(), errors

    prices = pd.concat(columns, axis=1).sort_index()
    if resample_rule is not None:
        prices = prices.resample(resample_rule).last()
    return prices.dropna(how="all"), errors


def transform(prices: pd.DataFrame, anchor: str, units: str) -> pd.DataFrame:
    if prices.empty:
        return prices
    ref = prices.bfill().iloc[0] if anchor == "Start" else prices.ffill().iloc[-1]
    ratio = prices.divide(ref)
    if units == "ln":
        out = np.log(ratio)
    elif units == "dB":
        out = 10.0 * np.log10(ratio)
    else:
        out = ratio
    return out.dropna(how="all")


def parse_custom(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in text.replace(";", ",").split(","):
        t = raw.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
