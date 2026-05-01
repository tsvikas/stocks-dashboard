"""Stock log-returns dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance_cache as yfc

st.set_page_config(
    page_title="Stock Log-Returns",
    page_icon="\N{CHART WITH UPWARDS TREND}",
    layout="wide",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,400;1,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      html, body, [data-testid="stAppViewContainer"] {
        background-color: #f7f3ec;
      }
      [data-testid="stSidebar"] {
        background-color: #efe9dd;
        border-right: 1px solid #d8cfbd;
      }
      h1.editorial-title {
        font-family: 'Fraunces', Georgia, serif;
        font-style: italic;
        font-weight: 600;
        font-size: 2.6rem;
        color: #1f1b16;
        margin-bottom: 0;
      }
      p.editorial-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #6b5e49;
        margin-top: 0.2rem;
        letter-spacing: 0.04em;
      }
      code, .stMarkdown pre, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
      }
      hr.hairline {
        border: 0;
        border-top: 1px solid #d8cfbd;
        margin: 0.6rem 0 1rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Lookback periods ---------------------------------------------------- #
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

QUICK_TICKERS: list[tuple[str, list[tuple[str, str, bool]]]] = [
    (
        "Single names",
        [
            ("NVDA", "NVIDIA", True),
            ("AAPL", "Apple", False),
            ("MSFT", "Microsoft", False),
            ("GOOGL", "Alphabet", False),
            ("META", "Meta", False),
            ("AMZN", "Amazon", False),
            ("TSLA", "Tesla", False),
        ],
    ),
    (
        "Indices / broad",
        [
            ("SPY", "S&P 500", True),
            ("QQQ", "Nasdaq 100", False),
            ("URTH", "MSCI World", True),
            ("ACWI", "All-Country World", False),
            ("VTI", "US Total Market", False),
        ],
    ),
    (
        "Commodities",
        [
            ("GLD", "Gold", True),
            ("SLV", "Silver", False),
        ],
    ),
    (
        "Bonds",
        [
            ("TLT", "20Y Treasuries", False),
        ],
    ),
    (
        "Crypto",
        [
            ("BTC-USD", "Bitcoin", False),
            ("ETH-USD", "Ethereum", False),
        ],
    ),
]


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_close(ticker: str, period: str) -> pd.Series:
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


# Cache key intentionally excludes anchor/units: those toggles drive the cheap
# `_apply` step and must not invalidate the expensive concat+resample frame.
@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def _load_prices(
    tickers: tuple[str, ...],
    lookback_key: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    period, slice_years, resample_rule = LOOKBACKS[lookback_key]
    cutoff = (
        pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=slice_years)
        if slice_years is not None
        else None
    )

    columns: list[pd.Series] = []
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            s = _fetch_close(t, period)
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


def _apply(prices: pd.DataFrame, anchor: str, units: str) -> pd.DataFrame:
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


# ---- Sidebar ------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Lookback")
    lookback_key = st.radio(
        "Lookback",
        list(LOOKBACKS.keys()),
        index=3,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("### Quick tickers")
    selected_quick: list[str] = []
    for group_name, items in QUICK_TICKERS:
        with st.expander(group_name, expanded=True):
            for sym, label, default in items:
                star = " \N{BLACK STAR}" if default else ""
                checked = st.checkbox(
                    f"{sym}{star} — {label}",
                    value=default,
                    key=f"chk_{sym}",
                )
                if checked:
                    selected_quick.append(sym)

    st.markdown("### Custom tickers")
    custom_text = st.text_input(
        "Custom tickers",
        placeholder="e.g. AMD, ASML, ^GSPC",
        label_visibility="collapsed",
    )
    custom = parse_custom(custom_text)

    st.markdown("### Anchor")
    anchor = st.radio(
        "Anchor",
        ["Start", "End"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="Start: line begins at 0. End: line ends at 0.",
    )

    st.markdown("### Y-axis units")
    units = st.radio(
        "Units",
        ["ln", "dB", "factor"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="ln: natural log return. dB: 10·log10. factor: P_t / P_ref.",
    )


# ---- Main ---------------------------------------------------------------- #
st.markdown(
    "<h1 class='editorial-title'>Stock log-returns</h1>"
    f"<p class='editorial-sub'>{lookback_key} · anchor: {anchor.lower()} · units: {units}</p>"
    "<hr class='hairline'/>",
    unsafe_allow_html=True,
)

# Merge quick + custom, preserving order, dedup case-insensitively.
seen: set[str] = set()
tickers: list[str] = []
for t in [*selected_quick, *custom]:
    key = t.upper()
    if key not in seen:
        seen.add(key)
        tickers.append(key)

if not tickers:
    st.info("Pick at least one ticker from the sidebar.")
    st.stop()

prices, errors = _load_prices(tuple(tickers), lookback_key)
frame = _apply(prices, anchor, units)

if errors:
    st.warning(
        " · ".join(f"**{t}**: {msg}" for t, msg in errors.items()),
        icon="\N{WARNING SIGN}",
    )

if frame.empty:
    st.error("No data available for the selected tickers and window.")
    st.stop()

st.line_chart(frame, height=520)

with st.expander("Latest values"):
    last = frame.dropna(how="all").iloc[-1].sort_values(ascending=False)
    st.dataframe(
        last.rename("value").to_frame().style.format("{:.4f}"),
        use_container_width=True,
    )

st.caption(
    f"Data via yfinance-cache · {len(frame.columns)} series · "
    f"{frame.index.min():%Y-%m-%d} → {frame.index.max():%Y-%m-%d}"
)
