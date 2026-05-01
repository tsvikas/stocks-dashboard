"""Stock log-returns dashboard."""

from __future__ import annotations

import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance_cache as yfc

# Upstream deprecation warnings from yfinance / yfinance_cache. They originate
# inside the libraries (Timestamp.utcnow, lowercase 'd' offset alias, old proxy
# and raise_errors kwargs) and can only be silenced here until those packages
# are updated. These filters must run AFTER `import yfinance*` because yfinance
# installs its own 'default' filter for DeprecationWarning at import time and
# warnings.filterwarnings inserts at the front of the list — last writer wins.
warnings.filterwarnings(
    "ignore",
    message=r"Timestamp\.utcnow is deprecated.*",
    module=r"yfinance_cache\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"'d' is deprecated.*",
    module=r"yfinance_cache\..*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Set proxy via new config control.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"'raise_errors' deprecated.*",
)

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

QUICK_TICKERS: list[tuple[str, bool, list[tuple[str, str, bool]]]] = [
    (
        "Single names",
        True,
        [
            ("NVDA", "NVIDIA", False),
            ("AAPL", "Apple", False),
            ("MSFT", "Microsoft", False),
            ("GOOGL", "Alphabet", False),
            ("META", "Meta", False),
            ("AMZN", "Amazon", False),
            ("TSLA", "Tesla", False),
        ],
    ),
    (
        "US indices (long history)",
        True,
        [
            ("^GSPC", "S&P 500 (1927)", True),
            ("^NYA", "NYSE Composite (1965)", False),
            ("^IXIC", "NASDAQ Composite (1971)", False),
            ("^RUT", "Russell 2000 (1987)", False),
            ("^DJI", "Dow Jones (1992)", False),
        ],
    ),
    (
        "US ETFs",
        False,
        [
            ("SPY", "S&P 500", False),
            ("QQQ", "Nasdaq 100", False),
            ("QLD", "Nasdaq 100 ×2", False),
            ("TQQQ", "Nasdaq 100 ×3", False),
            ("VTI", "US Total Market", False),
        ],
    ),
    (
        "World / global",
        True,
        [
            ("^990100-USD-STRD", "MSCI World index — developed", False),
            ("^892400-USD-STRD", "MSCI ACWI index — all-country", False),
            ("^991000-USD-STRD", "MSCI World ex USA index", False),
            ("AW01.FGI", "FTSE All-World index", False),
            ("^SPG1200", "S&P Global 1200 index", False),
            ("URTH", "MSCI World ETF — developed (2012)", True),
            ("ACWI", "MSCI ACWI ETF — all-country (2008)", True),
            ("VT", "Vanguard Total World ETF (2008)", False),
            ("EFA", "Developed ex-US ETF (2001)", False),
            ("EEM", "Emerging Markets ETF (2003)", False),
            ("VEU", "FTSE All-World ex-US ETF (2007)", False),
        ],
    ),
    (
        "Europe indices",
        False,
        [
            ("^STOXX50E", "Euro Stoxx 50", False),
            ("^STOXX", "STOXX Europe 600", False),
            ("^FTSE", "FTSE 100 — UK", False),
            ("^GDAXI", "DAX — Germany", False),
            ("^FCHI", "CAC 40 — France", False),
            ("^SSMI", "SMI — Switzerland", False),
            ("^AEX", "AEX — Netherlands", False),
            ("^IBEX", "IBEX 35 — Spain", False),
            ("FTSEMIB.MI", "FTSE MIB — Italy", False),
            ("^OMX", "OMX Stockholm 30 — Sweden", False),
        ],
    ),
    (
        "Asia indices",
        False,
        [
            ("^N225", "Nikkei 225 — Japan", False),
            ("^HSI", "Hang Seng — Hong Kong", False),
            ("000001.SS", "SSE Composite — Shanghai", False),
            ("399001.SZ", "SZSE Component — Shenzhen", False),
            ("^KS11", "KOSPI — South Korea", False),
            ("^TWII", "TAIEX — Taiwan", False),
            ("^BSESN", "BSE SENSEX — India", False),
            ("^NSEI", "NIFTY 50 — India", False),
            ("^STI", "Straits Times — Singapore", False),
        ],
    ),
    (
        "Pacific & Americas",
        False,
        [
            ("^AORD", "All Ordinaries — Australia", False),
            ("^AXJO", "S&P/ASX 200 — Australia", False),
            ("^NZ50", "NZX 50 — New Zealand", False),
            ("^GSPTSE", "S&P/TSX Composite — Canada", False),
            ("^BVSP", "Bovespa — Brazil", False),
            ("^MXX", "IPC — Mexico", False),
            ("^MERV", "MERVAL — Argentina", False),
        ],
    ),
    (
        "Middle East & Africa",
        False,
        [
            ("TA35.TA", "TA-35 — Israel", False),
            ("TA125.TA", "TA-125 — Israel", False),
            ("^TASI.SR", "Tadawul — Saudi Arabia", False),
            ("^JN0U.JO", "FTSE/JSE Top 40 — South Africa", False),
        ],
    ),
    (
        "Commodities",
        False,
        [
            ("^XAU", "PHLX Gold/Silver miners — equity (1983)", False),
            ("GC=F", "Gold futures (2000)", False),
            ("SI=F", "Silver futures (2000)", False),
            ("CL=F", "WTI crude futures (2000)", False),
            ("GLD", "Gold ETF (2004)", False),
            ("SLV", "Silver ETF (2006)", False),
            ("DBC", "Broad commodities ETF (2006)", False),
        ],
    ),
    (
        "Bonds",
        False,
        [
            ("TLT", "20Y Treasuries (2002)", False),
            ("IEF", "7-10Y Treasuries (2002)", False),
            ("SHY", "1-3Y Treasuries (2002)", False),
            ("AGG", "US Aggregate Bond (2003)", False),
            ("LQD", "Investment-grade corp (2002)", False),
            ("HYG", "High-yield corp (2007)", False),
        ],
    ),
    (
        "Crypto",
        False,
        [
            ("BTC-USD", "Bitcoin", False),
            ("ETH-USD", "Ethereum", False),
        ],
    ),
]


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def load_prices(
    tickers: tuple[str, ...],
    lookback_key: str,
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
            s = fetch_close(t, period)
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


# ---- Sidebar (tickers only) --------------------------------------------- #
with st.sidebar:
    st.markdown("### Custom tickers")
    custom_text = st.text_input(
        "Custom tickers",
        placeholder="e.g. AMD, ASML, ^GSPC",
        label_visibility="collapsed",
    )
    custom = parse_custom(custom_text)

    st.markdown("### Quick tickers")
    selected_quick: list[str] = []
    for group_name, expanded, items in QUICK_TICKERS:
        with st.expander(group_name, expanded=expanded):
            for sym, label, default in items:
                checked = st.checkbox(
                    f"{sym} — {label}",
                    value=default,
                    key=f"chk_{sym}",
                )
                if checked:
                    selected_quick.append(sym)


# ---- Main ---------------------------------------------------------------- #
st.markdown(
    "<h1 class='editorial-title'>Stock log-returns</h1><hr class='hairline'/>",
    unsafe_allow_html=True,
)

ctl_lookback, ctl_anchor, ctl_units = st.columns([1, 1, 1])
with ctl_lookback:
    lookback_key = st.selectbox(
        "Lookback",
        list(LOOKBACKS.keys()),
        index=list(LOOKBACKS).index("20Y"),
    )
with ctl_anchor:
    anchor = st.radio(
        "Anchor",
        ["Start", "End"],
        index=1,
        horizontal=True,
        help="Start: line begins at 0. End: line ends at 0.",
    )
with ctl_units:
    units = st.radio(
        "Y-axis units",
        ["ln", "dB", "ratio"],
        index=1,
        horizontal=True,
        help="ln: natural log return. dB: 10·log10. ratio: P_t / P_ref (log y-scale).",
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

prices, errors = load_prices(tuple(tickers), lookback_key)
frame = transform(prices, anchor, units)

if errors:
    st.warning(
        " · ".join(f"**{t}**: {msg}" for t, msg in errors.items()),
        icon="\N{WARNING SIGN}",
    )

if frame.empty:
    st.error("No data available for the selected tickers and window.")
    st.stop()

chart_df = frame.reset_index()
chart_df = (
    chart_df.rename(columns={chart_df.columns[0]: "Date"})
    .melt(id_vars="Date", var_name="Ticker", value_name="Value")
    .dropna()
)

y_scale = alt.Scale(type="log") if units == "ratio" else alt.Scale(type="linear")
chart = (
    alt.Chart(chart_df)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Value:Q", scale=y_scale, title=units),
        color=alt.Color("Ticker:N", legend=alt.Legend(title=None)),
    )
    .properties(height=520)
)
st.altair_chart(chart, width="stretch")

UNIT_HELP = {
    "ln": "-0.5 ≈ -40%<br>-0.1 ≈ -10%<br>+0.1 ≈ +10%<br>+0.5 ≈ +65%<br>+0.7 ≈ ×2<br>+1.0 ≈ ×2.7<br>+2.3 ≈ ×10",
    "dB": "-1 dB ≈ -20%<br>+1 dB ≈ +25%<br>+3 dB ≈ ×2<br>+10 dB = ×10",
    "ratio": "1.0 = unchanged<br>2.0 = ×2<br>0.5 = ÷2",
}
st.markdown(UNIT_HELP[units], unsafe_allow_html=True)

st.caption(
    f"Data via yfinance-cache · {len(frame.columns)} series · "
    f"{frame.index.min():%Y-%m-%d} → {frame.index.max():%Y-%m-%d}"
)
