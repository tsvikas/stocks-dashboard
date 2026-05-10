"""Stock log-returns dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    LOOKBACKS,
    QUICK_TICKERS,
    fetch_close,
    load_prices,
    parse_custom,
    transform,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_close(ticker: str, period: str) -> pd.Series:
    return fetch_close(ticker, period)


@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def cached_load_prices(
    tickers: tuple[str, ...],
    lookback_key: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    return load_prices(tickers, lookback_key, fetch_fn=cached_fetch_close)


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

ctl_lookback, ctl_anchor, ctl_units, ctl_baseline = st.columns([1, 1, 1, 1])
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
with ctl_baseline:
    baseline_text = st.text_input(
        "Baseline",
        placeholder="e.g. SPY",
        help="If set to a ticker, all other tickers are shown relative to it.",
    )
baseline = baseline_text.strip().upper() if baseline_text else ""

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

fetch_tickers = list(tickers)
if baseline and baseline not in fetch_tickers:
    fetch_tickers.append(baseline)

prices, errors = cached_load_prices(tuple(fetch_tickers), lookback_key)
frame = transform(prices, anchor, units)

baseline_active = bool(baseline) and baseline in frame.columns
if baseline_active:
    base_series = frame[baseline].ffill()
    if units == "ratio":
        frame = frame.div(base_series, axis=0)
    else:
        frame = frame.sub(base_series, axis=0)
    frame = frame.drop(columns=[baseline])
elif baseline:
    st.warning(
        f"Baseline **{baseline}** unavailable; showing absolute view.",
        icon="\N{WARNING SIGN}",
    )

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
y_title = f"{units} vs {baseline}" if baseline_active else units
chart = (
    alt.Chart(chart_df)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Value:Q", scale=y_scale, title=y_title),
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
