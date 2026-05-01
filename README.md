# Stock log-returns dashboard

A Streamlit app for comparing log-returns across stocks, indices, commodities, bonds, and crypto. Powered by [`yfinance-cache`](https://pypi.org/project/yfinance-cache/), so daily-bar fetches hit the network at most once every 6 hours.

## Run locally

```bash
uv run streamlit run app.py
```

That's it. `uv` resolves and installs everything from `pyproject.toml` / `uv.lock` on first run. No API keys required.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
1. On [share.streamlit.io](https://share.streamlit.io), point a new app at the repo, branch, and `app.py`.
1. Streamlit Cloud detects `pyproject.toml` + `uv.lock` and installs via uv automatically.

## Controls

- **Lookback** — `1Y · 2Y · 3Y · 5Y · 10Y · 20Y · 30Y · 50Y · MAX`. Ranges yfinance doesn't support natively (`3Y`, `20Y`, `30Y`, `50Y`) fetch `period="max"` and slice client-side, so the disk cache stays warm regardless of which lookback the user picks.
- **Quick tickers** — grouped checkboxes. Defaults: NVDA, SPY, URTH, GLD.
- **Custom tickers** — comma-separated, appended to whatever's checked.
- **Anchor** — `Start` (`y = ln(P_t / P_0)`) or `End` (`y = ln(P_t / P_T)`).
- **Y-axis units** — `ln`, `dB` (= `10·log10`), or `factor` (= `P_t / P_ref`).

## Cache layout

`yfinance-cache` keeps its persistent on-disk store at `~/.cache/py-yfinance-cache/`. Split- and dividend-adjustment is handled inside the library; the app does not re-implement either. Per-ticker post-fetch transforms (log-return matrix construction) are wrapped in `@st.cache_data` keyed on `(tickers, lookback, anchor, units)` so toggling sidebar controls re-renders instantly.
