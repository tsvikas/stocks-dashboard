"""End-to-end fetch test for every symbol in tickers.toml.

Run with `uv run pytest -m network -v`. Network is required; tests are
skipped automatically when Yahoo Finance is unreachable.
"""

from __future__ import annotations

import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yfinance as yf
import yfinance_cache as yfc
import yfinance_cache.yfc_cache_manager as yfcm

TICKERS_FILE = Path(__file__).resolve().parent.parent / "tickers.toml"


def _all_symbols() -> list[str]:
    data = tomllib.loads(TICKERS_FILE.read_text(encoding="utf-8"))
    return [t["symbol"] for g in data["group"] for t in g["tickers"]]


def _yahoo_reachable() -> bool:
    req = urllib.request.Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5d&interval=1d",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point yfinance-cache at a per-test directory so the user cache is untouched."""
    monkeypatch.setattr(yfcm, "GetCacheDirpath", lambda: str(tmp_path))


@pytest.fixture(scope="session", autouse=True)
def _require_network():
    if not _yahoo_reachable():
        pytest.skip(
            "Yahoo Finance unreachable from this environment", allow_module_level=True
        )


@pytest.mark.network
@pytest.mark.parametrize("ticker", _all_symbols())
def test_yfc_history(ticker: str) -> None:
    """yfinance-cache (used by app.py) must return non-empty Close prices."""
    df = yfc.Ticker(ticker).history(
        period="1mo", max_age="6h", adjust_splits=True, adjust_divs=True
    )
    assert df is not None, f"{ticker}: history() returned None"
    assert not df.empty, f"{ticker}: empty frame"
    assert "Close" in df.columns, f"{ticker}: no Close column ({list(df.columns)})"
    assert df["Close"].notna().any(), f"{ticker}: Close all-NaN"


@pytest.mark.network
@pytest.mark.parametrize("ticker", _all_symbols())
def test_plain_yfinance_history(ticker: str) -> None:
    """Sanity check against plain yfinance to distinguish yfc bugs from missing data."""
    df = yf.Ticker(ticker).history(period="1mo", auto_adjust=True)
    assert df is not None and not df.empty, f"{ticker}: yfinance returned no data"
    assert "Close" in df.columns
    assert df["Close"].notna().any(), f"{ticker}: Close all-NaN"
