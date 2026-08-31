"""End-to-end fetch test for every symbol in tickers.toml.

Calls the same ``fetch_close`` the Streamlit app calls, so a regression
in the cache wrapper or the upstream Yahoo response shows up here.

Run with ``uv run pytest -v``. Requires network; auto-skips when Yahoo
Finance is unreachable.
"""

from __future__ import annotations

import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yfinance_cache.yfc_cache_manager as yfcm

from data import fetch_close

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


# When Yahoo withdraws a symbol's history it keeps answering with a single
# live quote, which is neither empty nor all-NaN — so a non-empty check alone
# lets a dead symbol through. Healthy symbols return ~20 trading days over a
# 1mo window and withdrawn ones return exactly 1, leaving room for a threshold
# well clear of both.
MIN_POINTS = 5


@pytest.mark.network
@pytest.mark.parametrize("ticker", _all_symbols())
def test_fetch_close(ticker: str) -> None:
    s = fetch_close(ticker, "1mo")
    assert len(s) >= MIN_POINTS, (
        f"{ticker}: {len(s)} point(s) over 1mo, expected ~20; "
        "Yahoo has likely withdrawn history for this symbol"
    )
    assert s.notna().any(), f"{ticker}: all-NaN"
