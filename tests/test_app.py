"""Unit tests for pure helpers in app.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app import parse_custom, transform


def _prices(data: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="D")
    return pd.DataFrame(data, index=idx)


def test_parse_custom_dedup_case_and_separators() -> None:
    # Comma + semicolon separators, whitespace, mixed case, dedup preserves order.
    assert parse_custom(" aapl, MSFT ; aapl ;  nvda,MSFT") == ["AAPL", "MSFT", "NVDA"]
    assert parse_custom("") == []
    assert parse_custom("   ,;  ,") == []


def test_transform_start_anchor_first_row_is_zero() -> None:
    frame = transform(_prices({"A": [10.0, 11.0, 12.0], "B": [50.0, 55.0, 60.0]}), "Start", "ln")
    # ln(P_t / P_0) → first row is exactly 0 for every column.
    assert frame.iloc[0].abs().max() == 0.0


def test_transform_end_anchor_last_row_is_zero() -> None:
    frame = transform(_prices({"A": [10.0, 11.0, 12.0], "B": [50.0, 55.0, 60.0]}), "End", "ln")
    assert frame.iloc[-1].abs().max() == 0.0


def test_transform_db_is_ln_scaled_by_10_over_ln10() -> None:
    prices = _prices({"A": [10.0, 11.0, 9.5, 12.3]})
    ln_frame = transform(prices, "Start", "ln")
    db_frame = transform(prices, "Start", "dB")
    scale = 10.0 / math.log(10.0)
    np.testing.assert_allclose(db_frame.to_numpy(), ln_frame.to_numpy() * scale)


def test_transform_factor_units_returns_ratio() -> None:
    frame = transform(_prices({"A": [10.0, 12.0, 15.0]}), "Start", "factor")
    np.testing.assert_allclose(frame["A"].to_numpy(), [1.0, 1.2, 1.5])


def test_transform_missing_column_does_not_break_others() -> None:
    # A column that is entirely NaN (e.g. ticker fetched but had no data in window)
    # must not poison sibling columns — they should still render normally.
    prices = _prices({"A": [10.0, 11.0, 12.0], "MISSING": [float("nan")] * 3})
    frame = transform(prices, "Start", "ln")
    assert frame["MISSING"].isna().all()
    np.testing.assert_allclose(
        frame["A"].to_numpy(),
        np.log(np.array([10.0, 11.0, 12.0]) / 10.0),
    )
