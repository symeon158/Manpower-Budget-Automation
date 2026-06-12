"""Tests for date helpers: yf_scalar, yf_vec, eomonth."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mb.calc.date_helpers import eomonth, roundup, yf_scalar, yf_vec


class TestEomonth:
    def test_end_of_current_month(self):
        assert eomonth(pd.Timestamp(2025, 5, 15), 0) == pd.Timestamp(2025, 5, 31)

    def test_feb_non_leap(self):
        assert eomonth(pd.Timestamp(2025, 2, 1), 0) == pd.Timestamp(2025, 2, 28)

    def test_feb_leap(self):
        assert eomonth(pd.Timestamp(2024, 2, 1), 0) == pd.Timestamp(2024, 2, 29)

    def test_forward_month(self):
        assert eomonth(pd.Timestamp(2025, 5, 15), 2) == pd.Timestamp(2025, 7, 31)

    def test_year_rollover(self):
        assert eomonth(pd.Timestamp(2025, 11, 15), 3) == pd.Timestamp(2026, 2, 28)

    def test_nat_returns_nat(self):
        assert pd.isna(eomonth(pd.NaT))


class TestYfScalar:
    def test_one_full_year(self):
        # 30/360 from Jan 1 to Dec 31 → 360/360 = 1.0 ... almost
        # Actually 30*(12-1) + (31-1) with 31→30 adjustment: 330 + 30 = 360, but d2=31,d1=1 → d2 stays 31
        # Let's test explicit easy case
        result = yf_scalar(pd.Timestamp(2025, 1, 1), pd.Timestamp(2026, 1, 1))
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_half_year(self):
        result = yf_scalar(pd.Timestamp(2025, 1, 1), pd.Timestamp(2025, 7, 1))
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_same_day_zero(self):
        assert yf_scalar(pd.Timestamp(2025, 6, 15), pd.Timestamp(2025, 6, 15)) == 0.0

    def test_day_31_adjustment(self):
        # Excel convention: if d1=31, treat as 30
        # Jan 31 → Feb 28: raw would be 30*1 + (28-31) = 27; with d1 adjust: 30 + (28-30) = 28
        result = yf_scalar(pd.Timestamp(2025, 1, 31), pd.Timestamp(2025, 2, 28))
        assert result == pytest.approx(28.0 / 360.0, abs=1e-6)

    def test_nan_propagates(self):
        assert pd.isna(yf_scalar(pd.NaT, pd.Timestamp(2025, 1, 1)))
        assert pd.isna(yf_scalar(pd.Timestamp(2025, 1, 1), pd.NaT))


class TestYfVec:
    def test_matches_scalar_version(self):
        starts = pd.Series([pd.Timestamp(2025, 1, 1), pd.Timestamp(2025, 6, 15),
                             pd.Timestamp(2024, 3, 10)])
        ends   = pd.Series([pd.Timestamp(2025, 7, 1), pd.Timestamp(2026, 6, 15),
                             pd.Timestamp(2025, 3, 10)])
        vec    = yf_vec(starts, ends)
        for i in range(len(starts)):
            assert vec.iloc[i] == pytest.approx(
                yf_scalar(starts.iloc[i], ends.iloc[i]), abs=1e-6)

    def test_preserves_index(self):
        starts = pd.Series([pd.Timestamp(2025, 1, 1)], index=["emp_x"])
        ends   = pd.Series([pd.Timestamp(2025, 7, 1)], index=["emp_x"])
        result = yf_vec(starts, ends)
        assert result.index.tolist() == ["emp_x"]

    def test_empty_series(self):
        result = yf_vec(pd.Series([], dtype="datetime64[ns]"),
                         pd.Series([], dtype="datetime64[ns]"))
        assert len(result) == 0


class TestRoundup:
    def test_basic_roundup(self):
        assert roundup(1.21, 1) == pytest.approx(1.3, abs=1e-6)
        assert roundup(1.20, 1) == pytest.approx(1.2, abs=1e-6)

    def test_nan_returns_nan(self):
        assert pd.isna(roundup(np.nan, 1))
