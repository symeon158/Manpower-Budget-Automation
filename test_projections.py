"""Tests for contribution rate lookup and employer contribution calc."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mb.calc.contrib import (apply_rate_lookup, compute_employer_contrib,
                              normalize_kratisis_code)
from mb.config import CONTRIB_RATE_MAP
from tests.fixtures import DEFAULT_PROJECTION_DATE, build_sample_df


class TestNormalizeCode:
    def test_extracts_5_digits_from_simple_string(self):
        s = pd.Series(["40050", "40084", "40510"])
        result = normalize_kratisis_code(s)
        assert list(result) == ["40050", "40084", "40510"]

    def test_extracts_from_messy_input(self):
        s = pd.Series(["  40050  ", "code-40084", "N.40510/xx"])
        result = normalize_kratisis_code(s)
        assert list(result) == ["40050", "40084", "40510"]

    def test_returns_nan_when_no_5_digit_block(self):
        s = pd.Series(["", "abc", "123"])
        result = normalize_kratisis_code(s)
        assert result.isna().all()


class TestApplyRateLookup:
    def test_maps_known_codes(self):
        df = pd.DataFrame({"Κωδικός Κράτησης": ["40050", "40602", "40090"]})
        out = apply_rate_lookup(df)
        assert out["Contributions%"].tolist() == [0.2179, 0.1879, 0.0140]
        assert out["Contrib Rate Matched"].tolist() == ["Yes", "Yes", "Yes"]

    def test_unknown_code_gets_nan(self):
        df = pd.DataFrame({"Κωδικός Κράτησης": ["99999"]})
        out = apply_rate_lookup(df)
        assert pd.isna(out["Contributions%"].iloc[0])
        assert out["Contrib Rate Matched"].iloc[0] == "No"

    def test_missing_column(self):
        df = pd.DataFrame({"Hrms Id": ["E001"]})
        out = apply_rate_lookup(df)
        assert pd.isna(out["Contributions%"].iloc[0])
        assert out["Contrib Rate Matched"].iloc[0] == "No"

    def test_scenario_specific_rate_map(self):
        """Passing a custom map should override the global defaults."""
        custom = {"40050": 0.30}
        df = pd.DataFrame({"Κωδικός Κράτησης": ["40050"]})
        out = apply_rate_lookup(df, rate_map=custom)
        assert out["Contributions%"].iloc[0] == 0.30


class TestYouthDiscount:
    def test_employee_under_25_gets_discount(self):
        df = build_sample_df()
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df,
            apply_youth_discount=True,
            youth_discount_pp=0.0666,
            youth_threshold=25,
            reference_date=DEFAULT_PROJECTION_DATE)
        # E002 is age 23 on 2025-10-01 → should be eligible
        e2 = df[df["Hrms Id"] == "E002"].iloc[0]
        assert e2["Youth Discount Applied"] == "Yes"
        # Effective rate = 0.2179 - 0.0666 = 0.1513
        assert e2["Effective Contribution Rate"] == pytest.approx(0.1513, abs=1e-4)
        # Monthly contribution = 1000 × 0.1513 = 151.30 (no add-on for this rate)
        assert e2["Monthly Employer's Contributions"] == pytest.approx(151.30, abs=0.01)

    def test_employee_over_25_no_discount(self):
        df = build_sample_df()
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df, apply_youth_discount=True, reference_date=DEFAULT_PROJECTION_DATE)
        e1 = df[df["Hrms Id"] == "E001"].iloc[0]
        assert e1["Youth Discount Applied"] == "No"
        assert e1["Effective Contribution Rate"] == pytest.approx(0.2179, abs=1e-6)
        # 1500 × 0.2179 = 326.85
        assert e1["Monthly Employer's Contributions"] == pytest.approx(326.85, abs=0.01)

    def test_discount_can_be_disabled(self):
        df = build_sample_df()
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df, apply_youth_discount=False, reference_date=DEFAULT_PROJECTION_DATE)
        e2 = df[df["Hrms Id"] == "E002"].iloc[0]
        assert e2["Youth Discount Applied"] == "No"
        assert e2["Effective Contribution Rate"] == pytest.approx(0.2179, abs=1e-6)

    def test_custom_threshold_and_discount(self):
        df = build_sample_df()
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df,
            apply_youth_discount=True,
            youth_discount_pp=0.10,          # 10pp instead of 6.66
            youth_threshold=30,              # under 30 instead of under 25
            reference_date=DEFAULT_PROJECTION_DATE)
        # E002 (age 23) should get it
        assert df[df["Hrms Id"] == "E002"]["Youth Discount Applied"].iloc[0] == "Yes"
        # E004 was born 1993-11-05 → age ~32 on 2025-10-01 → NOT eligible
        assert df[df["Hrms Id"] == "E004"]["Youth Discount Applied"].iloc[0] == "No"

    def test_fixed_addons_not_reduced_by_youth_discount(self):
        """The €30 add-on (rate 0.1879) applies even to under-25 employees."""
        # Create a synthetic under-25 with rate 0.1879
        df = pd.DataFrame([{
            "Hrms Id": "Y001",
            "Monthly Gross Salary (Current)": 1000.00,
            "Κωδικός Κράτησης": "40602",                   # 0.1879 + €30
            "Date of Birth": pd.Timestamp(2003, 1, 1),     # age 22 on 2025-10-01
        }])
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df, apply_youth_discount=True, reference_date=DEFAULT_PROJECTION_DATE)
        row = df.iloc[0]
        # Effective rate: 0.1879 - 0.0666 = 0.1213
        assert row["Effective Contribution Rate"] == pytest.approx(0.1213, abs=1e-4)
        # Monthly = 1000 × 0.1213 + €30 = 121.30 + 30 = 151.30
        assert row["Monthly Employer's Contributions"] == pytest.approx(151.30, abs=0.01)


class TestFixedAddons:
    def test_30_euro_addon(self):
        df = build_sample_df()
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df, apply_youth_discount=False, reference_date=DEFAULT_PROJECTION_DATE)
        e5 = df[df["Hrms Id"] == "E005"].iloc[0]
        # Rate 0.1879, salary 1800 → 1800 × 0.1879 + 30 = 338.22 + 30 = 368.22
        assert e5["Monthly Employer's Contributions"] == pytest.approx(368.22, abs=0.01)

    def test_25_euro_addon(self):
        df = pd.DataFrame([{
            "Hrms Id": "X001",
            "Monthly Gross Salary (Current)": 2000.00,
            "Κωδικός Κράτησης": "40510",     # 0.1738 + €25
            "Date of Birth": pd.Timestamp(1980, 1, 1),
        }])
        df = apply_rate_lookup(df)
        df = compute_employer_contrib(
            df, apply_youth_discount=False, reference_date=DEFAULT_PROJECTION_DATE)
        row = df.iloc[0]
        # 2000 × 0.1738 + 25 = 347.60 + 25 = 372.60
        assert row["Monthly Employer's Contributions"] == pytest.approx(372.60, abs=0.01)


class TestRateMapCoverage:
    """Make sure the global rate map has the codes we depend on in production."""
    @pytest.mark.parametrize("code", [
        "40010", "40011", "40012", "40050", "40060", "40061", "40070",
        "40084", "40090", "40380", "40510", "40602", "40603",
    ])
    def test_code_present(self, code):
        assert code in CONTRIB_RATE_MAP

    def test_all_rates_between_zero_and_one(self):
        for code, rate in CONTRIB_RATE_MAP.items():
            assert 0.0 <= rate <= 1.0, f"{code} has out-of-range rate {rate}"
