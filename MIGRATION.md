"""Tests for the data loader (column normalisation + filters)."""
from __future__ import annotations

import pandas as pd
import pytest

from mb.data_loader import (apply_standard_filters, build_cost_center,
                             coerce_dates, drop_dup_cols, find_and_rename_col,
                             normalise_columns, normalise_salary)


class TestCoerceDates:
    def test_parses_dayfirst_strings(self):
        df = pd.DataFrame({"Hiring Date": ["15/01/2020", "03/06/2018"]})
        out = coerce_dates(df)
        assert out["Hiring Date"].iloc[0] == pd.Timestamp(2020, 1, 15)
        assert out["Hiring Date"].iloc[1] == pd.Timestamp(2018, 6, 3)

    def test_invalid_date_becomes_nat(self):
        df = pd.DataFrame({"Hiring Date": ["not a date"]})
        out = coerce_dates(df)
        assert pd.isna(out["Hiring Date"].iloc[0])

    def test_derives_hire_year(self):
        df = pd.DataFrame({"Hire Date": ["15/01/2020"]})
        out = coerce_dates(df)
        assert out["Hire Year"].iloc[0] == 2020


class TestFindAndRenameCol:
    def test_renames_matching_col(self):
        df = pd.DataFrame({"col_x": ["foo", "ADMINISTRATIVE", "bar"]})
        out = find_and_rename_col(df, "ADMINISTRATIVE", "Job Property")
        assert "Job Property" in out.columns
        assert "col_x" not in out.columns

    def test_does_not_overwrite_existing(self):
        df = pd.DataFrame({"col_x": ["ADMINISTRATIVE"],
                            "Job Property": ["anything"]})
        out = find_and_rename_col(df, "ADMINISTRATIVE", "Job Property")
        # "Job Property" already exists → col_x should NOT be renamed
        assert "col_x" in out.columns


class TestDropDupCols:
    def test_keeps_first_duplicate(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        out = drop_dup_cols(df)
        assert list(out.columns) == ["a", "b"]


class TestNormaliseColumns:
    def test_explicit_greek_rename(self):
        df = pd.DataFrame({
            "Ημ/νία γέννησης":     [pd.Timestamp(1980, 1, 1)],
            "Κωδικός εργαζομένου": ["E001"],
            "Επώνυμο":              ["Alpha"],
        })
        out = normalise_columns(df)
        assert "Date of Birth" in out.columns
        assert "Hrms Id"        in out.columns
        assert "Surname"        in out.columns


class TestApplyStandardFilters:
    def test_drops_past_retirees(self):
        df = pd.DataFrame({
            "Hrms Id":      ["E001", "E002"],
            "Retire Date": [pd.NaT, pd.Timestamp(2020, 1, 1)],
        })
        out = apply_standard_filters(df, pd.Timestamp(2025, 10, 1))
        assert list(out["Hrms Id"]) == ["E001"]

    def test_keeps_future_retirees(self):
        df = pd.DataFrame({
            "Hrms Id":      ["E001", "E002"],
            "Retire Date": [pd.NaT, pd.Timestamp(2026, 6, 1)],
        })
        out = apply_standard_filters(df, pd.Timestamp(2025, 10, 1))
        assert len(out) == 2

    def test_dedupes_by_hrms_id(self):
        df = pd.DataFrame({
            "Hrms Id":      ["E001", "E001"],
            "Retire Date": [pd.NaT, pd.NaT],
        })
        out = apply_standard_filters(df, pd.Timestamp(2025, 10, 1))
        assert len(out) == 1


class TestNormaliseSalary:
    def test_renames_greek_salary_col(self):
        df = pd.DataFrame({"Ονομαστικός μισθός": ["1500.00"]})
        out = normalise_salary(df)
        assert "Monthly Gross Salary (Current)" in out.columns
        assert out["Monthly Gross Salary (Current)"].iloc[0] == 1500.0

    def test_european_decimal_comma(self):
        df = pd.DataFrame({"Monthly Gross Salary (Current)": ["1500,50"]})
        out = normalise_salary(df)
        assert out["Monthly Gross Salary (Current)"].iloc[0] == 1500.5

    def test_daily_rate_multiplied_by_26(self):
        # A salary of 50 (< 90) is treated as daily and × 26
        df = pd.DataFrame({"Monthly Gross Salary (Current)": [50.0]})
        out = normalise_salary(df)
        assert out["Monthly Gross Salary (Current)"].iloc[0] == 1300.0


class TestBuildCostCenter:
    def test_combines_code_and_description(self):
        df = pd.DataFrame({
            "Κέντρο Κόστους":              ["CC01"],
            "Περιγραφή Κέντρου Κόστους": ["Production"],
        })
        out = build_cost_center(df)
        assert out["Cost Center"].iloc[0] == "CC01 - Production"

    def test_only_code(self):
        df = pd.DataFrame({"Κέντρο Κόστους": ["CC01"]})
        out = build_cost_center(df)
        assert out["Cost Center"].iloc[0] == "CC01"
