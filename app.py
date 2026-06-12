"""Integration tests: run full scenarios + pydantic validation coverage."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from mb.config import CONTRIB_RATE_MAP
from mb.scenarios import Scenario, ScenarioColumns, compare_scenarios, run_scenario
from tests.fixtures import DEFAULT_PROJECTION_DATE, build_sample_df


@pytest.fixture
def base_scenario() -> Scenario:
    return Scenario(
        name="Base",
        projection_date         = DEFAULT_PROJECTION_DATE,
        no_increase_cutoff      = date(2025, 8, 1),
        effective_increase_date = date(2026, 5, 1),
        payroll_periods         = 14,
        salary_increase_pct     = 0.03,
        salary_increase_pct2    = 0.05,
        apply_youth_discount    = True,
        youth_threshold         = 25,
        youth_discount_pp       = 0.0666,
        contrib_rate_map        = dict(CONTRIB_RATE_MAP),
    )


class TestScenarioValidation:
    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Scenario(name="")

    def test_negative_increase_too_large(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", salary_increase_pct=-0.30)

    def test_salary_increase_too_high(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", salary_increase_pct=0.60)

    def test_payroll_periods_out_of_range(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", payroll_periods=15)
        with pytest.raises(ValidationError):
            Scenario(name="Bad", payroll_periods=11)

    def test_youth_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", youth_threshold=17)
        with pytest.raises(ValidationError):
            Scenario(name="Bad", youth_threshold=36)

    def test_youth_discount_too_large(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", youth_discount_pp=0.25)

    def test_bad_rate_map_nonnumeric(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", contrib_rate_map={"40050": "zero"})  # type: ignore

    def test_bad_rate_map_out_of_range(self):
        with pytest.raises(ValidationError):
            Scenario(name="Bad", contrib_rate_map={"40050": 1.5})

    def test_date_ordering_enforced(self):
        with pytest.raises(ValidationError):
            Scenario(
                name="Bad",
                no_increase_cutoff      = date(2026, 8, 1),
                effective_increase_date = date(2026, 5, 1))

    def test_projection_date_accepts_str(self):
        s = Scenario(name="OK", projection_date="2025-10-01")
        assert s.projection_date == pd.Timestamp(2025, 10, 1)

    def test_signature_stable_across_equivalent_scenarios(self, base_scenario):
        s2 = base_scenario.model_copy()
        assert base_scenario.signature() == s2.signature()

    def test_signature_differs_when_params_change(self, base_scenario):
        s2 = base_scenario.model_copy(update={"salary_increase_pct": 0.05})
        assert base_scenario.signature() != s2.signature()


class TestRunScenario:
    def test_scenario_runs_end_to_end(self, base_scenario):
        df = build_sample_df()
        out, cols = run_scenario(df, base_scenario)
        for c in [cols.months_col, cols.fy_months_col, cols.fy_gross_col,
                  cols.fy_emp_cont_col, cols.total_cost_py,
                  cols.ann_gross_col, cols.ann_emp_cont_col, cols.fy_cost_by,
                  "Annual Training Cost", "Annual Meal Allowance/ Coupons Cost"]:
            assert c in out.columns, f"Missing output column: {c}"

    def test_scenario_column_names_use_year_suffixes(self, base_scenario):
        cols = ScenarioColumns.from_scenario(base_scenario)
        assert cols.months_col    == "Months Projection 25"
        assert cols.fy_months_col == "FY Months Budget 26"
        assert cols.fy_cost_by    == "FY PAYROLL COST BUDGET 2026"

    def test_all_fy_cost_values_are_nonnegative(self, base_scenario):
        df = build_sample_df()
        out, cols = run_scenario(df, base_scenario)
        values = pd.to_numeric(out[cols.fy_cost_by], errors="coerce").dropna()
        assert (values >= 0).all()

    def test_higher_increase_pct_yields_higher_budget(self, base_scenario):
        df = build_sample_df()
        out_low, cols_low = run_scenario(df, base_scenario)
        high = base_scenario.model_copy(update={"salary_increase_pct": 0.10})
        out_high, cols_high = run_scenario(df, high)
        assert out_high[cols_high.fy_cost_by].sum() > out_low[cols_low.fy_cost_by].sum()

    def test_youth_discount_reduces_budget(self, base_scenario):
        df = build_sample_df()
        out_on,  cols_on  = run_scenario(df, base_scenario)
        off = base_scenario.model_copy(update={"apply_youth_discount": False})
        out_off, cols_off = run_scenario(df, off)
        assert out_on[cols_on.fy_cost_by].sum() < out_off[cols_off.fy_cost_by].sum()


class TestCompareScenarios:
    def test_delta_is_b_minus_a(self, base_scenario):
        df = build_sample_df()
        out_a, cols_a = run_scenario(df, base_scenario)
        scenario_b = base_scenario.model_copy(
            update={"name": "B", "salary_increase_pct": 0.05})
        out_b, cols_b = run_scenario(df, scenario_b)
        summary = compare_scenarios(out_a, cols_a, out_b, cols_b)
        assert summary["delta_cost"] == pytest.approx(
            summary["total_cost_b"] - summary["total_cost_a"], abs=0.01)
        assert summary["delta_cost"] > 0

    def test_identical_scenarios_produce_zero_delta(self, base_scenario):
        df = build_sample_df()
        out_a, cols_a = run_scenario(df, base_scenario)
        out_b, cols_b = run_scenario(df, base_scenario)
        summary = compare_scenarios(out_a, cols_a, out_b, cols_b)
        assert summary["delta_cost"] == pytest.approx(0.0, abs=0.01)

    def test_by_division_breakdown(self, base_scenario):
        df = build_sample_df()
        out_a, cols_a = run_scenario(df, base_scenario)
        scenario_b = base_scenario.model_copy(
            update={"name": "B", "salary_increase_pct": 0.05})
        out_b, cols_b = run_scenario(df, scenario_b)
        summary = compare_scenarios(out_a, cols_a, out_b, cols_b)
        bd = summary["by_division"]
        assert {"Division", "Scenario A", "Scenario B", "Δ (€)", "Δ (%)"} <= set(bd.columns)


class TestScenarioRateMapOverride:
    def test_per_scenario_rate_map(self, base_scenario):
        df = build_sample_df()
        out_a, cols_a = run_scenario(df, base_scenario)
        custom_map = dict(CONTRIB_RATE_MAP)
        custom_map["40050"] = 0.30
        scenario_b = base_scenario.model_copy(
            update={"name": "B", "contrib_rate_map": custom_map})
        out_b, cols_b = run_scenario(df, scenario_b)
        assert out_b[cols_b.fy_cost_by].sum() > out_a[cols_a.fy_cost_by].sum()
