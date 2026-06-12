"""Scenario abstraction with pydantic validation."""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mb.calc.budget import (compute_annual_gross_salary_budget,
                             compute_fy_months_budget)
from mb.calc.contrib import (apply_rate_lookup, compute_annual_employer_contrib,
                              compute_employer_contrib)
from mb.calc.projections import (compute_fy_gross_salary_projection,
                                  compute_months_projection)
from mb.config import (CONTRIB_RATE_MAP, MEAL_ALLOWANCE_3, MEAL_ALLOWANCE_4,
                       TRAINING_COST_BANDS, YOUTH_AGE_THRESHOLD,
                       YOUTH_DISCOUNT_PP)


class Scenario(BaseModel):
    """All parameters that can differ between two scenarios. Pydantic-validated."""
    model_config = ConfigDict(arbitrary_types_allowed=True,
                              validate_assignment=True)

    name: str = Field(min_length=1, max_length=40)

    projection_date:          pd.Timestamp = Field(
        default_factory=lambda: pd.Timestamp(2025, 10, 1))
    no_increase_cutoff:       date = Field(
        default_factory=lambda: date(2025, 8, 1))
    effective_increase_date:  date = Field(
        default_factory=lambda: date(2026, 5, 1))

    payroll_periods:          int   = Field(default=14, ge=12, le=14)
    salary_increase_pct:      float = Field(default=0.03, ge=-0.20, le=0.50)
    salary_increase_pct2:     float = Field(default=0.05, ge=-0.20, le=0.50)

    apply_youth_discount:     bool  = True
    youth_threshold:          int   = Field(default=YOUTH_AGE_THRESHOLD, ge=18, le=35)
    youth_discount_pp:        float = Field(default=YOUTH_DISCOUNT_PP, ge=0.0, le=0.20)

    contrib_rate_map:         dict[str, float] = Field(
        default_factory=lambda: dict(CONTRIB_RATE_MAP))

    @field_validator("projection_date", mode="before")
    @classmethod
    def _coerce_projection_date(cls, v):
        return pd.to_datetime(v) if not isinstance(v, pd.Timestamp) else v

    @field_validator("contrib_rate_map")
    @classmethod
    def _validate_rates(cls, v: dict[str, float]) -> dict[str, float]:
        for code, rate in v.items():
            if not isinstance(rate, (int, float)):
                raise ValueError(f"Rate for {code} is not numeric: {rate!r}")
            if not (0.0 <= rate <= 1.0):
                raise ValueError(f"Rate for {code} out of range [0, 1]: {rate}")
        return v

    @model_validator(mode="after")
    def _check_date_ordering(self) -> "Scenario":
        if self.no_increase_cutoff >= self.effective_increase_date:
            raise ValueError(
                "`no_increase_cutoff` must be strictly before "
                "`effective_increase_date`")
        return self

    @property
    def proj_year(self) -> int:
        return pd.to_datetime(self.projection_date).year

    @property
    def budget_year(self) -> int:
        return self.proj_year + 1

    @property
    def budget_base_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.budget_year, 1, 1)

    @property
    def yy(self) -> str:
        return str(self.proj_year)[-2:]

    @property
    def byy(self) -> str:
        return str(self.budget_year)[-2:]

    def signature(self) -> str:
        parts = (
            str(self.projection_date),
            str(self.no_increase_cutoff),
            str(self.effective_increase_date),
            self.payroll_periods,
            self.salary_increase_pct,
            self.salary_increase_pct2,
            self.apply_youth_discount,
            self.youth_threshold,
            self.youth_discount_pp,
            tuple(sorted(self.contrib_rate_map.items())),
        )
        return hashlib.md5(str(parts).encode()).hexdigest()


class ScenarioColumns(BaseModel):
    model_config = ConfigDict(frozen=True)

    months_col:       str
    fy_months_col:    str
    fy_gross_col:     str
    fy_emp_cont_col:  str
    total_cost_py:    str
    ann_gross_col:    str
    ann_emp_cont_col: str
    fy_cost_by:       str

    @classmethod
    def from_scenario(cls, s: Scenario) -> "ScenarioColumns":
        yy, byy, by = s.yy, s.byy, s.budget_year
        return cls(
            months_col       = f"Months Projection {yy}",
            fy_months_col    = f"FY Months Budget {byy}",
            fy_gross_col     = f"FY Gross Salary Projection For {yy}",
            fy_emp_cont_col  = f"FY Employer's Contributions Projection {yy}",
            total_cost_py    = f"Total Payroll Projection Cost {yy}",
            ann_gross_col    = f"Annual Gross Salary FY Budget {by}",
            ann_emp_cont_col = f"Annual Employer's Contributions For {by}",
            fy_cost_by       = f"FY PAYROLL COST BUDGET {by}",
        )


def run_scenario(
    df_precontrib: pd.DataFrame, s: Scenario
) -> tuple[pd.DataFrame, ScenarioColumns]:
    df = df_precontrib.copy()
    cols = ScenarioColumns.from_scenario(s)

    df = apply_rate_lookup(df, rate_map=s.contrib_rate_map)
    df = compute_employer_contrib(
        df,
        apply_youth_discount = s.apply_youth_discount,
        youth_discount_pp    = s.youth_discount_pp,
        youth_threshold      = s.youth_threshold,
        reference_date       = s.projection_date)
    df = compute_months_projection(df, s.projection_date, s.proj_year)
    df = compute_fy_months_budget(df, s.budget_base_date,
                                   s.payroll_periods, s.budget_year)
    df = compute_fy_gross_salary_projection(df, s.projection_date, s.proj_year)
    df = compute_annual_gross_salary_budget(
        df,
        effective_increase_date = s.effective_increase_date,
        no_increase_cutoff      = s.no_increase_cutoff,
        inc_pct                 = s.salary_increase_pct,
        inc_pct2                = s.salary_increase_pct2,
        budget_year             = s.budget_year)

    monthly_contrib = next(
        (c for c in ["Monthly Employer's Contributions",
                     "Monthly Employer'S Contributions"]
         if c in df.columns), None)

    # ── Projection-year costs: only compute when projection months > 0 ─────
    proj_months = (pd.to_numeric(df[cols.months_col], errors="coerce").fillna(0.0)
                    if cols.months_col in df.columns else None)
    proj_active = (proj_months > 0) if proj_months is not None else None

    if monthly_contrib and proj_months is not None:
        raw = (pd.to_numeric(df[monthly_contrib], errors="coerce") *
                proj_months).round(2)
        df[cols.fy_emp_cont_col] = np.where(proj_active, raw, 0.0)

    if cols.fy_gross_col in df.columns and cols.fy_emp_cont_col in df.columns:
        # Zero the FY gross too when no projection months (defensive — most
        # formulas already produce 0, but this guarantees it)
        if proj_active is not None:
            df[cols.fy_gross_col] = np.where(
                proj_active,
                pd.to_numeric(df[cols.fy_gross_col], errors="coerce").fillna(0.0),
                0.0).round(2)
        df[cols.total_cost_py] = np.where(
            proj_active if proj_active is not None else True,
            (pd.to_numeric(df[cols.fy_gross_col],    errors="coerce").fillna(0) +
             pd.to_numeric(df[cols.fy_emp_cont_col], errors="coerce").fillna(0)),
            0.0).round(2)

    df = compute_annual_employer_contrib(
        df,
        ann_gross_col = cols.ann_gross_col,
        fy_months_col = cols.fy_months_col,
        out_col       = cols.ann_emp_cont_col)

    # ── Budget-year costs: only compute when budget months > 0 ─────────────
    budget_months = (pd.to_numeric(df[cols.fy_months_col], errors="coerce").fillna(0.0)
                      if cols.fy_months_col in df.columns else None)
    budget_active = (budget_months > 0) if budget_months is not None else None

    if cols.ann_gross_col in df.columns and cols.ann_emp_cont_col in df.columns:
        if budget_active is not None:
            df[cols.ann_gross_col] = np.where(
                budget_active,
                pd.to_numeric(df[cols.ann_gross_col], errors="coerce").fillna(0.0),
                0.0).round(2)
            df[cols.ann_emp_cont_col] = np.where(
                budget_active,
                pd.to_numeric(df[cols.ann_emp_cont_col], errors="coerce").fillna(0.0),
                0.0).round(2)
        df[cols.fy_cost_by] = np.where(
            budget_active if budget_active is not None else True,
            (pd.to_numeric(df[cols.ann_gross_col],   errors="coerce").fillna(0) +
             pd.to_numeric(df[cols.ann_emp_cont_col], errors="coerce").fillna(0)),
            0.0).round(2)

    df = _add_training_and_meal(df)

    if "Hire Date" in df.columns and "Hiring Date" not in df.columns:
        df = df.rename(columns={"Hire Date": "Hiring Date"})

    return df, cols


def _add_training_and_meal(df: pd.DataFrame) -> pd.DataFrame:
    if "Grade" in df.columns:
        gn = pd.to_numeric(
            df["Grade"].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce")
        conds, costs, prev_upper = [], [], None
        for upper, cost in TRAINING_COST_BANDS:
            conds.append(gn.notna() & (gn < upper if prev_upper is None else gn <= upper))
            costs.append(cost)
            prev_upper = upper
        df["Annual Training Cost"] = np.select(conds, costs, default=0).astype(float)
    else:
        df["Annual Training Cost"] = 0.0

    meal_col = "ΚΑΡΤΑ ΣΙΤΙΣΗΣ"
    if meal_col in df.columns:
        mv = pd.to_numeric(
            df[meal_col].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce")
        df["Annual Meal Allowance/ Coupons Cost"] = np.select(
            [mv.eq(3), mv.eq(4)],
            [MEAL_ALLOWANCE_3, MEAL_ALLOWANCE_4], default=0).astype(float)
    else:
        df["Annual Meal Allowance/ Coupons Cost"] = 0.0
    return df


def compare_scenarios(
    df_a: pd.DataFrame, cols_a: ScenarioColumns,
    df_b: pd.DataFrame, cols_b: ScenarioColumns,
) -> dict:
    def _sum(df, col):
        return float(pd.to_numeric(df[col], errors="coerce").sum()) if col in df.columns else 0.0
    def _hc(df):
        return int(df["Hrms Id"].nunique()) if "Hrms Id" in df.columns else 0
    def _avg_sal(df):
        if "Monthly Gross Salary (Current)" not in df.columns:
            return 0.0
        s = pd.to_numeric(df["Monthly Gross Salary (Current)"], errors="coerce")
        return float(s.mean()) if s.notna().any() else 0.0

    total_a = _sum(df_a, cols_a.fy_cost_by)
    total_b = _sum(df_b, cols_b.fy_cost_by)
    delta   = total_b - total_a
    pct     = (delta / total_a * 100) if total_a else 0.0

    by_div = pd.DataFrame()
    if "Division" in df_a.columns and "Division" in df_b.columns:
        div_a = (df_a.groupby("Division")[cols_a.fy_cost_by].sum()
                 if cols_a.fy_cost_by in df_a.columns else pd.Series(dtype=float))
        div_b = (df_b.groupby("Division")[cols_b.fy_cost_by].sum()
                 if cols_b.fy_cost_by in df_b.columns else pd.Series(dtype=float))
        by_div = pd.concat([div_a.rename("Scenario A"),
                             div_b.rename("Scenario B")], axis=1).fillna(0.0)
        by_div["Δ (€)"] = by_div["Scenario B"] - by_div["Scenario A"]
        by_div["Δ (%)"] = np.where(
            by_div["Scenario A"] > 0,
            (by_div["Δ (€)"] / by_div["Scenario A"] * 100).round(2), 0.0)
        by_div = by_div.round(2).reset_index()

    return {
        "total_cost_a": total_a, "total_cost_b": total_b,
        "delta_cost":   delta,   "delta_pct":    pct,
        "headcount_a":  _hc(df_a), "headcount_b": _hc(df_b),
        "avg_sal_a":    _avg_sal(df_a), "avg_sal_b": _avg_sal(df_b),
        "by_division":  by_div,
    }
