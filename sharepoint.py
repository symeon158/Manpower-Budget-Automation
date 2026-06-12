"""Employer contribution computation with under-25 discount support."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from mb.config import (CONTRIB_RATE_MAP, FIXED_ADDON_25, FIXED_ADDON_30,
                       YOUTH_AGE_THRESHOLD, YOUTH_DISCOUNT_PP)


def normalize_kratisis_code(series: pd.Series) -> pd.Series:
    """Extract a 5-digit Κωδικός Κράτησης from free-text cells."""
    s = series.astype(str).str.strip().str.replace(r"\D+", " ", regex=True)
    return s.str.extract(r"(\d{5})", expand=False)


def _is_near(series: pd.Series, value: float, atol: float = 1e-6) -> np.ndarray:
    return np.isclose(pd.to_numeric(series, errors="coerce"), value, atol=atol)


def apply_rate_lookup(
    df: pd.DataFrame,
    rate_map: Optional[dict[str, float]] = None,
    code_col: str = "Κωδικός Κράτησης",
) -> pd.DataFrame:
    """
    Map `code_col` → `Contributions%` using `rate_map` (defaults to CONTRIB_RATE_MAP).
    Also writes `Κωδικός Κράτησης (norm)` and `Contrib Rate Matched` columns.
    """
    if rate_map is None:
        rate_map = CONTRIB_RATE_MAP

    if code_col in df.columns:
        df["Κωδικός Κράτησης (norm)"] = normalize_kratisis_code(df[code_col])
        df["Contributions%"] = df["Κωδικός Κράτησης (norm)"].map(rate_map)
        df["Contrib Rate Matched"] = np.where(df["Contributions%"].notna(), "Yes", "No")
    else:
        df["Κωδικός Κράτησης (norm)"] = np.nan
        df["Contributions%"]           = np.nan
        df["Contrib Rate Matched"]     = "No"
    return df


def compute_employer_contrib(
    df: pd.DataFrame,
    apply_youth_discount: bool = True,
    youth_discount_pp: float   = YOUTH_DISCOUNT_PP,
    youth_threshold: int       = YOUTH_AGE_THRESHOLD,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute monthly employer contribution.

    Requires columns:
        - "Monthly Gross Salary (Current)"
        - "Contributions%"
        - "Date of Birth"  (only if youth discount is applied)

    Writes columns:
        - "Monthly Employer's Contributions"
        - "Effective Contribution Rate"
        - "Youth Discount Applied"
    """
    need = {"Monthly Gross Salary (Current)", "Contributions%"}
    if not need.issubset(df.columns):
        return df

    rate   = pd.to_numeric(df["Contributions%"], errors="coerce")
    salary = pd.to_numeric(df["Monthly Gross Salary (Current)"], errors="coerce")

    # ── Youth eligibility flag (age < threshold on reference_date) ──────────
    eligible = pd.Series(False, index=df.index)
    if apply_youth_discount and "Date of Birth" in df.columns:
        ref = pd.to_datetime(reference_date) if reference_date is not None else pd.Timestamp.today()
        dob = pd.to_datetime(df["Date of Birth"], errors="coerce")
        age = ((ref - dob).dt.days / 365.25).astype(float)
        eligible = (age.notna()) & (age < float(youth_threshold))

    # Effective rate after discount (clipped at 0 to be safe)
    eff_rate = rate.where(~eligible, (rate - youth_discount_pp).clip(lower=0))

    base   = eff_rate * salary
    add_30 = np.zeros(len(df), dtype=bool)
    add_25 = np.zeros(len(df), dtype=bool)
    for v in FIXED_ADDON_30:
        add_30 |= _is_near(rate, v)
    for v in FIXED_ADDON_25:
        add_25 |= _is_near(rate, v)

    amount = np.where(add_25, base + 25,
             np.where(add_30, base + 30, base))

    df["Monthly Employer's Contributions"] = np.round(amount, 2)
    df["Effective Contribution Rate"]      = eff_rate.round(6)
    df["Youth Discount Applied"]           = np.where(eligible, "Yes", "No")
    return df


def compute_annual_employer_contrib(
    df: pd.DataFrame,
    ann_gross_col: str,
    fy_months_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Annual employer contribution = (Annual Gross × effective rate)
                                   + fixed add-ons × active months.

    Fixed add-ons always use the ORIGINAL rate (the €25/€30 adds are not
    removed by the youth discount).
    """
    if not {"Contributions%", ann_gross_col, fy_months_col}.issubset(df.columns):
        return df

    rate = pd.to_numeric(
        df["Contributions%"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce")

    if "Effective Contribution Rate" in df.columns:
        eff_rate = pd.to_numeric(df["Effective Contribution Rate"],
                                  errors="coerce").fillna(rate)
    else:
        eff_rate = rate

    ann = pd.to_numeric(df[ann_gross_col], errors="coerce").fillna(0.0)
    mon = pd.to_numeric(df[fy_months_col], errors="coerce").fillna(0.0)

    add_30_mask = np.zeros(len(df), dtype=bool)
    add_25_mask = np.zeros(len(df), dtype=bool)
    for v in FIXED_ADDON_30:
        add_30_mask |= _is_near(rate, v)
    for v in FIXED_ADDON_25:
        add_25_mask |= _is_near(rate, v)

    df[out_col] = np.round(
        ann * eff_rate
        + np.where(add_30_mask, 30.0 * mon, 0.0)
        + np.where(add_25_mask, 25.0 * mon, 0.0),
        2)
    return df
