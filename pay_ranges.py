"""Budget-year formulas (FY Months Budget, Annual Gross Salary Budget)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from mb.calc.date_helpers import yf_scalar, yf_vec
from mb.config import MONTH_DIVISOR


def _ensure_hiring_date(df: pd.DataFrame) -> pd.DataFrame | None:
    if "Hiring Date" not in df.columns:
        if "Hire Date" in df.columns:
            return df.rename(columns={"Hire Date": "Hiring Date"})
        return None
    return df


def compute_fy_months_budget(
    df: pd.DataFrame,
    fy_base_date: pd.Timestamp,
    full_periods: float,
    budget_year: int,
) -> pd.DataFrame:
    """Active months in the budget year, incl. Easter/Summer/Christmas bonuses."""
    col_name = f"FY Months Budget {str(budget_year)[-2:]}"
    df = _ensure_hiring_date(df)
    if df is None or "Retire Date" not in df.columns:
        return df if df is not None else pd.DataFrame()

    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")
    CY = pd.to_datetime(fy_base_date).year
    PY = CY - 1
    SCY = pd.Timestamp(CY,  1,  1)
    APR = pd.Timestamp(CY,  4, 30)
    JUL = pd.Timestamp(CY,  7,  1)
    DEC = pd.Timestamp(CY, 12, 31)
    CHR = pd.Timestamp(CY,  5,  1)
    FP  = float(full_periods)
    DIV = MONTH_DIVISOR

    EoA = (H <= SCY) & (R.isna() | (R >= APR))
    EoJ = (H <= SCY) & (R.isna() | (R >= JUL))
    EoD = (H <= CHR) & (R.isna() | (R >= DEC))

    h_cl  = H.clip(lower=SCY).where(~pd.isna(H), SCY)
    r_apr = R.clip(upper=APR).where(~pd.isna(R), APR)
    an    = yf_vec(h_cl, r_apr)
    ad    = yf_scalar(SCY, APR)
    raw_ap = np.where(ad > 0, (an / ad) * 0.5, 0.0)
    AP = np.where(EoA, 0.5,
         np.where(H < APR, np.ceil(raw_ap * 10) / 10, 0.0)).astype(float)

    r_dec = R.clip(upper=DEC).where(~pd.isna(R), DEC)
    jn    = yf_vec(h_cl, r_dec)

    r_jd_end = R.clip(lower=DEC).where(~pd.isna(R), DEC)
    jd    = yf_vec(pd.Series([SCY] * len(H), index=H.index), r_jd_end)
    raw_jp = np.where(jd > 0, (jn / jd) * 0.5, 0.0)
    JP = np.where(EoJ, 0.5,
         np.where(H <= DEC, np.ceil(raw_jp * 10) / 10, 0.0)).astype(float)

    r_dd_end = R.clip(lower=DEC).where(~pd.isna(R), DEC)
    dd    = yf_vec(pd.Series([CHR] * len(H), index=H.index), r_dd_end)
    raw_dp = np.where(dd > 0, jn / dd, 0.0)
    DP = np.where(EoD, 1.0,
         np.where(H <= DEC, np.ceil(raw_dp * 10) / 10, 0.0)).astype(float)
    # Christmas gift period must NEVER exceed 1.0 — cap it explicitly
    DP = np.minimum(DP, 1.0)

    BonM = (AP + JP + DP).astype(float)
    YH, YR = H.dt.year, R.dt.year
    de = (pd.Timestamp(CY, 12, 31) - H).dt.days.fillna(0)
    NR = np.where(H <= SCY, np.where(BonM > 1.89, FP, (FP - 2.0) + BonM),
         np.where(YH == CY, (de / DIV) + BonM, 0.0)).astype(float)

    drs = (R - SCY).dt.days
    drh = (R - H).dt.days
    c1 = R.isna()
    c2 = (~R.isna()) & (YH <= PY) & (YR <= PY)
    c3 = (~R.isna()) & (YH <= PY) & (YR == CY)
    c4 = (~R.isna()) & (YH == CY) & (YR == CY)
    RC = np.where(c1, 0.0,
         np.where(c2, 0.0,
         np.where(c3, (drs / DIV) + BonM,
         np.where(c4, (drh / DIV) + BonM, np.nan)))).astype(float)

    final = np.where(~R.isna(), RC, NR)
    df[col_name] = np.round(np.where(pd.isna(final), np.nan, final), 1)
    return df


def compute_annual_gross_salary_budget(
    df: pd.DataFrame,
    effective_increase_date: pd.Timestamp,
    no_increase_cutoff: pd.Timestamp,
    inc_pct: float,
    inc_pct2: float,
    budget_year: int,
    active_months_col: str | None  = None,
    monthly_cost_col: str  = "Monthly Gross Salary (Current)",
    grade_col: str         = "Grade",
) -> pd.DataFrame:
    """Annual gross salary budget with grade-aware salary increase."""
    col_name = f"Annual Gross Salary FY Budget {budget_year}"
    BY2 = str(budget_year)[-2:]
    if active_months_col is None:
        active_months_col = f"FY Months Budget {BY2}"

    df = _ensure_hiring_date(df)
    if df is None:
        return pd.DataFrame()
    for col in ["Retire Date", active_months_col, monthly_cost_col]:
        if col not in df.columns:
            return df

    YD   = pd.to_datetime(effective_increase_date)
    H5   = pd.to_datetime(no_increase_cutoff)
    y1H5 = H5 + relativedelta(years=1)
    y1YD = YD + relativedelta(years=1)
    CY   = YD.year
    SCY = pd.Timestamp(CY,  1,  1)
    APR = pd.Timestamp(CY,  4, 30)
    DEC = pd.Timestamp(CY, 12, 31)
    CHR = pd.Timestamp(CY,  5,  1)

    H  = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R  = pd.to_datetime(df["Retire Date"], errors="coerce")
    AM = pd.to_numeric(df[active_months_col], errors="coerce").fillna(0.0)
    MC = pd.to_numeric(
        df[monthly_cost_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce").fillna(0.0)
    G  = (pd.to_numeric(
              df[grade_col].astype(str).str.replace(",", ".", regex=False).str.strip(),
              errors="coerce")
          if grade_col in df.columns else pd.Series(np.nan, index=df.index))

    mA = AM > 0
    iu = np.where(np.isclose(G, 0.1, atol=1e-9), inc_pct2, inc_pct)

    EoA = (H <= SCY) & (R.isna() | (R >= APR))
    EoD = (H <= CHR) & (R.isna() | (R >= DEC))

    h_cl  = H.clip(lower=SCY).where(~pd.isna(H), SCY)
    r_apr = R.clip(upper=APR).where(~pd.isna(R), APR)
    an    = yf_vec(h_cl, r_apr)
    ad    = yf_scalar(SCY, APR)
    raw_ap = np.where(ad > 0, (an / ad) * 0.5, 0.0)
    AP = np.where(EoA, 0.5,
         np.where(H < APR, np.ceil(raw_ap * 10) / 10, 0.0)).astype(float)

    r_dec = R.clip(upper=DEC).where(~pd.isna(R), DEC)
    dn    = yf_vec(h_cl, r_dec)
    r_dd_end = R.clip(lower=DEC).where(~pd.isna(R), DEC)
    dd    = yf_vec(pd.Series([CHR] * len(H), index=H.index), r_dd_end)
    raw_dp = np.where(dd > 0, dn / dd, 0.0)
    DP = np.where(EoD, 1.0,
         np.where(H >= SCY, np.ceil(raw_dp * 10) / 10, 0.0)).astype(float)
    # Christmas gift period must NEVER exceed 1.0 — cap it explicitly
    DP = np.minimum(DP, 1.0)

    DF  = np.where((H <= H5) & (R.isna() | (R > y1H5)), 1.0 + iu, 1.0)
    sp  = float(YD.month) - 0.5
    pre = np.minimum(AM, sp)
    post = np.maximum(AM - sp, 0.0)
    BC  = np.round(np.where((H <= H5) & (R.isna() | (R > y1YD)),
                             pre * MC + post * MC * (1.0 + iu),
                             AM * MC), 2)
    AL  = np.round(MC * 0.04166 * (AP + DP * DF), 2)
    df[col_name] = np.round(np.where(mA, BC + AL, 0.0), 2)
    return df
