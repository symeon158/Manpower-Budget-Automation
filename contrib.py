"""Current-year projection formulas (Months Projection, FY Gross Projection)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mb.calc.date_helpers import bonus_months_vec, yf_scalar, yf_vec


def _ensure_hiring_date(df: pd.DataFrame) -> pd.DataFrame | None:
    if "Hiring Date" not in df.columns:
        if "Hire Date" in df.columns:
            return df.rename(columns={"Hire Date": "Hiring Date"})
        return None
    return df


def compute_months_projection(
    df: pd.DataFrame,
    projection_date: pd.Timestamp,
    proj_year: int,
) -> pd.DataFrame:
    """Compute active months for the current-year projection."""
    col_name = f"Months Projection {str(proj_year)[-2:]}"
    df = _ensure_hiring_date(df)
    if df is None or "Retire Date" not in df.columns:
        return df if df is not None else pd.DataFrame()

    prodate = pd.to_datetime(projection_date)
    CY = prodate.year
    PY = CY - 1
    BY = CY + 1
    ECY = pd.Timestamp(CY, 12, 31)

    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")
    YH, YR = H.dt.year, R.dt.year

    BM  = yf_scalar(prodate, ECY) * 12.0
    RM  = yf_vec(pd.Series([prodate] * len(H), index=H.index), R) * 12.0
    HM  = yf_vec(H, pd.Series([ECY] * len(H), index=H.index)) * 12.0
    RHM = yf_vec(H, R) * 12.0

    _JP, BonM = bonus_months_vec(H, R, prodate, CY)

    NR = np.where(YH == BY, 0.0,
         np.where((YH < PY) | (YH == PY), BM + BonM,
         np.where(YH == CY, np.where(H > prodate, HM + BonM, BM + BonM), 0.0))).astype(float)

    c1 = (YH <= CY) & (R <= prodate)
    c2 = (YH <  PY) & (YR == CY)
    c3 = (YH == PY) & (YR == CY)
    c4 = (YH == CY) & (YR == CY)
    c5 = (YH == CY) & (YR == BY)
    c6 = (YH <= PY) & (YR == BY)
    RC = np.where(c1, 0.0,
         np.where(c2 | c3, RM + BonM,
         np.where(c4, np.where(H <= prodate, RM + BonM, RHM + BonM),
         np.where(c5, np.where(H <= prodate, BM + BonM, HM + BonM),
         np.where(c6, BM + BonM, np.nan))))).astype(float)

    res = np.where(YH == BY, 0.0, np.where(~R.isna(), RC, NR)).astype(float)
    df[col_name] = np.where(pd.isna(res), np.nan, np.ceil(res * 10) / 10)
    return df


def compute_fy_gross_salary_projection(
    df: pd.DataFrame,
    projection_date: pd.Timestamp,
    proj_year: int,
    monthly_cost_col: str = "Monthly Gross Salary (Current)",
) -> pd.DataFrame:
    """Compute FY gross salary projection for the current year."""
    col_name = f"FY Gross Salary Projection For {str(proj_year)[-2:]}"
    df = _ensure_hiring_date(df)
    if df is None or "Retire Date" not in df.columns or monthly_cost_col not in df.columns:
        return df if df is not None else pd.DataFrame()

    df[monthly_cost_col] = pd.to_numeric(
        df[monthly_cost_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce")

    pd0 = pd.to_datetime(projection_date)
    CY = pd0.year
    PY = CY - 1
    BY = CY + 1
    ECY = pd.Timestamp(CY, 12, 31)

    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")
    YH, YR = H.dt.year, R.dt.year

    BM  = yf_scalar(pd0, ECY) * 12.0
    RM  = yf_vec(pd.Series([pd0] * len(H), index=H.index), R) * 12.0
    HM  = yf_vec(H, pd.Series([ECY] * len(H), index=H.index)) * 12.0
    RHM = yf_vec(H, R) * 12.0

    JP, BonM = bonus_months_vec(H, R, pd0, CY)

    NR = np.where(YH == BY, 0.0,
         np.where((YH < PY) | (YH == PY), BM + BonM,
         np.where(YH == CY, np.where(H > pd0, HM + BonM, BM + BonM), 0.0))).astype(float)

    c1 = (YH <= CY) & (R <= pd0)
    c2 = (YH <  PY) & (YR == CY)
    c3 = (YH == PY) & (YR == CY)
    c4 = (YH == CY) & (YR == CY)
    c5 = (YH == CY) & (YR == BY)
    c6 = (YH <= PY) & (YR == BY)
    RC = np.where(c1, 0.0,
         np.where(c2 | c3, RM + BonM,
         np.where(c4, np.where(H <= pd0, RM + BonM, RHM + BonM),
         np.where(c5, np.where(H <= pd0, BM + BonM, HM + BonM),
         np.where(c6, BM + BonM, np.nan))))).astype(float)

    raw = np.where(YH == BY, 0.0, np.where(~R.isna(), RC, NR)).astype(float)
    Rm  = np.ceil(np.where(pd.isna(raw), np.nan, raw) * 10) / 10
    XO  = (BonM - JP).astype(float)
    mc  = df[monthly_cost_col].astype(float).values
    df[col_name] = np.round(Rm * mc + XO * mc * 0.04166, 2)
    return df
