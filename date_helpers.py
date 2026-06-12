"""Pure date/time helpers used throughout the budget calculations.

These functions implement Greek payroll calendar conventions (30/360 day
counting, bonus-month allocation rules) and are fully vectorised for pandas.
"""
from __future__ import annotations

from calendar import monthrange

import numpy as np
import pandas as pd


def eomonth(dt: pd.Timestamp, months: int = 0) -> pd.Timestamp:
    """Excel-style EOMONTH: end of month `months` away from `dt`."""
    if pd.isna(dt):
        return pd.NaT
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    return pd.Timestamp(y, m, monthrange(y, m)[1])


def roundup(x: float, decimals: int = 1) -> float:
    """Round up to `decimals` places. NaN-safe."""
    if pd.isna(x):
        return np.nan
    return np.ceil(x * 10**decimals) / 10**decimals


def yf_scalar(start, end) -> float:
    """30/360 year-fraction between two scalar dates. Matches Excel YEARFRAC basis 0."""
    if pd.isna(start) or pd.isna(end):
        return np.nan
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year,   end.month,   end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0


def yf_vec(s_series: pd.Series, e_series: pd.Series) -> pd.Series:
    """Vectorised 30/360 year-fraction."""
    s = pd.to_datetime(s_series)
    e = pd.to_datetime(e_series)
    y1 = s.dt.year
    m1 = s.dt.month
    d1 = s.dt.day
    y2 = e.dt.year
    m2 = e.dt.month
    d2 = e.dt.day
    d1 = d1.where(d1 != 31, 30)
    d2 = np.where((d2 == 31) & (d1 == 30), 30, d2)
    result = (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0
    return pd.Series(result, index=s_series.index, dtype=float)


def bonus_months_vec(
    H: pd.Series, R: pd.Series, ref_date: pd.Timestamp, CY: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute vector of (JP, BonM) — July-bonus months and total bonus months
    for each employee given Hire dates H, Retire dates R, reference date, and
    current calendar year CY.  Pure function — returns numpy arrays.
    """
    DBD = pd.Timestamp(CY, 12, 31)
    JBD = pd.Timestamp(CY,  7, 31)
    AV  = pd.Timestamp(CY,  4,  1)
    CHR = pd.Timestamp(CY,  5,  1)
    SCY = pd.Timestamp(CY,  1,  1)

    SR = R.where(~R.isna() & (R <= DBD), DBD)

    sr_clamped = SR.clip(upper=None)
    den_sr     = sr_clamped.where(sr_clamped > DBD, DBD)
    denom      = yf_vec(pd.Series([SCY] * len(H), index=H.index), den_sr)

    sr_A  = SR.clip(upper=DBD)
    num_A = yf_vec(pd.Series([JBD] * len(H), index=H.index), sr_A)
    h_B   = H.clip(lower=SCY)
    sr_B  = SR.clip(upper=DBD)
    num_B = yf_vec(h_B, sr_B)

    num    = np.where(H < JBD, num_A, num_B)
    raw_jp = np.where(denom > 0, (num / denom) * 0.5, 0.0)
    JP = np.where(H >= AV, np.ceil(raw_jp * 100) / 100, 0.0).astype(float)
    JP = np.where(pd.isna(H), 0.0, JP)

    el    = eomonth(DBD, 0)
    s_pc  = H.clip(lower=CHR).where(~pd.isna(H), CHR)
    r_pc  = R.clip(upper=el).where(~pd.isna(R), el)
    piece = (yf_vec(s_pc, r_pc) * 12.0 / 8.0).values
    # Christmas gift period must NEVER exceed 1.0
    piece = np.minimum(piece, 1.0)

    HL = (H <= CHR).values
    RO = (R.isna() | (R >= DBD)).values

    h_nr   = H.clip(lower=CHR).where(~pd.isna(H), CHR)
    piece2 = (yf_vec(h_nr, pd.Series([el] * len(H), index=H.index)) * 12.0 / 8.0).values
    piece2 = np.minimum(piece2, 1.0)

    BonM = np.where(HL &  RO, 1.0 + JP,
           np.where(HL & ~RO, piece + JP,
           np.where(~HL & RO, piece2 + JP,
                               piece + JP))).astype(float)
    return JP, BonM
