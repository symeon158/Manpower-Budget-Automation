# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import io

st.title("📊 Manpower Budget Automation for Alumil S.A. & Subsidiaries")

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar inputs
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Projection Parameters")

# Cache reset (helps when file contents/dtypes change)
if st.sidebar.button("♻️ Reset file read cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Re-run with your files.")

format_mode = st.sidebar.radio(
    "Row coloring mode", ("None", "Conditional"), index=0,
    help="We’ll wire this up after basic display works."
)

payroll_periods = st.sidebar.number_input(
    "Payroll Periods per Year", value=14, step=1,
    help="e.g., 12 for monthly payroll"
)

projection_date = pd.to_datetime(
    st.sidebar.date_input("Projection Date (for 2025)", value=datetime(2025, 10, 1))
)
no_increase_cutoff = st.sidebar.date_input(
    "No Increases will be granted after this Date",
    value=datetime(2025, 8, 1)
)
effective_increase_date = st.sidebar.date_input(
    "Effective Date of Salary Increases",
    value=datetime(2026, 5, 1)
)

New_Hires = st.sidebar.date_input(
    "Date Threshold for New Hires",
    value=datetime(2025, 4, 1)
)

budget_year = projection_date.year + 1
budget_base_date = pd.Timestamp(budget_year, 1, 1)

salary_increase_pct = st.sidebar.number_input(
    "Average Salary Increase %", value=3.0, step=0.1
) / 100.0

salary_increase_pct2 = st.sidebar.number_input(
    "Average Salary Increase % for 0.1", value=5.0, step=0.1
) / 100.0

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
DATE_CANDIDATES = [
    "Ημ/νία γέννησης", "Ημ/νία αποχώρησης", "Ημ/νία πρόσληψης",
    "Hiring Date", "Retire Date", "Date", "Date of Birth", "Hire Date"
]

@st.cache_data(show_spinner=True)
def read_any(uploaded_file, sheet_name=None, header_row=0):
    """Read Excel/CSV robustly (Greek encodings, ; or , delimiters)."""
    name = getattr(uploaded_file, "name", "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)

    # CSV fallbacks
    for enc, delim in [("iso-8859-7", ";"), ("utf-8", ";"), ("utf-8", ",")]:
        try:
            return pd.read_csv(uploaded_file, encoding=enc, delimiter=delim)
        except Exception:
            pass
    raise ValueError("Could not read the uploaded file as CSV/Excel.")

def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known date columns (Greek & English) with dayfirst=True."""
    for c in DATE_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Derived years (works regardless of which naming is present)
    if "Ημ/νία πρόσληψης" in df.columns and "Hire Year" not in df.columns:
        df["Hire Year"] = pd.to_datetime(df["Ημ/νία πρόσληψης"], errors="coerce", dayfirst=True).dt.year
    if "Ημ/νία αποχώρησης" in df.columns and "Departure Year" not in df.columns:
        df["Departure Year"] = pd.to_datetime(df["Ημ/νία αποχώρησης"], errors="coerce", dayfirst=True).dt.year
    if "Hire Date" in df.columns and "Hire Year" not in df.columns:
        df["Hire Year"] = pd.to_datetime(df["Hire Date"], errors="coerce").dt.year
    if "Retire Date" in df.columns and "Departure Year" not in df.columns:
        df["Departure Year"] = pd.to_datetime(df["Retire Date"], errors="coerce").dt.year
    return df

def find_and_rename_column_with_exact_value(df: pd.DataFrame, value: str, new_name: str) -> pd.DataFrame:
    """
    Find the first column that contains an EXACT row value (stripped) == `value`,
    then rename that column to `new_name`. Handles mixed dtypes safely.
    """
    target = str(value).strip()
    for col in df.columns:
        series = df[col]
        if not isinstance(series, pd.Series):
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            try:
                if series.dropna().astype(str).str.strip().eq(target).any():
                    if new_name not in df.columns:
                        df = df.rename(columns={col: new_name})
                    break
            except Exception:
                pass
    return df

def to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()

def normalize_code(series: pd.Series) -> pd.Series:
    """
    Extract the first 5-digit booking code from each cell.
    Examples:
      '40,602'     -> '40602'
      '40602,0'    -> '40602'
      '40.602,0'   -> '40602'
      'Κωδ: 40602' -> '40602'
    """
    s = series.astype(str).str.strip()
    # Replace all non-digits with a space, then extract the first 5-digit token
    s = s.str.replace(r"\D+", " ", regex=True)
    code5 = s.str.extract(r"(\d{5})", expand=False)
    return code5

# --- Helper: drop duplicate-named columns, keep first ---
def drop_dup_named_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

def compute_employer_contrib(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute employer contribution amount from Contributions% and Monthly Gross Salary (Current).
    Rules:
      - If rate == 0.1738 → rate*salary + 25
      - If rate == 0.1879 → rate*salary + 30
      - Else → rate*salary
    Writes: "Monthly Employer's Contributions" (rounded to 2 decimals).
    """
    need = {"Monthly Gross Salary (Current)", "Contributions%"}
    if not need.issubset(df.columns):
        st.warning("⚠️ Missing columns to compute employer contributions (need salary & Contributions%).")
        return df

    rate = pd.to_numeric(df["Contributions%"], errors="coerce")
    salary = pd.to_numeric(df["Monthly Gross Salary (Current)"], errors="coerce")

    base = rate * salary
    add_25 = np.isclose(rate, 0.1738, atol=1e-6)
    add_30 = np.isclose(rate, 0.1879, atol=1e-6)

    amount = np.where(add_25, rate * salary + 25, base)
    amount = np.where(add_30, rate * salary + 30, amount)

    df["Monthly Employer's Contributions"] = np.round(amount, 2)
    return df

from calendar import monthrange

def eomonth(dt: pd.Timestamp, months: int = 0) -> pd.Timestamp:
    """Excel-like EOMONTH(dt, months). Returns last day of month after offset."""
    if pd.isna(dt):
        return pd.NaT
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    last_day = monthrange(y, m)[1]
    return pd.Timestamp(y, m, last_day)

def yearfrac(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Approximate Excel YEARFRAC(start, end) with ACT/365.
    (If you need exact 30/360 later, we can switch.)
    """
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return (end - start).days / 365.0


def yearfrac_30360_us(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day
    # US (NASD) 30/360 rules
    if d1 == 31: d1 = 30
    if d2 == 31 and d1 == 30: d2 = 30
    return ((360*(y2 - y1)) + (30*(m2 - m1)) + (d2 - d1)) / 360.0


def roundup(x, decimals=1):
    if pd.isna(x):
        return np.nan
    factor = 10 ** decimals
    return np.ceil(x * factor) / factor


def compute_months_projection_25(df: pd.DataFrame, projection_date: pd.Timestamp) -> pd.DataFrame:
    """
    Python translation of your revised Excel LET() for 'Months Projection 25',
    with robust datetime handling to avoid int↔Timestamp comparisons.
    """
    # Ensure expected columns
    if "Hiring Date" not in df.columns:
        if "Hire Date" in df.columns:
            df = df.rename(columns={"Hire Date": "Hiring Date"})
        else:
            st.warning("⚠️ Cannot compute 'Months Projection 25' (missing 'Hiring Date').")
            return df
    if "Retire Date" not in df.columns:
        st.warning("⚠️ Cannot compute 'Months Projection 25' (missing 'Retire Date').")
        return df

    # Scalars from projection_date (prodate)
    prodate    = pd.to_datetime(projection_date)
    CurrYear   = prodate.year
    PrevYear   = CurrYear - 1
    BudgetYear = CurrYear + 1

    # Anchors
    StartCurrYear = pd.Timestamp(CurrYear, 1, 1)
    EndCurYear    = pd.Timestamp(CurrYear, 12, 31)
    DecBonusDate  = pd.Timestamp(CurrYear, 12, 31)
    JulyBonusDate = pd.Timestamp(CurrYear, 7, 31)  # end of July
    AprilVac      = pd.Timestamp(CurrYear, 4, 1)
    Christmther   = pd.Timestamp(CurrYear, 5, 1)

    # Row data as proper datetimes
    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")
    YearH = H.dt.year
    YearR = R.dt.year

    # Base months
    BaseMonths   = yearfrac_30360_us(prodate, EndCurYear) * 12.0
    RetireMonths = (R.apply(lambda x: yearfrac_30360_us(prodate, x)) * 12.0).astype(float)
    HireMonths   = (H.apply(lambda x: yearfrac_30360_us(x, EndCurYear)) * 12.0).astype(float)
    RetireHiring = (pd.Series([yearfrac_30360_us(h, r) for h, r in zip(H, R)]) * 12.0).astype(float)

    # SafeRetire as a pandas Series of Timestamps (no np.where for dates)
    safe_retire = R.fillna(DecBonusDate)

    # ---------- JulyPart (robust) ----------
    def _july_part(h: pd.Timestamp, sr: pd.Timestamp) -> float:
        # Coerce sr to Timestamp just in case
        if pd.isna(sr):
            sr = DecBonusDate
        else:
            sr = pd.to_datetime(sr)

        if pd.isna(h):
            return 0.0

        if h >= AprilVac:
            # Denominator: yearfrac_30360_us(StartCurrYear, max(DecBonusDate, SafeRetire))
            denom_end = sr if (sr > DecBonusDate) else DecBonusDate
            denom = yearfrac_30360_us(StartCurrYear, denom_end)

            if h < JulyBonusDate:
                num_end = sr if (sr < DecBonusDate) else DecBonusDate
                num = yearfrac_30360_us(JulyBonusDate, num_end)
            else:  # h >= JulyBonusDate
                start = h if (h > StartCurrYear) else StartCurrYear
                end   = sr if (sr < DecBonusDate) else DecBonusDate
                num   = yearfrac_30360_us(start, end)

            val = 0.0 if (pd.isna(num) or pd.isna(denom) or denom <= 0) else (num / denom) * 0.5
            return roundup(val, 2)
        else:
            return 0.0

    JulyPart = np.array([_july_part(h, sr) for h, sr in zip(H, safe_retire)], dtype=float)

    # ---------- BonusMonths ----------
    def _piece(h: pd.Timestamp, r: pd.Timestamp) -> float:
        start = Christmther if pd.isna(h) else (h if h > Christmther else Christmther)
        end_lim = eomonth(DecBonusDate, 0)
        end   = end_lim if (pd.isna(r) or r > end_lim) else r
        return yearfrac_30360_us(start, end) * 12.0 / 8.0

    H_le_Christ = (H <= Christmther)
    R_blank_or_after_dec = (R.isna() | (R >= DecBonusDate))
    piece_vec = np.array([_piece(h, r) for h, r in zip(H, R)], dtype=float)

    BonusMonths = np.where(
        H_le_Christ & R_blank_or_after_dec,
        1.0 + JulyPart,
        np.where(
            H_le_Christ & ~R_blank_or_after_dec,
            piece_vec + JulyPart,
            np.where(
                (~H_le_Christ) & R_blank_or_after_dec,
                np.array([
                    yearfrac_30360_us(h if (not pd.isna(h) and h > Christmther) else Christmther,
                             eomonth(DecBonusDate, 0)) * 12.0 / 8.0
                    for h in H
                ]) + JulyPart,
                piece_vec + JulyPart
            )
        )
    ).astype(float)

    # ---------- NoRetire ----------
    NoRetire = np.where(
        YearH == BudgetYear, 0.0,
        np.where(
            (YearH < PrevYear) | (YearH == PrevYear),
            BaseMonths + BonusMonths,
            np.where(
                YearH == CurrYear,
                np.where(H > prodate, HireMonths + BonusMonths, BaseMonths + BonusMonths),
                0.0
            )
        )
    ).astype(float)

    # ---------- RetireCalc (IFS) ----------
    cond1 = (YearH <= CurrYear) & (R <= prodate)
    cond2 = (YearH < PrevYear) & (YearR == CurrYear)
    cond3 = (YearH == PrevYear) & (YearR == CurrYear)
    cond4 = (YearH == CurrYear) & (YearR == CurrYear)
    cond5 = (YearH == CurrYear) & (YearR == BudgetYear)
    cond6 = (YearH <= PrevYear) & (YearR == BudgetYear)

    RetireCalc = np.where(
        cond1, 0.0,
        np.where(
            cond2 | cond3, RetireMonths + BonusMonths,
            np.where(
                cond4, np.where(H <= prodate, RetireMonths + BonusMonths, RetireHiring + BonusMonths),
                np.where(
                    cond5, np.where(H <= prodate, BaseMonths + BonusMonths, HireMonths + BonusMonths),
                    np.where(
                        cond6, BaseMonths + BonusMonths,
                        np.nan
                    )
                )
            )
        )
    ).astype(float)

    # ---------- Final ----------
    result = np.where(
        YearH == BudgetYear, 0.0,
        np.where(~R.isna(), RetireCalc, NoRetire)
    ).astype(float)

    df["Months Projection 25"] = [roundup(x, 1) for x in result]
    return df




def compute_fy_months_budget_26(df: pd.DataFrame, fy_base_date: pd.Timestamp, full_periods: int, divisor: float = 30.42) -> pd.DataFrame:
    """
    Excel -> Python for 'FY Months Budget 26'.

    Mapping:
      CurrYear = YEAR(fy_base_date)
      PrevYear = CurrYear - 1
      StartCurrYear = 1/1/CurrYear
      EndCurrYear   = 12/31/CurrYear
      FullPeriods   = 12
      Divisor       = 30.42  (days per month)
    """
    if "Hiring Date" not in df.columns:
        if "Hire Date" in df.columns:
            df = df.rename(columns={"Hire Date": "Hiring Date"})
        else:
            st.warning("⚠️ Missing 'Hiring Date' for FY Months Budget 26.")
            return df
    if "Retire Date" not in df.columns:
        st.warning("⚠️ Missing 'Retire Date' for FY Months Budget 26.")
        return df

    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")

    CurrYear = pd.to_datetime(fy_base_date).year
    PrevYear = CurrYear - 1

    StartCurrYear = pd.Timestamp(CurrYear, 1, 1)
    EndCurrYear   = pd.Timestamp(CurrYear, 12, 31)
    AprilDate     = pd.Timestamp(CurrYear, 4, 30)
    JulyDate      = pd.Timestamp(CurrYear, 7, 1)
    DecDate       = pd.Timestamp(CurrYear, 12, 31)
    ChristmasThr  = pd.Timestamp(CurrYear, 5, 1)

    FullPeriods = float(payroll_periods)
    Divisor = float(30.42)
    
    # Booleans
    EmployedOnApril = (H <= StartCurrYear) & (R.isna() | (R >= AprilDate))
    EmployedOnJuly  = (H <= StartCurrYear) & (R.isna() | (R >= JulyDate))
    EmployedOnDec   = (H <= ChristmasThr)  & (R.isna() | (R >= DecDate))

    # AprilPart
    # IF(EmployedOnApril; 0.5; IF(H < AprilDate; ROUNDUP( yearfrac_30360_us(MAX(H,Start), MIN(April,R)) / yearfrac_30360_us(Start,April) * 0.5 ;1); 0))
    april_num = [
        yearfrac_30360_us(max(h, StartCurrYear) if not pd.isna(h) else StartCurrYear,
                 min(AprilDate, r) if not pd.isna(r) else AprilDate)
        for h, r in zip(H, R)
    ]
    april_den = yearfrac_30360_us(StartCurrYear, AprilDate)  # constant
    AprilPart = np.where(
        EmployedOnApril, 0.5,
        np.where(
            (H < AprilDate),
            [roundup((num / april_den) * 0.5, 1) if (not pd.isna(num) and april_den > 0) else 0.0 for num in april_num],
            0.0
        )
    ).astype(float)

    # JulyPart
    # IF(EmployedOnJuly; 0.5; IF(H<=DecDate; ROUNDUP(yearfrac_30360_us(MAX(H,Start), MIN(Dec,R)) / yearfrac_30360_us(Start, MAX(Dec,R)) * 0.5;1); 0))
    july_num = [
        yearfrac_30360_us(max(h, StartCurrYear) if not pd.isna(h) else StartCurrYear,
                 min(DecDate, r) if not pd.isna(r) else DecDate)
        for h, r in zip(H, R)
    ]
    july_den = [
        yearfrac_30360_us(StartCurrYear, max(DecDate, r) if not pd.isna(r) else DecDate)
        for r in R
    ]
    JulyPart = np.where(
        EmployedOnJuly, 0.5,
        np.where(
            (H <= DecDate),
            [roundup(((n / d) * 0.5) if (not pd.isna(n) and not pd.isna(d) and d > 0) else 0.0, 1)
             for n, d in zip(july_num, july_den)],
            0.0
        )
    ).astype(float)

    # DecPart
    # IF(EmployedOnDec; 1; IF(H<=DecDate; ROUNDUP(yearfrac_30360_us(MAX(H,Start), MIN(Dec,R)) / yearfrac_30360_us(ChristmasThr, MAX(Dec,R)) * 1;1); 0))
    dec_num = july_num  # same numerator as JulyPart: yearfrac_30360_us(MAX(H,Start), MIN(Dec,R))
    dec_den = [
        yearfrac_30360_us(ChristmasThr, max(DecDate, r) if not pd.isna(r) else DecDate)
        for r in R
    ]
    DecPart = np.where(
        EmployedOnDec, 1.0,
        np.where(
            (H <= DecDate),
            [roundup(((n / d) * 1.0) if (not pd.isna(n) and not pd.isna(d) and d > 0) else 0.0, 1)
             for n, d in zip(dec_num, dec_den)],
            0.0
        )
    ).astype(float)

    BonusMonths = (AprilPart + JulyPart + DecPart).astype(float)

    YearH = H.dt.year

    # NoRetire
    # IF(H <= Start) THEN
    #     IF(BonusMonths > 1.89; FullPeriods; (FullPeriods - 2) + BonusMonths)
    # ELSE IF(YEAR(H) = CurrYear)
    #     (EndCurrYear - H)/Divisor + BonusMonths
    # ELSE 0
    days_end_minus_h = (EndCurrYear - H).dt.days
    days_end_minus_h = days_end_minus_h.where(~pd.isna(days_end_minus_h), 0)

    NoRetire = np.where(
        H <= StartCurrYear,
        np.where(BonusMonths > 1.89, FullPeriods, (FullPeriods - 2.0) + BonusMonths),
        np.where(
            YearH == CurrYear,
            (days_end_minus_h / Divisor) + BonusMonths,
            0.0
        )
    ).astype(float)

    # RetireCalc
    # IF(R=""; 0; IFS(
    #   YEAR(H) <= PrevYear & YEAR(R) <= PrevYear → 0
    #   YEAR(H) <= PrevYear & YEAR(R) = CurrYear → (R - Start)/Div + BonusMonths
    #   YEAR(H) = CurrYear  & YEAR(R) = CurrYear → (R - H)    /Div + BonusMonths
    #   TRUE → NA()
    # ))
    YearR = R.dt.year
    days_r_minus_start = (R - StartCurrYear).dt.days
    days_r_minus_start = days_r_minus_start.where(~pd.isna(days_r_minus_start), np.nan)

    days_r_minus_h = (R - H).dt.days
    days_r_minus_h = days_r_minus_h.where(~pd.isna(days_r_minus_h), np.nan)

    cond1 = R.isna()
    cond2 = (~R.isna()) & (YearH <= PrevYear) & (YearR <= PrevYear)
    cond3 = (~R.isna()) & (YearH <= PrevYear) & (YearR == CurrYear)
    cond4 = (~R.isna()) & (YearH == CurrYear) & (YearR == CurrYear)

    RetireCalc = np.where(
        cond1, 0.0,
        np.where(
            cond2, 0.0,
            np.where(
                cond3, (days_r_minus_start / Divisor) + BonusMonths,
                np.where(
                    cond4, (days_r_minus_h / Divisor) + BonusMonths,
                    np.nan  # TRUE; NA()
                )
            )
        )
    ).astype(float)

    # Final: ROUND(IF(R<>""; RetireCalc; NoRetire); 1)
    final_val = np.where(~R.isna(), RetireCalc, NoRetire)
    df["FY Months Budget 26"] = [round(np.nan if pd.isna(x) else x, 1) for x in final_val]
    return df

def compute_fy_gross_salary_projection_25(
    df: pd.DataFrame,
    projection_date: pd.Timestamp,
    monthly_cost_col: str = "Monthly Gross Salary (Current)",
) -> pd.DataFrame:
    """
    Excel LET() → Python for 'FY Gross Salary Projection For 25'.

    Depends on:
      - Hiring Date, Retire Date (datetime columns)
      - Monthly cost column (default: 'Monthly Gross Salary (Current)')

    Uses the same JulyPart / BonusMonths / ResultMonths logic as Months Projection 25,
    then computes:
      TotalCost = ResultMonths * MonthlyCost + XmasOnlyMonths * MonthlyCost * BonusRate
    with BonusRate = 0.04166 and rounds to 2 decimals.
    """
    # ---- Preconditions ----
    if "Hiring Date" not in df.columns:
        if "Hire Date" in df.columns:
            df = df.rename(columns={"Hire Date": "Hiring Date"})
        else:
            st.warning("⚠️ Missing 'Hiring Date' — cannot compute FY Gross Salary Projection For 25.")
            return df
    if "Retire Date" not in df.columns:
        st.warning("⚠️ Missing 'Retire Date' — cannot compute FY Gross Salary Projection For 25.")
        return df
    if monthly_cost_col not in df.columns:
        st.warning(f"⚠️ Missing '{monthly_cost_col}' — cannot compute FY Gross Salary Projection For 25.")
        return df

    # Ensure numeric monthly cost
    df[monthly_cost_col] = pd.to_numeric(
        df[monthly_cost_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # ---- Anchors from projection_date ----
    prodate    = pd.to_datetime(projection_date)
    CurrYear   = prodate.year
    PrevYear   = CurrYear - 1
    BudgetYear = CurrYear + 1

    StartCurrYear = pd.Timestamp(CurrYear, 1, 1)
    EndCurYear    = pd.Timestamp(CurrYear, 12, 31)
    DecBonusDate  = pd.Timestamp(CurrYear, 12, 31)
    JulyBonusDate = pd.Timestamp(CurrYear, 7, 31)  # end of July (per your sheet)
    AprilVac      = pd.Timestamp(CurrYear, 4, 1)
    Christmther   = pd.Timestamp(CurrYear, 5, 1)   # threshold for Christmas bonus logic

    # ---- Row dates ----
    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")
    YearH = H.dt.year
    YearR = R.dt.year

    # ---- Base months pieces (ACT/365) ----
    BaseMonths   = yearfrac_30360_us(prodate, EndCurYear) * 12.0
    RetireMonths = (R.apply(lambda x: yearfrac_30360_us(prodate, x)) * 12.0).astype(float)
    HireMonths   = (H.apply(lambda x: yearfrac_30360_us(x, EndCurYear)) * 12.0).astype(float)
    RetireHiring = (pd.Series([yearfrac_30360_us(h, r) for h, r in zip(H, R)]) * 12.0).astype(float)

    # SafeRetire = R if present else DecBonusDate (keep as Timestamp)
    safe_retire = R.fillna(DecBonusDate)

    # ---- JulyPart (new piecewise with ROUNDUP(...,2)) ----
    def _july_part(h: pd.Timestamp, sr: pd.Timestamp) -> float:
        if pd.isna(h):
            return 0.0
        sr = DecBonusDate if pd.isna(sr) else pd.to_datetime(sr)

        if h >= AprilVac:
            denom_end = sr if (sr > DecBonusDate) else DecBonusDate
            denom = yearfrac_30360_us(StartCurrYear, denom_end)

            if h < JulyBonusDate:
                num_end = sr if (sr < DecBonusDate) else DecBonusDate
                num = yearfrac_30360_us(JulyBonusDate, num_end)
            else:  # h >= JulyBonusDate
                start = h if (h > StartCurrYear) else StartCurrYear
                end   = sr if (sr < DecBonusDate) else DecBonusDate
                num   = yearfrac_30360_us(start, end)

            val = 0.0 if (pd.isna(num) or pd.isna(denom) or denom <= 0) else (num / denom) * 0.5
            return roundup(val, 2)
        return 0.0

    JulyPart = np.array([_july_part(h, sr) for h, sr in zip(H, safe_retire)], dtype=float)

    # ---- BonusMonths (same as Months Projection 25, plus JulyPart) ----
    def _piece(h: pd.Timestamp, r: pd.Timestamp) -> float:
        start   = Christmther if pd.isna(h) else (h if h > Christmther else Christmther)
        end_lim = eomonth(DecBonusDate, 0)
        end     = end_lim if (pd.isna(r) or r > end_lim) else r
        return yearfrac_30360_us(start, end) * 12.0 / 8.0

    H_le_Christ = (H <= Christmther)
    R_blank_or_after_dec = (R.isna() | (R >= DecBonusDate))
    piece_vec = np.array([_piece(h, r) for h, r in zip(H, R)], dtype=float)

    BonusMonths = np.where(
        H_le_Christ & R_blank_or_after_dec,
        1.0 + JulyPart,
        np.where(
            H_le_Christ & ~R_blank_or_after_dec,
            piece_vec + JulyPart,
            np.where(
                (~H_le_Christ) & R_blank_or_after_dec,
                np.array([
                    yearfrac_30360_us(h if (not pd.isna(h) and h > Christmther) else Christmther,
                             eomonth(DecBonusDate, 0)) * 12.0 / 8.0
                    for h in H
                ]) + JulyPart,
                piece_vec + JulyPart
            )
        )
    ).astype(float)

    # ---- NoRetire / RetireCalc (same branches as Months Projection 25) ----
    NoRetire = np.where(
        YearH == BudgetYear, 0.0,
        np.where(
            (YearH < PrevYear) | (YearH == PrevYear),
            BaseMonths + BonusMonths,
            np.where(
                YearH == CurrYear,
                np.where(H > prodate, HireMonths + BonusMonths, BaseMonths + BonusMonths),
                0.0
            )
        )
    ).astype(float)

    cond1 = (YearH <= CurrYear) & (R <= prodate)
    cond2 = (YearH < PrevYear) & (YearR == CurrYear)
    cond3 = (YearH == PrevYear) & (YearR == CurrYear)
    cond4 = (YearH == CurrYear) & (YearR == CurrYear)
    cond5 = (YearH == CurrYear) & (YearR == BudgetYear)
    cond6 = (YearH <= PrevYear) & (YearR == BudgetYear)

    RetireCalc = np.where(
        cond1, 0.0,
        np.where(
            cond2 | cond3, RetireMonths + BonusMonths,
            np.where(
                cond4, np.where(H <= prodate, RetireMonths + BonusMonths, RetireHiring + BonusMonths),
                np.where(
                    cond5, np.where(H <= prodate, BaseMonths + BonusMonths, HireMonths + BonusMonths),
                    np.where(
                        cond6, BaseMonths + BonusMonths,
                        np.nan
                    )
                )
            )
        )
    ).astype(float)

    # ResultMonths = ROUNDUP(IF(YearH = BudgetYear, 0, IF(R present, RetireCalc, NoRetire)), 1)
    raw_months = np.where(YearH == BudgetYear, 0.0, np.where(~R.isna(), RetireCalc, NoRetire)).astype(float)
    ResultMonths = np.array([roundup(x, 1) for x in raw_months], dtype=float)

    # XmasOnlyMonths = BonusMonths - JulyPart
    XmasOnlyMonths = (BonusMonths - JulyPart).astype(float)

    # ---- Cost calculation ----
    BonusRate = 0.04166  # from your sheet (0,04166)
    monthly_cost = df[monthly_cost_col].astype(float)

    TotalCost = np.round(
        ResultMonths * monthly_cost + XmasOnlyMonths * monthly_cost * BonusRate,
        2
    )

    df["FY Gross Salary Projection For 25"] = TotalCost
    return df
def compute_annual_gross_salary_fy_budget_2026(
    df: pd.DataFrame,
    effective_increase_date: pd.Timestamp,      # IncStart == YearDate
    no_increase_cutoff: pd.Timestamp,           # H5
    inc_pct: float,                             # e.g., 0.10
    inc_pct2: float = salary_increase_pct2,                     # 5%
    active_months_col: str = "FY Months Budget 26",
    monthly_cost_col: str = "Monthly Gross Salary (Current)",
    grade_col: str = "Grade",
) -> pd.DataFrame:
    # --- Preconditions ---
    if "Hiring Date" not in df.columns and "Hire Date" in df.columns:
        df = df.rename(columns={"Hire Date": "Hiring Date"})
    for col in ["Hiring Date", "Retire Date", active_months_col, monthly_cost_col]:
        if col not in df.columns:
            st.warning(f"⚠️ Missing '{col}' — cannot compute Annual Gross Salary FY Budget 2026.")
            return df

    # --- Anchors ---
    YearDate = pd.to_datetime(effective_increase_date)
    IncStart = pd.to_datetime(effective_increase_date)
    H5 = pd.to_datetime(no_increase_cutoff)
    one_year_after_H5   = H5 + relativedelta(years=1)
    one_year_after_YrDt = YearDate + relativedelta(years=1)

    CurrYear = YearDate.year
    StartCurrYear = pd.Timestamp(CurrYear, 1, 1)
    AprilDate     = pd.Timestamp(CurrYear, 4, 30)
    DecDate       = pd.Timestamp(CurrYear, 12, 31)
    ChristmasThr  = pd.Timestamp(CurrYear, 5, 1)

    # --- Row data & numerics ---
    H = pd.to_datetime(df["Hiring Date"], errors="coerce")
    R = pd.to_datetime(df["Retire Date"], errors="coerce")

    ActiveMonths = pd.to_numeric(df[active_months_col], errors="coerce").fillna(0.0)
    MonthlyCost  = pd.to_numeric(
        df[monthly_cost_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)

    # Robust Grade coercion: handle '0,1', '0.1', text
    if grade_col in df.columns:
        Grade = pd.to_numeric(
            df[grade_col].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    else:
        Grade = pd.Series(np.nan, index=df.index)

    # Only compute for rows with ActiveMonths > 0
    mask_active = ActiveMonths > 0

    # Helpers
    def yearfrac(start, end):
        if pd.isna(start) or pd.isna(end):
            return np.nan
        return (end - start).days / 365.0  # ACT/365

    def roundup(x, decimals=1):
        if pd.isna(x):
            return np.nan
        f = 10**decimals
        return np.ceil(x * f) / f

    # --- Bonus parts (April/Dec) ---
    EmployedOnApril = (H <= StartCurrYear) & (R.isna() | (R >= AprilDate))
    EmployedOnDec   = (H <= ChristmasThr)  & (R.isna() | (R >= DecDate))

    april_num = [
        yearfrac_30360_us(max(h, StartCurrYear) if not pd.isna(h) else StartCurrYear,
                 min(AprilDate, r) if not pd.isna(r) else AprilDate)
        for h, r in zip(H, R)
    ]
    april_den = yearfrac_30360_us(StartCurrYear, AprilDate)
    AprilPart = np.where(
        EmployedOnApril, 0.5,
        np.where(
            (H < AprilDate),
            [roundup((num / april_den) * 0.5, 1) if (not pd.isna(num) and april_den > 0) else 0.0 for num in april_num],
            0.0
        )
    ).astype(float)

    dec_num = [
        yearfrac_30360_us(max(h, StartCurrYear) if not pd.isna(h) else StartCurrYear,
                 min(DecDate, r) if not pd.isna(r) else DecDate)
        for h, r in zip(H, R)
    ]
    dec_den = [
        yearfrac_30360_us(ChristmasThr, max(DecDate, r) if not pd.isna(r) else DecDate)
        for r in R
    ]
    DecPart = np.where(
        EmployedOnDec, 1.0,
        np.where(
            (H >= StartCurrYear),
            [roundup(((n / d) * 1.0) if (not pd.isna(n) and not pd.isna(d) and d > 0) else 0.0, 1)
             for n, d in zip(dec_num, dec_den)],
            0.0
        )
    ).astype(float)

    # --- Per-row increase rule: if Grade == 0.1 -> inc_pct2 else inc_pct ---
    inc_used = np.where(np.isclose(Grade, 0.1, atol=1e-9), inc_pct2, inc_pct)

    # --- Factors ---
    AprilFactor = 1.0
    dec_factor_cond = (H <= H5) & (R.isna() | (R > one_year_after_H5))
    DecFactor = np.where(dec_factor_cond, 1.0 + inc_used, 1.0)

    # --- BaseCost (split at MONTH(IncStart)-0.5), only for active rows ---
    inc_split = float(IncStart.month) - 0.5
    pre_increase_months  = np.minimum(ActiveMonths, inc_split)
    post_increase_months = np.maximum(ActiveMonths - inc_split, 0.0)

    basecost_condition = (H <= H5) & (R.isna() | (R > one_year_after_YrDt))

    BaseCost_all = np.where(
        basecost_condition,
        pre_increase_months * MonthlyCost + post_increase_months * MonthlyCost * (1.0 + inc_used),
        ActiveMonths * MonthlyCost
    )
    BaseCost_all = np.round(BaseCost_all, 2)

    # --- Allowances ---
    BonusRate = 0.04166
    Allowances_all = np.round(
        MonthlyCost * BonusRate * (AprilPart * AprilFactor + DecPart * DecFactor),
        2
    )

    # Zero out rows with no active months
    Total_all = np.round(BaseCost_all + Allowances_all, 2)
    Total = np.where(mask_active, Total_all, 0.0)

    df["Annual Gross Salary FY Budget 2026"] = Total
    return df




# 1) Upload MAIN file
# ───────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📎 Upload your MAIN Manpower file (Excel or CSV)", type=["xlsx", "xls", "csv"], key="main"
)

# --- Process MAIN file only if it's uploaded ---
if uploaded:
    # Excel options for MAIN
    sheet_name = None
    header_row = 0
    file_name = uploaded.name.lower()
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded)
        with st.sidebar.expander("Excel import options (MAIN)", expanded=False):
            sheet_name = st.selectbox("Select sheet", options=xls.sheet_names, index=0, key="main_sheet")
            header_row = st.number_input(
                "Header row (0-based)", min_value=0, max_value=100, value=0, step=1,
                help="If your data headers start after some top rows, set this accordingly.",
                key="main_header_row",
            )
    uploaded.seek(0)
    df_raw = read_any(uploaded, sheet_name=sheet_name, header_row=header_row)

    # Heuristic renames (optional + SAFE)
    rename_conditions = {
        "DATA ANALYST": "Job Title",
        "ΠΑΠΑΔΟΠΟΥΛ": "Surname",
        "ΓΕΩΡΓΙΟΣ": "Name",
        "DIVISION": "Division",
        "ΑΛΟΥΜΥΛ Α.Ε.": "Company",
        "ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ": "Department"
    }
    for val, new_col in rename_conditions.items():
        for col in df_raw.columns:
            series = df_raw[col]
            if not isinstance(series, pd.Series):
                continue
            try:
                if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                    if series.astype(str).str.contains(val, na=False).any() and new_col not in df_raw.columns:
                        df_raw = df_raw.rename(columns={col: new_col})
                        break
            except Exception:
                pass

    # Explicit renames (Greek → English)
    explicit_rename_map = {
        "Ημ/νία γέννησης": "Date of Birth",
        "Ημ/νία πρόσληψης": "Hire Date",
        "Ημ/νία αποχώρησης": "Retire Date",
        "Κωδικός εργαζόμενου": "Hrms Id",
        "Κωδικός εργαζομένου": "Hrms Id",      # variant
        "Εταιρία": "Company_code",
        "Περιγραφή εταιρίας": "Company",
        "Επώνυμο": "Surname",
        "Ονομα": "Name",
        "Όνομα": "Name",
        "GRADE": "Grade",
    }
    df_raw = df_raw.rename(columns=explicit_rename_map)
    df_raw = drop_dup_named_cols(df_raw)

    df_raw = find_and_rename_column_with_exact_value(df_raw, "ADMINISTRATIVE", "Job Property")
    df = coerce_dates(df_raw.copy())

    # ───────────────────────────────────────────────────────────────────────────────
    # Filtering rules
    # ───────────────────────────────────────────────────────────────────────────────
    # 1) Retire Date
    if "Retire Date" in df.columns:
        df = df[(df["Retire Date"].isna()) | (df["Retire Date"] >= projection_date)]
    else:
        st.warning("⚠️ 'Retire Date' column not found; skipping retire-date filter.")

    # 2) Είναι το κύριο Κ.Κ. = 1
    primary_col = "Είναι το κύριο Κ.Κ."
    if primary_col in df.columns:
        mask_primary = (df[primary_col] == 1) | (df[primary_col] == True)
        try:
            mask_primary = mask_primary | (df[primary_col].astype(str).str.strip() == "1")
        except Exception:
            pass
        df = df[mask_primary]
    else:
        st.warning(f"⚠️ '{primary_col}' column not found; skipping primary cost center filter.")

    # 3) Remove duplicates by Hrms Id
    if "Hrms Id" in df.columns:
        df["Hrms Id"] = df["Hrms Id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["Hrms Id"])
    else:
        st.warning("⚠️ 'Hrms Id' column not found after renaming; cannot drop duplicates on it.")

    # ───────────────────────────────────────────────────────────────────────────────
    # Salary & Cost Center
    # ───────────────────────────────────────────────────────────────────────────────
    if "Ονομαστικός μισθός" in df.columns:
        df = df.rename(columns={"Ονομαστικός μισθός": "Monthly Gross Salary (Current)"})
        df["Monthly Gross Salary (Current)"] = (
            df["Monthly Gross Salary (Current)"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df["Monthly Gross Salary (Current)"] = pd.to_numeric(
            df["Monthly Gross Salary (Current)"], errors="coerce"
        )
        mask = (df["Monthly Gross Salary (Current)"] > 0) & (df["Monthly Gross Salary (Current)"] < 90)
        df.loc[mask, "Monthly Gross Salary (Current)"] *= 26
    else:
        st.warning("⚠️ Column 'Ονομαστικός μισθός' not found for salary conversion.")

    # Cost Center concat
    if "Κέντρο Κόστους" in df.columns and "Περιγραφή Κέντρου Κόστους" in df.columns:
        df["Cost Center"] = (
            df["Κέντρο Κόστους"].astype(str).str.strip()
            + " - "
            + df["Περιγραφή Κέντρου Κόστους"].astype(str).str.strip()
        )
    elif "Κέντρο Κόστους" in df.columns:
        df["Cost Center"] = df["Κέντρο Κόστους"].astype(str).str.strip()
    elif "Περιγραφή Κέντρου Κόστους" in df.columns:
        df["Cost Center"] = df["Περιγραφή Κέντρου Κόστους"].astype(str).str.strip()

    # ───────────────────────────────────────────────────────────────────────────────
    # Override setup (normalized code + initial flag)
    # ───────────────────────────────────────────────────────────────────────────────
    booking_code_col = "Κωδικός Κράτησης"
    override_map = {
        "40602": 0.1879,
        "40603": 0.1879,
        "40380": 0.1879,
        "40084": 0.1879,
        "40510": 0.1738,  # new case
    }
    override_codes = set(override_map.keys())

    if booking_code_col in df.columns:
        df["Κωδικός Κράτησης (norm)"] = normalize_code(df[booking_code_col])
        df["Contrib Override Applied"] = np.where(df["Κωδικός Κράτησης (norm)"].isin(override_codes), "Yes", "No")
    else:
        df["Κωδικός Κράτησης (norm)"] = np.nan
        df["Contrib Override Applied"] = np.nan
        st.warning("⚠️ 'Κωδικός Κράτησης' not found in MAIN file; override flag cannot be computed.")

    # --- STORE THE PROCESSED BASE DF IN SESSION STATE ---
    st.session_state.base_df = df.copy()

# --- Stop if base_df is not (yet) in session state ---
if "base_df" not in st.session_state:
    st.info("Upload the MAIN manpower file to begin. Accepted: .xlsx, .xls, .csv")
    st.stop()

# --- Start every run with the stored base_df ---
df = st.session_state.base_df.copy()

# ───────────────────────────────────────────────────────────────────────────────
# 2) Upload CONTRIBUTIONS file and merge
# ───────────────────────────────────────────────────────────────────────────────
uploaded_contrib = st.file_uploader(
    "📎 Upload the CONTRIBUTIONS file (Excel or CSV)", type=["xlsx", "xls", "csv"], key="contrib"
)

# --- If user *cleared* the file, remove the processed version from state ---
if uploaded_contrib is None and "processed_df_contrib" in st.session_state:
    del st.session_state.processed_df_contrib

# --- If user *just* uploaded a new file, process and store it ---
if uploaded_contrib:
    contrib_sheet = None
    contrib_header_row = 0
    file_name2 = uploaded_contrib.name.lower()
    if file_name2.endswith(".xlsx") or file_name2.endswith(".xls"):
        xls2 = pd.ExcelFile(uploaded_contrib)
        with st.sidebar.expander("Excel import options (CONTRIBUTIONS)", expanded=False):
            contrib_sheet = st.selectbox("Select sheet", options=xls2.sheet_names, index=0, key="contrib_sheet")
            contrib_header_row = st.number_input(
                "Header row (0-based)", min_value=0, max_value=100, value=0, step=1,
                help="If headers start after some rows.", key="contrib_header_row",
            )
    uploaded_contrib.seek(0)
    df_contrib = read_any(uploaded_contrib, sheet_name=contrib_sheet, header_row=contrib_header_row)

    if "Αριθμός μητρώου" in df_contrib.columns and "Hrms Id" not in df_contrib.columns:
        df_contrib = df_contrib.rename(columns={"Αριθμός μητρώου": "Hrms Id"})

    if "Hrms Id" not in df_contrib.columns:
        st.error("The contributions file must contain 'Αριθμός μητρώου' (or 'Hrms Id').")
    else:
        contrib_col = "Contributions"
        if contrib_col not in df_contrib.columns:
            st.error("The contributions file must contain a 'Contributions' column.")
        else:
            df_contrib["Hrms Id"] = df_contrib["Hrms Id"].astype(str).str.strip()
            # Store the *processed* df in session state
            st.session_state.processed_df_contrib = df_contrib.copy()

# --- Now, check if we have a valid contrib file in state and merge it ---
if "processed_df_contrib" in st.session_state:
    df_contrib = st.session_state.processed_df_contrib
    contrib_col = "Contributions" 
    
    df = df.merge(
        df_contrib[["Hrms Id", contrib_col]],
        on="Hrms Id",
        how="left",
    )
    df[contrib_col] = pd.to_numeric(
        df[contrib_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    if "Κωδικός Κράτησης (norm)" in df.columns:
        df["Contributions%"] = df.apply(
            lambda r: override_map.get(r["Κωδικός Κράτησης (norm)"], r.get(contrib_col, np.nan)),
            axis=1
        )
        df["Contrib Override Applied"] = df["Κωδικός Κράτησης (norm)"].apply(
            lambda x: "Yes" if x in override_map else "No"
        )
    else:
        st.warning("⚠️ Normalized booking code column missing; using contributions as-is.")
        df["Contributions%"] = df[contrib_col]

    df = df.drop(columns=[contrib_col], errors="ignore")
else:
    st.info("(Optional) Upload the CONTRIBUTIONS file to add 'Contributions%'. Without it, the column stays empty.")
    if "Contributions%" not in df.columns:
        df["Contributions%"] = np.nan

# --- Compute all derived columns ---
df = compute_employer_contrib(df)
df = compute_months_projection_25(df, projection_date)
df = compute_fy_months_budget_26(df, budget_base_date, payroll_periods) 
df = compute_fy_gross_salary_projection_25(df, projection_date, monthly_cost_col="Monthly Gross Salary (Current)")
df = compute_annual_gross_salary_fy_budget_2026(
    df,
    effective_increase_date=effective_increase_date,
    no_increase_cutoff=no_increase_cutoff,
    inc_pct=salary_increase_pct,
    inc_pct2= salary_increase_pct2,
    active_months_col="FY Months Budget 26",
    monthly_cost_col="Monthly Gross Salary (Current)",
    grade_col="Grade",
)

# --- FY Employer's Contributions Projection 25 ---
contrib_amount_col = None
for cand in ["Monthly Employer's Contributions", "Monthly Employer'S Contributions"]:
    if cand in df.columns:
        contrib_amount_col = cand
        break
if contrib_amount_col is None:
    st.warning("⚠️ Can't compute FY Employer's Contributions Projection 25 (monthly contribution amount not found).")
else:
    if "Months Projection 25" not in df.columns:
        st.warning("⚠️ Can't compute FY Employer's Contributions Projection 25 (Months Projection 25 missing).")
    else:
        amt = pd.to_numeric(df[contrib_amount_col], errors="coerce")
        months = pd.to_numeric(df["Months Projection 25"], errors="coerce")
        df["FY Employer's Contributions Projection 25"] = np.round(amt * months, 2)

# --- Total Payroll Projection Cost 25 ---
gross_col = "FY Gross Salary Projection For 25"
employer_col = "FY Employer's Contributions Projection 25"
if gross_col in df.columns and employer_col in df.columns:
    df["Total Payroll Projection Cost 25"] = np.round(
        pd.to_numeric(df[gross_col], errors="coerce") +
        pd.to_numeric(df[employer_col], errors="coerce"),
        2
    )
else:
    st.warning("⚠️ Missing required columns to compute 'Total Payroll Projection Cost 25'.")

# --- Annual Employer's Contributions For 2026 ---
need_cols = ["Contributions%", "Annual Gross Salary FY Budget 2026", "FY Months Budget 26"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.warning(f"⚠️ Missing columns for 'Annual Employer's Contributions For 2026': {missing}")
else:
    rate   = pd.to_numeric(df["Contributions%"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    annual = pd.to_numeric(df["Annual Gross Salary FY Budget 2026"], errors="coerce").fillna(0.0)
    months = pd.to_numeric(df["FY Months Budget 26"], errors="coerce").fillna(0.0)
    base = annual * rate
    add_30 = np.where(np.isclose(rate, 0.1879, atol=1e-6), 30.0 * months, 0.0)
    add_25 = np.where(np.isclose(rate, 0.1738, atol=1e-6), 25.0 * months, 0.0)
    df["Annual Employer's Contributions For 2026"] = np.round(base + add_30 + add_25, 2)

# --- FY PAYROLL COST BUDGET 2026 ---
gross_col = "Annual Gross Salary FY Budget 2026"
employer_col = "Annual Employer's Contributions For 2026"
if gross_col in df.columns and employer_col in df.columns:
    df["FY PAYROLL COST BUDGET 2026"] = np.round(
        pd.to_numeric(df[gross_col], errors="coerce") +
        pd.to_numeric(df[employer_col], errors="coerce"),
        2
    )
else:
    st.warning(f"⚠️ Missing one of the required columns: '{gross_col}' or '{employer_col}'")

# ───────────────────────────────────────────────────────────────────────────────
# Diagnostics (helps verify override logic)
# ───────────────────────────────────────────────────────────────────────────────
with st.expander("🔎 Diagnostics (first rows with booking code)"):
    diag_cols = [c for c in ["Hrms Id", booking_code_col, "Κωδικός Κράτησης (norm)", "Contrib Override Applied", "Contributions%"] if c in df.columns]
    if diag_cols:
        st.dataframe(df[diag_cols].head(20), use_container_width=True)
    if {"Κωδικός Κράτησης (norm)", "Contrib Override Applied"}.issubset(df.columns):
        bad = df[(df["Κωδικός Κράτησης (norm)"].isin(list(override_codes))) & (df["Contrib Override Applied"] != "Yes")]
        if not bad.empty:
            st.error("Found rows that match override codes but flag is not 'Yes'. Showing first 10:")
            st.dataframe(bad.head(10), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# Final column normalization & ordering
# ───────────────────────────────────────────────────────────────────────────────
if "Hire Date" in df.columns and "Hiring Date" not in df.columns:
    df = df.rename(columns={"Hire Date": "Hiring Date"})

final_order = [
    "Company", "Hrms Id", "Surname", "Name", "Division", "Department",
    "Job Title", "Job Property", "Grade", "Hiring Date", "Retire Date",
    "Cost Center", "Monthly Gross Salary (Current)", "Κωδικός Κράτησης",
    "Κωδικός Κράτησης (norm)", "Contrib Override Applied", "Contributions%",
    "Monthly Employer's Contributions", "Months Projection 25",
    "FY Months Budget 26", "FY Gross Salary Projection For 25",
    "FY Employer's Contributions Projection 25",
    "Total Payroll Projection Cost 25",
    "Annual Gross Salary FY Budget 2026",
    "Annual Employer's Contributions For 2026",
    "FY PAYROLL COST BUDGET 2026",
]
df = drop_dup_named_cols(df)
existing = [c for c in final_order if c in df.columns]
df_final = df[existing].copy() # This is the full, unfiltered final dataset

# ───────────────────────────────────────────────────────────────────────────────
# Filters (no groupby) + Totals for all numeric calculated columns
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("🔎 Filter (Company / Division / Department / Cost Center)")

DIM_COLS = ["Company", "Division", "Department", "Cost Center"]

# --- MODIFIED multiselect_with_all function ---
def multiselect_with_all(label, series, key):
    """Sidebar multiselect. Returns full list if nothing is selected."""
    opts = sorted(
        pd.Series(series, dtype="object").dropna().astype(str).str.strip().unique().tolist()
    )
    # No "All" token. Default is empty list.
    sel = st.sidebar.multiselect(label, opts, default=[], key=key)
    # Return all options if selection is empty, otherwise return the selection
    return opts if len(sel) == 0 else sel

# Build and apply filters
filtered = df_final.copy()

for dim in DIM_COLS:
    if dim in filtered.columns:
        selected = multiselect_with_all(f"Filter by {dim}", filtered[dim], key=f"flt_{dim}")
        # compare as strings to be robust
        filtered = filtered[filtered[dim].astype(str).str.strip().isin([str(x).strip() for x in selected])]

st.caption(f"Filtered rows: {len(filtered):,}")

# Compute totals
numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
totals_series = filtered[numeric_cols].sum(numeric_only=True)
totals_row = {c: "" for c in filtered.columns}
for dim in DIM_COLS:
    if dim in filtered.columns:
        totals_row[dim] = "TOTAL"
        break
for c in numeric_cols:
    totals_row[c] = round(float(totals_series.get(c, 0.0)), 2)

filtered_with_totals = pd.concat(
    [filtered, pd.DataFrame([totals_row])], ignore_index=True
)

st.markdown("### 📄 Filtered Data (with totals)")
st.dataframe(filtered_with_totals, use_container_width=True, height=520)

with st.expander("View totals-only summary"):
    totals_only = pd.DataFrame(totals_series.round(2)).T
    totals_only.index = ["TOTAL"]
    st.table(totals_only)

# Define the download helper function
def _to_xlsx_bytes(df_in: pd.DataFrame, sheet_name="Data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_in.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download Filtered (XLSX)",
        data=_to_xlsx_bytes(filtered, sheet_name="Filtered"),
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c2:
    st.download_button(
        "⬇️ Download Filtered + Totals (XLSX)",
        data=_to_xlsx_bytes(filtered_with_totals, sheet_name="Filtered+Totals"),
        file_name="filtered_with_totals.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ───────────────────────────────────────────────────────────────────────────────
# NEW: Report for New Active Hires
# ───────────────────────────────────────────────────────────────────────────────


report_cols_present = {"Hiring Date", "Retire Date"}.issubset(df_final.columns)
proj_date_present = "projection_date" in locals() or "projection_date" in globals()

if report_cols_present and proj_date_present:
    st.caption(f"Employees hired in {projection_date.year} AND are still active.")
    
    try:
        # Ensure dates are datetime objects for comparison
        hire_date_dt = pd.to_datetime(df_final["Hiring Date"], errors='coerce')
        retire_date_dt = pd.to_datetime(df_final["Retire Date"], errors='coerce')

        # Condition 1: Hired in the projection year
        mask_hire_year = (hire_date_dt.dt.date >= New_Hires)
        
        # Condition 2: Active (no retire date OR retire date is in the future)
        mask_active = (retire_date_dt.isna()) | (retire_date_dt > projection_date)
        
        # --- THIS IS THE MODIFIED LINE ---
        # Apply BOTH conditions using AND (&) instead of OR (|)
        df_special_report = df_final[mask_hire_year & mask_active].copy()
        # --- END MODIFICATION ---
        with st.expander("📊 New Active Hires Report"):
            st.dataframe(df_special_report, use_container_width=True, height=400)
        
        st.download_button(
            "⬇️ Download New Active Hires (XLSX)",
            data=_to_xlsx_bytes(df_special_report, sheet_name="New-Active-Hires"),
            file_name="new_active_hires_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_special_report"
        )
        
    except Exception as e:
        st.error(f"Could not generate new active hires report. Error: {e}")
else:
    st.warning("Cannot generate 'New Active Hires Report'. Missing 'Hiring Date', 'Retire Date', or 'projection_date'.")



# ───────────────────────────────────────────────────────────────────────────────
# PLOTLY VISUAL ANALYTICS DASHBOARD
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Visual Analytics Dashboard")
st.caption("Charts update automatically based on your filters.")

# Check if dataframe is empty after filtering
if filtered.empty:
    st.info("No data to display for the current filter selection.")
else:
    # --- Define required columns for the dashboard
    cost_col = "FY PAYROLL COST BUDGET 2026"
    annual_salary_col = "Annual Gross Salary FY Budget 2026"
    monthly_salary_col = "Monthly Gross Salary (Current)" # NEW COLUMN FOR BOX PLOT
    headcount_col = "Hrms Id"
    
    div_col = "Division"
    dept_col = "Department"
    comp_col = "Company"
    grade_col = "Grade" # NEW COLUMN FOR BOX PLOT
    
    # --- Check that all required columns exist
    required_cols = [cost_col, annual_salary_col, monthly_salary_col, headcount_col, div_col, dept_col, comp_col, grade_col]
    missing_cols = [col for col in required_cols if col not in filtered.columns]
    
    if missing_cols:
        st.warning(f"Dashboard cannot be displayed. Missing required columns: {', '.join(missing_cols)}")
    else:
        # --- Row 1: Dashboard Layout (2 columns)
        col1, col2 = st.columns(2)
        
        # --- Chart 1: Total Cost by Division (Bar Chart) ---
        with col1:
            try:
                div_stats = (
                    filtered.groupby(div_col)
                    .agg(
                        TotalCost=(cost_col, 'sum'),
                        Headcount=(headcount_col, 'nunique')
                    )
                    .reset_index()
                    .sort_values('TotalCost', ascending=False)
                )

                div_melt = div_stats.melt(
                    id_vars=div_col,
                    value_vars=['TotalCost', 'Headcount'],
                    var_name='Metric',
                    value_name='Value'
                )

                fig_bar = px.bar(
                    div_melt,
                    x='Value',
                    y=div_col,
                    color='Metric',
                    barmode='group',
                    orientation='h',
                    text='Value',
                    color_discrete_map={'TotalCost': '#1f77b4', 'Headcount': '#aec7e8'},
                    title=f"Total Budget & Headcount by {div_col}"
                )

                fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Values",
                    yaxis_title=None,
                    legend_title=None
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to create Division Cost chart: {e}")



        # --- Chart 2: Headcount vs. Avg. Salary (Scatter Plot) ---
        with col2:
            try:
                dept_agg = filtered.groupby(dept_col).agg(
                    Headcount=(headcount_col, 'count'),
                    Total_Budget=(cost_col, 'sum'),
                    Avg_Salary=(annual_salary_col, 'mean')
                ).reset_index()
                
                fig_scatter = px.scatter(
                    dept_agg,
                    x="Headcount",
                    y="Avg_Salary",
                    size="Total_Budget",
                    color=dept_agg.index, # Use a categorical color
                    hover_name=dept_col,
                    title=f"Department Analysis (Size = Total Budget)",
                    size_max=60,
                    log_x=True # Use log scale if headcount varies widely
                )
                fig_scatter.update_layout(
                    xaxis_title="Headcount (Log Scale)",
                    yaxis_title="Average Annual Salary"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to create Department Analysis chart: {e}")


        # --- Separator and New Row ---
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        # --- Chart 3 (NEW): Salary Distribution by Grade (Box Plot) ---
        with col3:
            try:
                # Ensure the Grade column is treated as categorical and sorted correctly
                # We sort the unique grades alphabetically or numerically before plotting
                sorted_grades = sorted(filtered[grade_col].unique().astype(str))
                
                fig_box = px.box(
                    filtered,
                    x=grade_col,
                    y=monthly_salary_col,
                    points="all", # Show individual data points as well
                    title='Monthly Salary Distribution by Grade',
                    category_orders={grade_col: sorted_grades},
                    color=grade_col # Color each box uniquely
                )
                fig_box.update_layout(
                    xaxis_title="Employee Grade",
                    yaxis_title="Monthly Gross Salary (Current)",
                    # Rotate labels if too many grades
                    xaxis={'tickangle': 45 if len(sorted_grades) > 10 else 0}
                )
                st.plotly_chart(fig_box, use_container_width=True)
                #  
            except Exception as e:
                st.error(f"Failed to create Grade Salary Distribution chart: {e}")

        # --- Chart 4 (Simple Headcount by Grade - to fill space) ---
        with col4:
            try:
                grade_count = filtered.groupby(grade_col)[headcount_col].count().sort_values(ascending=True).reset_index(name='Headcount')
                fig_count = px.bar(
                    grade_count,
                    x='Headcount',
                    y=grade_col,
                    orientation='h',
                    title=f"Headcount by {grade_col}",
                    text='Headcount',
                    color='Headcount',
                    color_continuous_scale='Mint'
                )
                fig_count.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_title="Total Number of Employees",
                    yaxis_title=None
                )
                st.plotly_chart(fig_count, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to create Headcount by Grade chart: {e}")

        # --- Chart 5: Treemap Breakdown (Full Width) ---
        try:
            st.markdown("---")

            # 1) Aggregate for each level
            agg_leaf = (
                filtered.groupby([comp_col, div_col, dept_col])
                .agg(TotalCost=(cost_col, "sum"),
                    Headcount_Count=(headcount_col, "nunique"))
                .reset_index()
            )

            agg_div = (
                filtered.groupby([comp_col, div_col])
                .agg(TotalCost=(cost_col, "sum"),
                    Headcount_Count=(headcount_col, "nunique"))
                .reset_index()
            )

            agg_comp = (
                filtered.groupby([comp_col])
                .agg(TotalCost=(cost_col, "sum"),
                    Headcount_Count=(headcount_col, "nunique"))
                .reset_index()
            )

            # 2) Build node tables for each level (ids/parents/labels)
            root_label = "Total Budget"
            root_df = pd.DataFrame({
                "id": [root_label],
                "parent": [""],
                "label": [root_label],
                "TotalCost": [agg_comp["TotalCost"].sum()],
                "Headcount_Count": [agg_comp["Headcount_Count"].sum()],
                "level": ["Root"],
            })

            comp_df = agg_comp.assign(
                id=lambda d: d[comp_col],
                parent=root_label,
                label=lambda d: d[comp_col],
                level="Company"
            )[["id", "parent", "label", "TotalCost", "Headcount_Count", "level"]]

            div_df = agg_div.assign(
                id=lambda d: d[comp_col] + " | " + d[div_col],
                parent=lambda d: d[comp_col],
                label=lambda d: d[div_col],
                level="Division"
            )[["id", "parent", "label", "TotalCost", "Headcount_Count", "level"]]

            dept_df = agg_leaf.assign(
                id=lambda d: d[comp_col] + " | " + d[div_col] + " | " + d[dept_col],
                parent=lambda d: d[comp_col] + " | " + d[div_col],
                label=lambda d: d[dept_col],
                level="Department"
            )[["id", "parent", "label", "TotalCost", "Headcount_Count", "level"]]

            # 3) Concatenate all nodes
            nodes = pd.concat([root_df, comp_df, div_df, dept_df], ignore_index=True)

            # 4) Build treemap with ids/parents and custom_data
            fig_tree = px.treemap(
                nodes,
                names="label",
                parents="parent",
                ids="id",
                values="TotalCost",
                title="Budget Cost Breakdown by Hierarchy (Size = Cost)",
                # hover_data won't aggregate for parents reliably; use custom_data instead:
                custom_data=["Headcount_Count", "TotalCost", "level"]
            )

            fig_tree.update_traces(
                branchvalues="total",
                textinfo="label+value+percent parent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Level: %{customdata[2]}<br>"
                    "Total Cost: %{customdata[1]:,.0f} €<br>"
                    "Headcount: %{customdata[0]:.0f}<br>"
                    "Parent Share: %{percentParent:.2%}<extra></extra>"
                )
            )

            st.plotly_chart(fig_tree, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to create Treemap: {e}")

        st.markdown("---")
