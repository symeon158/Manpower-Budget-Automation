"""Raw-data ingestion and column normalisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mb.config import DATE_CANDIDATES


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    if "Ημ/νία πρόσληψης" in df.columns and "Hire Year" not in df.columns:
        df["Hire Year"] = df["Ημ/νία πρόσληψης"].dt.year
    if "Ημ/νία αποχώρησης" in df.columns and "Departure Year" not in df.columns:
        df["Departure Year"] = df["Ημ/νία αποχώρησης"].dt.year
    if "Hire Date" in df.columns and "Hire Year" not in df.columns:
        df["Hire Year"] = df["Hire Date"].dt.year
    if "Retire Date" in df.columns and "Departure Year" not in df.columns:
        df["Departure Year"] = df["Retire Date"].dt.year
    return df


def find_and_rename_col(df: pd.DataFrame, value: str, new_name: str) -> pd.DataFrame:
    target = str(value).strip()
    for col in df.columns:
        s = df[col]
        if not isinstance(s, pd.Series):
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                if (s.dropna().astype(str).str.strip().eq(target).any()
                        and new_name not in df.columns):
                    return df.rename(columns={col: new_name})
            except Exception:
                pass
    return df


def drop_dup_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()


def normalise_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all value-based renames and explicit-column renames.
    """
    rename_conditions = {
        "DATA ANALYST":    "Job Title",
        "ΠΑΠΑΔΟΠΟΥΛ":       "Surname",
        "ΓΕΩΡΓΙΟΣ":         "Name",
        "DIVISION":        "Division",
        "ΑΛΟΥΜΥΛ Α.Ε.":    "Company",
        "ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ":  "Department",
    }
    for val, new_col in rename_conditions.items():
        for col in df_raw.columns:
            s = df_raw[col]
            if not isinstance(s, pd.Series):
                continue
            try:
                if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                    if (s.astype(str).str.contains(val, na=False).any()
                            and new_col not in df_raw.columns):
                        df_raw = df_raw.rename(columns={col: new_col})
                        break
            except Exception:
                pass

    explicit_rename = {
        "Ημ/νία γέννησης":       "Date of Birth",
        "Ημ/νία πρόσληψης":      "Hire Date",
        "Ημ/νία αποχώρησης":     "Retire Date",
        "Κωδικός εργαζόμενου":   "Hrms Id",
        "Κωδικός εργαζομένου":   "Hrms Id",
        "Εταιρία":                "Company_code",
        "Περιγραφή εταιρίας":    "Company",
        "Επώνυμο":               "Surname",
        "Ονομα":                 "Name",
        "Όνομα":                 "Name",
        "GRADE":                 "Grade",
    }
    df_raw = df_raw.rename(columns=explicit_rename)
    df_raw = drop_dup_cols(df_raw)
    df_raw = find_and_rename_col(df_raw, "ADMINISTRATIVE", "Job Property")
    return df_raw


def apply_standard_filters(df: pd.DataFrame, projection_date: pd.Timestamp) -> pd.DataFrame:
    """Remove retired employees, keep only primary Κ.Κ., drop duplicates."""
    if "Retire Date" in df.columns:
        df = df[(df["Retire Date"].isna()) | (df["Retire Date"] >= projection_date)]
    prim = "Είναι το κύριο Κ.Κ."
    if prim in df.columns:
        mp = (df[prim] == 1) | (df[prim] == True)
        try:
            mp = mp | (df[prim].astype(str).str.strip() == "1")
        except Exception:
            pass
        df = df[mp]
    if "Hrms Id" in df.columns:
        df["Hrms Id"] = df["Hrms Id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["Hrms Id"])
    return df


def normalise_salary(df: pd.DataFrame) -> pd.DataFrame:
    if "Ονομαστικός μισθός" in df.columns:
        df = df.rename(columns={"Ονομαστικός μισθός": "Monthly Gross Salary (Current)"})
    if "Monthly Gross Salary (Current)" in df.columns:
        df["Monthly Gross Salary (Current)"] = pd.to_numeric(
            df["Monthly Gross Salary (Current)"].astype(str)
              .str.replace(",", ".", regex=False).str.strip(),
            errors="coerce")
        mask = ((df["Monthly Gross Salary (Current)"] > 0) &
                (df["Monthly Gross Salary (Current)"] < 90))
        df.loc[mask, "Monthly Gross Salary (Current)"] *= 26
    return df


def build_cost_center(df: pd.DataFrame) -> pd.DataFrame:
    if "Κέντρο Κόστους" in df.columns and "Περιγραφή Κέντρου Κόστους" in df.columns:
        df["Cost Center"] = (df["Κέντρο Κόστους"].astype(str).str.strip() + " - " +
                              df["Περιγραφή Κέντρου Κόστους"].astype(str).str.strip())
    elif "Κέντρο Κόστους" in df.columns:
        df["Cost Center"] = df["Κέντρο Κόστους"].astype(str).str.strip()
    elif "Περιγραφή Κέντρου Κόστους" in df.columns:
        df["Cost Center"] = df["Περιγραφή Κέντρου Κόστους"].astype(str).str.strip()
    return df
