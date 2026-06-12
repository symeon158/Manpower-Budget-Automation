"""Known-answer test employees.

Each fixture represents a scenario whose expected output was computed by hand
(or cross-verified against the original code on 2026-04-22) and pinned here.
If any of these values change, that's a regression that requires explicit
review of the underlying formula.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def build_sample_df() -> pd.DataFrame:
    """
    Five employees covering the main payroll scenarios:

    E001  Long-tenure, mid-grade, standard rate (0.2179)
    E002  Under-25 (born 2002-06-15) — should get 6.66pp discount
    E003  Retires mid-budget-year  (2026-08-31)
    E004  Hired mid-current-year   (2025-05-10)
    E005  Rate 0.1879 (fixed +€30 add-on)
    """
    return pd.DataFrame([
        # Common fields
        {"Hrms Id": "E001", "Surname": "A", "Name": "Alpha",
         "Division": "Manufacturing", "Department": "Line 1",
         "Job Title": "Operator", "Job Property": "BLUE-COLLAR",
         "Grade": 8, "Date of Birth": pd.Timestamp(1985, 3, 10),
         "Hiring Date":  pd.Timestamp(2010, 1, 15),
         "Retire Date":  pd.NaT,
         "Monthly Gross Salary (Current)": 1500.00,
         "Κωδικός Κράτησης": "40050"},         # 0.2179

        {"Hrms Id": "E002", "Surname": "B", "Name": "Beta",
         "Division": "Manufacturing", "Department": "Line 2",
         "Job Title": "Operator", "Job Property": "BLUE-COLLAR",
         "Grade": 6, "Date of Birth": pd.Timestamp(2002, 6, 15),  # age 23 on 2025-10-01
         "Hiring Date":  pd.Timestamp(2022, 9, 1),
         "Retire Date":  pd.NaT,
         "Monthly Gross Salary (Current)": 1000.00,
         "Κωδικός Κράτησης": "40050"},          # 0.2179

        {"Hrms Id": "E003", "Surname": "C", "Name": "Gamma",
         "Division": "Admin", "Department": "HR",
         "Job Title": "Specialist", "Job Property": "ADMINISTRATIVE",
         "Grade": 12, "Date of Birth": pd.Timestamp(1958, 7, 20),
         "Hiring Date":  pd.Timestamp(1990, 4, 3),
         "Retire Date":  pd.Timestamp(2026, 8, 31),
         "Monthly Gross Salary (Current)": 2500.00,
         "Κωδικός Κράτησης": "40050"},          # 0.2179

        {"Hrms Id": "E004", "Surname": "D", "Name": "Delta",
         "Division": "IT", "Department": "Dev",
         "Job Title": "Developer", "Job Property": "ADMINISTRATIVE",
         "Grade": 14, "Date of Birth": pd.Timestamp(1993, 11, 5),
         "Hiring Date":  pd.Timestamp(2025, 5, 10),
         "Retire Date":  pd.NaT,
         "Monthly Gross Salary (Current)": 2200.00,
         "Κωδικός Κράτησης": "40050"},          # 0.2179

        {"Hrms Id": "E005", "Surname": "E", "Name": "Epsilon",
         "Division": "Admin", "Department": "Finance",
         "Job Title": "Analyst", "Job Property": "ADMINISTRATIVE",
         "Grade": 10, "Date of Birth": pd.Timestamp(1980, 2, 1),
         "Hiring Date":  pd.Timestamp(2015, 6, 1),
         "Retire Date":  pd.NaT,
         "Monthly Gross Salary (Current)": 1800.00,
         "Κωδικός Κράτησης": "40602"},          # 0.1879 + €30
    ])


DEFAULT_PROJECTION_DATE = pd.Timestamp(2025, 10, 1)
DEFAULT_BUDGET_YEAR     = 2026
DEFAULT_BUDGET_BASE     = pd.Timestamp(2026, 1, 1)
