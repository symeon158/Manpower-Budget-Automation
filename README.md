# Manpower Budget Automation

An internal Streamlit application that automates the annual manpower budget process for **Alumil S.A.** — a manufacturing group of 3,000+ employees across multiple divisions. Replaces a manual Excel-based workflow with an interactive analytics platform that reads and writes live to SharePoint via the Microsoft Graph API.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![Pydantic](https://img.shields.io/badge/pydantic-2.5+-green)
![Tests](https://img.shields.io/badge/tests-106%20passing-brightgreen)
![License](https://img.shields.io/badge/license-Proprietary-lightgrey)

> **Note:** This is an internal tool. The repository is published for portfolio/reference purposes only — no real employee data is included.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Greek Payroll Rules Implemented](#greek-payroll-rules-implemented)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Roadmap](#roadmap)

---

## Overview

The annual manpower budget at Alumil used to involve multiple analysts manually transforming a master Excel sheet across two months, with a high error rate and no audit trail. This tool replaces that workflow:

| Before | After |
|---|---|
| Manual Excel transformations | Live SharePoint read/write via Microsoft Graph |
| Single-file, single-user | Multi-user with concurrent writes (no file lock) |
| Hard-coded contribution rules | Configurable, scenario-aware payroll engine |
| No audit trail | Every hiring request gets a `Request ID` + timestamp |
| One forecast at a time | Side-by-side A vs B scenario comparison |
| No data validation | Built-in Data Quality report |
| 8 charts in Excel, manually recreated each year | 8 interactive Plotly dashboards, auto-refreshed |

---

## Features

### 📊 Budget & Analytics Tab
- Real-time projection of FY gross salary, employer contributions, total payroll cost
- **A vs B scenario comparison** with per-scenario overrides for every parameter (rates, increase %, dates, youth-discount settings)
- KPI row with deltas between scenarios (€ and %)
- 8 interactive Plotly visualizations:
  - Division Budget (single-scenario or grouped A vs B)
  - Headcount by Division
  - Cost Waterfall (Salary → Contrib → Training → Meal → Total)
  - Salary Spread by Department (box-and-whisker with z-score coloring)
  - Salary Distribution by Grade
  - Salary Heatmap (Grade × Division)
  - Department Scatter (HC × Avg Salary × Total Budget bubble)
  - Budget Hierarchy Treemap (Company → Division → Department)
  - Per-Employee Δ Analysis (when comparing scenarios)
- Filterable on Company / Division / Department / Job Property / Cost Center
- Per-scenario XLSX export + multi-sheet comparison workbook

### 🔍 Data Quality Tab
- Built-in contribution rate lookup table (visible reference)
- Flags employees with: missing Cost Center, unmatched `Κωδικός Κράτησης` code, zero/missing salary, salary outliers (>5× median), missing hiring date
- Under-25 discount audit list

### ➕ New Hiring Request Tab
- Cascading dropdowns: Division → Department → Job Description / Job Property
- Company **pay-band lookup**: Reference Level → Pay Zone → Location → Median salary (P50)
  - Special case: Reference Level `< 8` uses a fixed €1,300 median
- Quantity field (1–50): writes N rows to SharePoint in a single batch
- Force-fresh preview of existing `New_Hirings` records

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit App (Azure Container App)         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  UI Layer (mb/ui/)                                     │  │
│  │  ├── header.py, sidebar.py                             │  │
│  │  ├── tab_budget.py, tab_quality.py, tab_hiring.py      │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Business Logic (mb/)                                  │  │
│  │  ├── config.py           — constants + creds dataclass │  │
│  │  ├── scenarios.py        — Pydantic Scenario + runner  │  │
│  │  ├── calc/                                             │  │
│  │  │   ├── date_helpers.py — 30/360 calendar             │  │
│  │  │   ├── contrib.py      — Employer contributions      │  │
│  │  │   ├── projections.py  — Current-year forecast       │  │
│  │  │   └── budget.py       — Budget-year forecast        │  │
│  │  ├── data_loader.py      — Column normalisation        │  │
│  │  ├── pay_ranges.py       — Salary band table           │  │
│  │  ├── export.py           — Formatted XLSX export       │  │
│  │  └── sharepoint.py       — Graph API client            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            │
                            │  Microsoft Graph API (OAuth 2.0)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│           SharePoint / OneDrive — ManpowerBudget.xlsx        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │  Master sheet    │  │  New_Hirings     │  │  Read-only │  │
│  │  (employees)     │  │  (writes from    │  │  reference │  │
│  │                  │  │   hire form)     │  │  data      │  │
│  └──────────────────┘  └──────────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Key architectural choices:**
- **Read path** downloads the workbook bytes once per cache TTL and parses with pandas in-memory
- **Write path** uses the Graph **Excel REST API** (`/workbook/worksheets/{name}/range(...)`) — single PATCH per submission, no file lock, works while users have the workbook open in Excel Online
- All calculation functions are **pure** (deterministic input → deterministic output) → fully unit-testable
- The Streamlit UI is a thin orchestrator; you can swap in a different frontend without touching `mb/`

---

## Greek Payroll Rules Implemented

Each rule is unit-tested. References to legislation in code comments where applicable.

### Contribution rates
13-code `Κωδικός Κράτησης → rate` lookup with two fixed monthly add-ons:
- Code `40510` → 17.38 % + €25/month
- Codes `40084 / 40380 / 40602 / 40603` → 18.79 % + €30/month
- Codes `40010 / 40011 / 40050 / 40060 / 40061 / 40070 / 40090 / 40012` → various

### Under-25 discount (N.4583/2018, e-ΕΦΚΑ Circular 28/2019)
- Employees whose age on the projection date is strictly under 25 receive a **6.66 pp reduction** on the employer's main pension contribution
- Fixed add-ons (€25 / €30) are **not** reduced
- Threshold and discount % are configurable in the sidebar

### Calendar arithmetic
- **30/360 day-counting** (Excel `YEARFRAC` basis 0) for all year-fraction calculations
- Easter / Summer (July) / Christmas (December) bonus periods with pro-rata rules for mid-year hires and retirees
- Christmas gift period (DP) hard-capped at 1.0 to prevent calculation artefacts in edge cases

### Annual budget
- Grade-aware salary increase: separate % for Grade 0.1 vs the rest
- Pre/post-cutoff split based on `Effective Increase Date` and `No Increases After` parameters
- Annual training cost by grade band (€25 / €150 / €250 / €450 / €500)
- Meal allowance by `ΚΑΡΤΑ ΣΙΤΙΣΗΣ` flag (€1,488 / €744)

### Gating
- Projection-cost columns → 0 when `Months Projection ≤ 0`
- Budget-cost columns → 0 when `FY Months Budget ≤ 0`

---

## Project Structure

```
manpower_budget/
├── app.py                          # Thin Streamlit orchestrator (~140 lines)
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # + pytest
├── pytest.ini                      # Test discovery config
├── .env.example                    # Template for credentials
│
├── mb/
│   ├── __init__.py
│   ├── config.py                   # Constants + credentials loader
│   ├── data_loader.py              # Excel column normalisation + filters
│   ├── sharepoint.py               # Graph auth, batch writer
│   ├── scenarios.py                # Pydantic Scenario + run_scenario
│   ├── pay_ranges.py               # Salary band table + cascade
│   ├── export.py                   # XLSX export with formatting
│   ├── calc/
│   │   ├── date_helpers.py         # 30/360 calendar arithmetic
│   │   ├── contrib.py              # Employer contributions + youth discount
│   │   ├── projections.py          # Current-year projection
│   │   └── budget.py               # Budget-year calculation
│   └── ui/
│       ├── header.py               # Branded header HTML
│       ├── sidebar.py              # Sidebar + per-scenario panels
│       ├── tab_budget.py           # Tab 1
│       ├── tab_quality.py          # Tab 2
│       └── tab_hiring.py           # Tab 3
│
└── tests/
    ├── fixtures.py                 # Known-answer employees
    ├── test_contrib.py             # 23 tests
    ├── test_date_helpers.py        # 16 tests
    ├── test_data_loader.py         # 13 tests
    ├── test_projections.py         # 11 tests
    ├── test_pay_ranges.py          # 12 tests
    └── test_scenarios.py           # 31 tests
```

**Total: 106 tests, ~3 seconds to run.**

---

## Getting Started

### Prerequisites
- Python 3.11+
- An Azure AD app registration with **Files.ReadWrite** (Application) permission on the target user's OneDrive
- A SharePoint URL to the `ManpowerBudget.xlsx` master file

### Local development

```bash
# 1. Clone and enter
git clone <repo-url>
cd manpower_budget

# 2. Set up credentials
cp .env.example .env
# Then edit .env with your SP_TENANT_ID, SP_CLIENT_ID, SP_CLIENT_SECRET, SP_FILE_URL

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest                          # All 106 tests
pytest -v                       # Verbose
pytest tests/test_contrib.py    # Just one file
pytest -k "youth"               # Tests matching a keyword
```

---

## Configuration

### Environment variables (required)

| Variable | Purpose |
|---|---|
| `SP_TENANT_ID` | Azure AD tenant ID |
| `SP_CLIENT_ID` | App registration client ID |
| `SP_CLIENT_SECRET` | App registration client secret |
| `SP_FILE_URL` | SharePoint URL of the master Excel file |

### Contribution rates

Defined as a Python dict in `mb/config.py::CONTRIB_RATE_MAP`. Add or change a code in one place — the tests verify the data still loads and looks up correctly.

```python
CONTRIB_RATE_MAP: dict[str, float] = {
    "40010": 0.2494,
    "40011": 0.2494,
    # ...
    "40602": 0.1879,   # + €30 add-on
    "40603": 0.1879,   # + €30 add-on
}
```

### Pay ranges

Defined in `mb/pay_ranges.py::PAY_RANGES_ROWS`. Each row maps `(RefLevel, PayZone, Location)` → percentiles. The synthetic level `"< 8"` uses a fixed €1,300 median (`BELOW_8_MEDIAN`).

---

## Testing

The test suite is the safety net for the most brittle part of the codebase — Greek payroll formulas with multiple edge cases.

**Coverage by area:**
- **Date helpers** — `yf_vec`, `yf_scalar`, `eomonth`, day-31 adjustments, leap years
- **Contributions** — rate lookup, rate normalisation from messy codes, youth-discount eligibility (active / inactive / custom threshold), fixed add-on preservation under discount
- **Projections** — months-projection for long-tenure / mid-year-retiree / already-retired
- **Budget** — `FY Months Budget` capped at payroll periods, salary increase impact, grade-aware multiplier
- **Pay ranges** — cascade options, BELOW_8 fixed-median behaviour, exec rows with P25=P50=P75
- **Scenarios** — pydantic validation rejects bad inputs, signature stability, identical scenarios produce zero delta, A vs B mathematical consistency

Example known-answer assertion:
```python
def test_30_euro_addon():
    df = build_sample_df()
    df = apply_rate_lookup(df)
    df = compute_employer_contrib(df, apply_youth_discount=False)
    e5 = df[df["Hrms Id"] == "E005"].iloc[0]
    # Rate 0.1879, salary 1800 → 1800 × 0.1879 + 30 = 338.22 + 30 = 368.22
    assert e5["Monthly Employer's Contributions"] == pytest.approx(368.22, abs=0.01)
```

---

## Deployment

### Azure Container Apps

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build & push
az acr build --registry <registry> --image manpower-budget:latest .

# Update container app
az containerapp update \
  --name manpower-budget \
  --resource-group <rg> \
  --image <registry>.azurecr.io/manpower-budget:latest
```

Environment variables (`SP_TENANT_ID`, `SP_CLIENT_ID`, `SP_CLIENT_SECRET`, `SP_FILE_URL`) are injected via Container App secrets — never baked into the image.

---

## Key Engineering Decisions

### Why Streamlit?
- Single-developer maintainable
- ~100 lines of Python ≈ a working dashboard
- Trade-off accepted: full script rerun per widget change (mitigated with `@st.cache_data` on heavy compute)

### Why a Pydantic `Scenario` object?
Replaces a sprawling set of sidebar variables with one validated object. Catches negative salary increases, age thresholds out of range, rate values > 1.0, and date-ordering issues (`no_increase_cutoff >= effective_increase_date`) at construction time, not deep inside a formula.

### Why batch writes via Graph Excel REST?
Original approach: download workbook → modify → upload. Required a file lock and broke when users had the workbook open in Excel Online. New approach: `PATCH /workbook/worksheets/{name}/range(address='A2:K6')` writes N rows in one call, locks nothing, works during user edits.

### Why hash_funcs on Scenario?
`st.cache_data` can't hash a Pydantic model containing `pd.Timestamp` and a dict. Using `hash_funcs={Scenario: lambda s: s.signature()}` lets the cache key on a stable MD5 of all fields, so a parameter change in either A or B reliably triggers a real recompute.

### Why modular over single-file?
The original prototype was a 2,200-line `app.py`. After splitting into the `mb/` package, adding the test suite (106 tests) became feasible, and adding new features (scenario comparison, pay ranges, retirement chart) became a matter of editing one focused file rather than scrolling through everything.

---

## Roadmap

Items planned but not yet implemented:

- [ ] **EFKA monthly contribution cap (€7,572.62 → €7,761.94 for 2026)** for salaries above the ceiling
- [ ] **Month-by-month youth eligibility** (currently evaluated at projection date only — fine for most cases, off by 0–1 month for employees turning 25 mid-year)
- [ ] **Per-user audit trail** on hiring submissions via Azure App Service Authentication
- [ ] **PDF executive summary export** (currently XLSX only)
- [ ] **Plotly chart caching** to eliminate full redraw on each filter change

---

## License

Proprietary — Alumil S.A. All rights reserved. Not licensed for redistribution or commercial use.

---

## Acknowledgements

Built as an internal HR/Finance tool. Greek payroll calendar logic informed by the [e-EFKA documentation](https://www.efka.gov.gr/) and the Ministerial decisions referenced inline in source code.
