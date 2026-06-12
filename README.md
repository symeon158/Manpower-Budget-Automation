# Manpower Budget Automation

Internal Streamlit application that automates the annual manpower budget process at **Alumil S.A.** Reads and writes live to SharePoint via the Microsoft Graph API, computes payroll projections under Greek EFKA rules, and supports side-by-side scenario comparison.

---

## Quick start

```bash
# 1. Configure credentials
cp .env.example .env
# Fill in SP_TENANT_ID, SP_CLIENT_ID, SP_CLIENT_SECRET, SP_FILE_URL

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## What it does

- **Tab 1 — Budget & Analytics:** FY projection + budget calculation with A vs B scenario comparison, KPI deltas, filterable employee table, 8 Plotly dashboards, XLSX export
- **Tab 2 — Data Quality:** flags missing rates, unmatched `Κωδικός Κράτησης` codes, zero/missing salaries, salary outliers, lists employees getting the under-25 discount
- **Tab 3 — New Hiring Request:** cascading form (Division → Department → Job → Pay Band) that writes structured rows directly to the `New_Hirings` sheet in SharePoint

## Payroll rules implemented

- **13-code contribution rate lookup** keyed on `Κωδικός Κράτησης`, with €25/€30 fixed monthly add-ons
- **Under-25 employer-contribution discount** (Ν.4583/2018 — 6.66pp reduction on main pension)
- **30/360 calendar** with Easter / Summer / Christmas bonus rules, Christmas period capped at 1.0
- **Grade-aware salary increases** with configurable cutoff & effective dates
- **Cost columns gated** on active months: zero when employee is inactive in that period

## Project layout

```
app.py                  Thin Streamlit orchestrator
mb/
├── config.py           Constants + Graph credentials loader
├── scenarios.py        Pydantic Scenario + run_scenario()
├── sharepoint.py       Graph auth + batch row writer
├── data_loader.py      Excel column normalisation
├── pay_ranges.py       Pay-band table
├── export.py           XLSX export with formatting
├── calc/               date_helpers, contrib, projections, budget
└── ui/                 header, sidebar, three tabs

tests/                  106 unit tests (pytest)
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest                    # 106 tests, ~3s
pytest -k "youth"         # filter by keyword
```

## Configuration

| Environment variable | Purpose |
|---|---|
| `SP_TENANT_ID` | Azure AD tenant ID |
| `SP_CLIENT_ID` | App registration client ID |
| `SP_CLIENT_SECRET` | App registration client secret |
| `SP_FILE_URL` | SharePoint URL of `ManpowerBudget.xlsx` |

The Azure AD app needs **Files.ReadWrite** (Application) permission on the target OneDrive.

To change contribution rates, edit `CONTRIB_RATE_MAP` in `mb/config.py`. To change pay bands, edit `PAY_RANGES_ROWS` in `mb/pay_ranges.py`. Both are covered by unit tests that will catch malformed updates.

## Deployment

Built as a Docker image and deployed on **Azure Container Apps**. Environment variables are injected via Container App secrets — never baked into the image.

## Stack

Python · Streamlit · pandas · NumPy · Plotly · Pydantic · Microsoft Graph API · Azure AD · Docker · Azure Container Apps · pytest

## License

Proprietary — Alumil S.A. Not licensed for redistribution.
