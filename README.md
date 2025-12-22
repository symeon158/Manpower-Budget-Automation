# 📊 Manpower Budget Automation for **Alumil S.A. & Subsidiaries**

A Streamlit app that turns raw HRMS exports into **auditable**, **parameter-driven** payroll projections and visual analytics — no macros, no manual formulas. Upload your files, set business parameters, and download clean, ready-to-share Excel outputs.

> 🚀 **Live Demo:** [View on Streamlit Cloud](https://esgautomation-6lucvjswyrkv3q5eadl9op.streamlit.app/)  
> ✅ Deployed and maintained by **Symeon Papadopoulos**

> 🐍 **Stack:** Streamlit · Pandas · NumPy · Plotly Express · XlsxWriter · python-dateutil

---

## ✨ Features

- **Robust file intake (Excel/CSV)**
  - Auto-detects encodings (`iso-8859-7`, `utf-8`) and delimiters (`;` / `,`)
  - Pick sheet & header row, safe coercion for **Greek/English date columns**
- **Smart column handling**
  - Greek ➜ English renames (e.g., `Ημ/νία πρόσληψης` → `Hire Date`)
  - Duplicate-name guardrails and heuristic renames by content
- **Booking code normalization & overrides**
  - Extracts the first 5-digit code from any messy text
  - Contribution overrides:  
    - `40602, 40603, 40380, 40084` → **18.79% + €30** / month  
    - `40510` → **17.38% + €25** / month
- **Excel-grade projection engines (in Python)**
  - `Months Projection 25` (30/360 US, July piecewise logic, bonus parts)
  - `FY Months Budget 26` (April/July/December accruals; full/partial periods)
  - `FY Gross Salary Projection For 25` (months + Xmas share)
  - `Annual Gross Salary FY Budget 2026`
    - Split pre/post **Effective Increase Date**
    - Apply **Average Salary Increase %** and **% for Grade 0.1**
    - Allowances on April/Dec rules
  - `Annual Employer's Contributions For 2026` and **Full FY Payroll Cost 2026**
- **Filters + totals**
  - Interactive **Company / Division / Department / Cost Center** filters
  - Automatic **totals row**; export as **XLSX** (filtered / filtered + totals)
- **New Active Hires report**
  - “Hired after threshold & still active at projection date” — with download
- **Visual Analytics Dashboard (auto-filters)**
  - Grouped **Bar:** Total Budget & Headcount by Division  
  - **Scatter (bubble):** Dept Headcount vs Avg Salary (size = Total Budget)  
  - **Box plot:** Monthly Salary distribution by Grade  
  - **Bar:** Headcount by Grade  
  - **Treemap:** Company → Division → Department (size = Cost; hover = Headcount)

   <img width="296" height="410" alt="image" src="https://github.com/user-attachments/assets/a2e0597b-292c-4553-b075-b8283167040f" />

---

## 🧠 Sidebar Parameters

- **Projection Date (for 2025)**
- **No Increases after** (cutoff for 2026)
- **Effective Date of Salary Increases (2026)**
- **Payroll Periods per Year** (default **14**)
- **Average Salary Increase %**
- **Average Salary Increase % for Grade 0.1**
- **New Hires threshold** (date)

<img width="323" height="756" alt="image" src="https://github.com/user-attachments/assets/bcb68b87-db60-48d9-9632-7a3ad2ac3a17" />


---

## 🗂️ Input Files

1. **MAIN manpower file** (Excel/CSV)  
   Should include (directly or via renames/heuristics):  
   `Hrms Id`, `Company`, `Division`, `Department`, `Job Title`, `Job Property`,  
   `Hire Date/Hiring Date`, `Retire Date`, `Ονομαστικός μισθός`, `Κωδικός Κράτησης`,  
   `Κέντρο Κόστους`, `Περιγραφή Κέντρου Κόστους`, `Είναι το κύριο Κ.Κ.`, (optional) `Grade`.

2. **CONTRIBUTIONS file** *(optional but recommended)*  
   - Columns: `Αριθμός μητρώου` (or `Hrms Id`) and `Contributions` (as %).  
   - Merged by `Hrms Id`; **override map** applied by normalized booking code.
  
  <img width="292" height="323" alt="image" src="https://github.com/user-attachments/assets/a3259f5d-e879-48ed-b7fe-3266f16777f6" />


---

## 📦 Key Derived Columns

- `Monthly Employer's Contributions` (with €25/€30 adders per override)  
- `Months Projection 25`  
- `FY Months Budget 26`  
- `FY Gross Salary Projection For 25`  
- `FY Employer's Contributions Projection 25`  
- `Total Payroll Projection Cost 25`  
- `Annual Gross Salary FY Budget 2026`  
- `Annual Employer's Contributions For 2026`  
- `FY PAYROLL COST BUDGET 2026`

---

## ⚙️ Under the Hood

- **Date math:** 30/360 US (`yearfrac_30360_us`) for bonus/period logic; `EOMONTH` helper  
- **Safe numerics:** Greek decimals normalized (`,` → `.`), coercion with `errors="coerce"`  
- **Deterministic merges:** Contributions merged once; overrides re-applied consistently  
- **Session persistence:** `st.session_state` holds processed base DataFrame  
- **Cache reset:** One click to clear `st.cache_data` if file structure changes

---

