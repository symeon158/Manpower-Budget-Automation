# 📊 Manpower Budget Automation for Subsidiaries

A full-featured **Streamlit application** designed to automate **manpower budgeting, payroll cost forecasting, and contribution calculations** across multiple subsidiaries.  
This app integrates dynamic Excel-like logic (translated from advanced `LET()` formulas) with modern Python data processing, making HR and financial reporting **faster, smarter, and auditable**.

---

## 🚀 Live App
👉 **[Launch the Streamlit App](https://your-app-url.streamlit.app/)**  
*(replace the link above with your actual Streamlit Cloud URL)*

---

## 🧠 Overview

This project automates the entire manpower budgeting process for subsidiaries of an organization — from importing HR data to generating full-year payroll projections.  
It translates complex Excel logic (with nested `LET()`, `IF()`, and `YEARFRAC()` functions) into transparent and maintainable Python code.

### 💼 Key Capabilities

✅ **Automatic data loading & cleaning**
- Reads Excel or CSV files in multiple encodings (`iso-8859-7`, `utf-8`)  
- Dynamically renames Greek/English columns (e.g., “Ημ/νία πρόσληψης” → `Hire Date`)  
- Validates and filters employees by hire/retire date  

✅ **Smart salary adjustments**
- Detects and multiplies daily/hourly salaries (`< 90`) by 26  
- Handles missing or malformed numeric values  
- Allows manual edits through an **interactive editor with Undo/Redo**

✅ **Contribution logic**
- Automatically applies contribution overrides based on “Κωδικός Κράτησης” codes  
- Computes `Monthly Employer's Contributions` with variable rates and fixed fees  
- Integrates optional **Contributions file** for custom rates per employee  

✅ **Projection engine**
- Implements translated Excel logic for:
  - `Months Projection 25`
  - `FY Months Budget 26`
  - `FY Gross Salary Projection for 25`
  - `Annual Gross Salary FY Budget 2026`
  - `Annual Employer’s Contributions for 2026`
  - `FY Payroll Cost Budget 2026`
- Supports per-employee logic based on hire/retire year and grade (e.g. conditional 5% vs 10% increases)

✅ **Filtering & Reporting**
- Sidebar filters by Company, Division, Department, and Cost Center (with “All” option)  
- Adds total row automatically across all numeric columns  
- Enables Excel download of both filtered data and totals  

✅ **Audit-friendly & consistent**
- Designed for ESG and HR reporting pipelines  
- All calculations are transparent, reproducible, and traceable back to Excel formulas

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | [Streamlit](https://streamlit.io/) — interactive Python dashboard |
| **Backend Logic** | Pandas, NumPy, and dateutil for time-based calculations |
| **Excel Parsing** | `openpyxl` and `xlsxwriter` for importing/exporting XLSX |
| **Data Editing** | Streamlit’s `st.data_editor` with Undo/Redo functionality |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ Setup & Run Locally

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/manpower-budget-automation.git
cd manpower-budget-automation

# 2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (use .venv\Scripts\activate on Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run streamlit_app.py
