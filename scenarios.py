"""Data Quality tab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mb.config import CONTRIB_RATE_MAP
from mb.scenarios import Scenario


def render_tab_quality(df_data: pd.DataFrame, booking_col: str,
                        scenario: Scenario) -> None:
    st.subheader("🔍 Data Quality Report")
    st.caption("Issues found in the loaded dataset — use this to clean your source Excel file.")

    with st.expander("📖 Contribution rate lookup (Κωδικός Κράτησης → %)", expanded=False):
        rate_tbl = pd.DataFrame([
            {"Κωδικός Κράτησης": c,
             "Rate %": f"{r*100:.2f}%",
             "Fixed Add-on": ("€30" if abs(r - 0.1879) < 1e-6
                               else ("€25" if abs(r - 0.1738) < 1e-6 else "—"))}
            for c, r in sorted(CONTRIB_RATE_MAP.items())])
        st.dataframe(rate_tbl, use_container_width=True, hide_index=True)
        st.caption(
            f"Scenario **{scenario.name}**: discount "
            f"{'ON' if scenario.apply_youth_discount else 'OFF'}  |  "
            f"threshold < {scenario.youth_threshold} yrs  |  "
            f"{scenario.youth_discount_pp*100:.2f} pp")

    dq_issues = 0

    if "Cost Center" in df_data.columns:
        bad_cc = df_data[df_data["Cost Center"].isna() |
                         df_data["Cost Center"].astype(str).str.strip()
                           .isin(["", "nan", "None"])]
        if not bad_cc.empty:
            dq_issues += len(bad_cc)
            with st.expander(f"🟠 Missing Cost Center — {len(bad_cc)} employee(s)",
                              expanded=True):
                st.dataframe(
                    bad_cc[["Hrms Id", "Surname", "Name", "Division", "Department"]]
                    .drop_duplicates(), use_container_width=True)
        else:
            st.success("✅ All employees have a Cost Center.")

    if "Contributions%" in df_data.columns:
        bad_rate = df_data[df_data["Contributions%"].isna()]
        if not bad_rate.empty:
            dq_issues += len(bad_rate)
            with st.expander(f"🟠 Missing Contributions % — {len(bad_rate)} employee(s)",
                              expanded=True):
                st.caption("These employees have a Κωδικός Κράτησης that is NOT in the "
                           "built-in rate table.")
                show = [c for c in ["Hrms Id", "Surname", "Name", "Division",
                                      booking_col, "Κωδικός Κράτησης (norm)",
                                      "Contrib Rate Matched"]
                        if c in bad_rate.columns]
                st.dataframe(bad_rate[show], use_container_width=True)
        else:
            st.success("✅ All employees matched to a contribution rate.")

    if "Youth Discount Applied" in df_data.columns:
        yth = df_data[df_data["Youth Discount Applied"] == "Yes"]
        with st.expander(f"👶 Under-25 discount applied — {len(yth)} employee(s)",
                          expanded=False):
            st.caption(
                f"Employees whose age on {pd.Timestamp(scenario.projection_date).date()} "
                f"was strictly under {scenario.youth_threshold}. Discount: "
                f"{scenario.youth_discount_pp*100:.2f} pp. "
                "Fixed add-ons (€25/€30) are NOT reduced.")
            if len(yth):
                show = [c for c in ["Hrms Id", "Surname", "Name", "Division",
                                      "Date of Birth", "Contributions%",
                                      "Effective Contribution Rate"]
                        if c in yth.columns]
                st.dataframe(yth[show], use_container_width=True)
            else:
                st.info("No under-25 employees on the projection date.")

    if "Monthly Gross Salary (Current)" in df_data.columns:
        bad_sal = df_data[df_data["Monthly Gross Salary (Current)"].isna() |
                          df_data["Monthly Gross Salary (Current)"].eq(0)]
        if not bad_sal.empty:
            dq_issues += len(bad_sal)
            with st.expander(f"🔴 Zero or Missing Salary — {len(bad_sal)} employee(s)",
                              expanded=True):
                st.dataframe(
                    bad_sal[["Hrms Id", "Surname", "Name", "Division", "Department",
                             "Monthly Gross Salary (Current)"]],
                    use_container_width=True)
        else:
            st.success("✅ All employees have a valid salary.")

        med = df_data["Monthly Gross Salary (Current)"].median()
        outliers = df_data[df_data["Monthly Gross Salary (Current)"] > 5 * med]
        if not outliers.empty:
            with st.expander(
                    f"⚠️ Salary Outliers (> 5× median €{med:,.0f}) — "
                    f"{len(outliers)} employee(s)"):
                st.dataframe(
                    outliers[["Hrms Id", "Surname", "Name", "Division",
                              "Monthly Gross Salary (Current)"]],
                    use_container_width=True)

    if "Hiring Date" in df_data.columns:
        bad_hd = df_data[df_data["Hiring Date"].isna()]
        if not bad_hd.empty:
            dq_issues += len(bad_hd)
            with st.expander(f"🟠 Missing Hiring Date — {len(bad_hd)} employee(s)"):
                st.dataframe(bad_hd[["Hrms Id", "Surname", "Name", "Division"]],
                              use_container_width=True)
        else:
            st.success("✅ All employees have a Hiring Date.")

    st.markdown("---")
    if dq_issues == 0:
        st.success(f"🎉 No critical data quality issues found in "
                   f"{len(df_data):,} employees.")
    else:
        st.error(f"❗ {dq_issues} critical issue(s) found. See sections above.")
