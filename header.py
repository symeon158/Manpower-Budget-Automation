"""New Hiring Request tab."""
from __future__ import annotations

import io
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st

from mb.pay_ranges import cascade_options, lookup_range
from mb.sharepoint import download_bytes, to_graph_url, write_rows_to_sp


def render_tab_hiring(df_data: pd.DataFrame, token: str, main_file_url: str,
                       cache_ttl_min: int, param_sig: str) -> None:
    st.subheader("➕ New Hiring Request")
    st.caption("Submitted records are written to the **New_Hirings** sheet "
               "in ManpowerBudget.xlsx on SharePoint.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _build_lookups(cache_key: str):
        _df = st.session_state.get("_df_data_for_hiring")
        if _df is None or _df.empty:
            return [], {}, {}, {}
        def _uniq(s):
            return sorted(s.dropna().astype(str).str.strip()
                           .loc[lambda x: x != ""].unique().tolist())
        divisions = _uniq(_df["Division"]) if "Division" in _df.columns else []
        div_dept = {}
        if {"Division", "Department"}.issubset(_df.columns):
            for d, grp in _df.groupby("Division"):
                div_dept[str(d).strip()] = _uniq(grp["Department"])
        dd_jt = {}
        if {"Division", "Department", "Job Title"}.issubset(_df.columns):
            for (d, dp), grp in _df.groupby(["Division", "Department"]):
                dd_jt[(str(d).strip(), str(dp).strip())] = _uniq(grp["Job Title"])
        dd_jp = {}
        if {"Division", "Department", "Job Property"}.issubset(_df.columns):
            for (d, dp), grp in _df.groupby(["Division", "Department"]):
                dd_jp[(str(d).strip(), str(dp).strip())] = _uniq(grp["Job Property"])
        return divisions, div_dept, dd_jt, dd_jp

    st.session_state["_df_data_for_hiring"] = df_data
    all_divisions, div_dept_map, dd_jt_map, dd_jp_map = _build_lookups(param_sig)

    def _on_division_change():
        for k in ("nh_department", "nh_jobtitle", "nh_jobprop",
                   "nh_jobtitle_custom"):
            st.session_state.pop(k, None)

    def _on_department_change():
        for k in ("nh_jobtitle", "nh_jobprop", "nh_jobtitle_custom"):
            st.session_state.pop(k, None)

    def _on_jobtitle_change():
        if st.session_state.get("nh_jobtitle") != "(other — type below)":
            st.session_state.pop("nh_jobtitle_custom", None)

    # Pay-range cascade callbacks: clear downstream when a parent changes
    def _on_ref_level_change():
        for k in ("nh_pay_zone", "nh_pay_location"):
            st.session_state.pop(k, None)

    def _on_pay_zone_change():
        st.session_state.pop("nh_pay_location", None)

    def _submit():
        div   = st.session_state.get("nh_division", "")
        dept  = st.session_state.get("nh_department", "")
        jt    = st.session_state.get("nh_jobtitle", "")
        if jt == "(other — type below)":
            jt = st.session_state.get("nh_jobtitle_custom", "").strip()
        jp    = st.session_state.get("nh_jobprop", "")
        hd    = st.session_state.get("nh_hiring_date", date.today())
        qty   = int(st.session_state.get("nh_quantity", 1))
        rb    = st.session_state.get("nh_requested_by", "")
        nt    = st.session_state.get("nh_notes", "")

        # Resolve pay range (validated in form_valid below)
        ref_level    = st.session_state.get("nh_ref_level",    "")
        pay_zone     = st.session_state.get("nh_pay_zone",     "")
        pay_location = st.session_state.get("nh_pay_location", "")
        pay = lookup_range(ref_level, pay_zone, pay_location)
        median_salary = pay["p50"] if pay else None
        p25 = pay["p25"] if pay else None
        p75 = pay["p75"] if pay else None

        request_id   = f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build N rows (one per person).  "Seq" removed per request; the
        # Median Salary goes where Seq used to be.
        rows = [
            {"Request ID":      request_id,
             "Division":        div,
             "Department":      dept,
             "Job Description": jt,
             "Job Property":    jp,
             "Ref Level":       ref_level,
             "Pay Zone":        pay_zone,
             "Pay Location":    pay_location,
             "Median Salary":   f"{median_salary:.2f}" if median_salary is not None else "",
             "P25":             f"{p25:.2f}" if p25 is not None else "",
             "P75":             f"{p75:.2f}" if p75 is not None else "",
             "Hiring Date":     str(hd),
             "Quantity":        qty,
             "Requested By":    rb,
             "Notes":           nt,
             "Submitted At":    submitted_at}
            for _ in range(qty)]

        try:
            write_rows_to_sp(token, main_file_url, "New_Hirings", rows)
            download_bytes.clear()
            st.session_state["nh_last_submission"] = {
                "count": qty, "request_id": request_id, "rows": rows[0]}
            st.session_state.pop("nh_submit_error", None)
        except Exception as e:
            st.session_state["nh_submit_error"] = str(e)
            st.session_state.pop("nh_last_submission", None)
            return
        for k in ["nh_division", "nh_department", "nh_jobtitle",
                   "nh_jobtitle_custom", "nh_jobprop", "nh_hiring_date",
                   "nh_quantity", "nh_requested_by", "nh_notes",
                   "nh_ref_level", "nh_pay_zone", "nh_pay_location"]:
            st.session_state.pop(k, None)

    st.markdown("### 📋 Fill in the hiring details")
    col_a, col_b = st.columns(2)

    with col_a:
        st.selectbox("Division *", ["— select —"] + all_divisions,
                      key="nh_division", on_change=_on_division_change)
        sel_div  = st.session_state.get("nh_division", "— select —")
        dept_opts = div_dept_map.get(sel_div, []) if sel_div != "— select —" else []
        st.selectbox("Department *", ["— select —"] + dept_opts,
                      key="nh_department", on_change=_on_department_change,
                      disabled=(sel_div == "— select —"))
        sel_dept = st.session_state.get("nh_department", "— select —")
        jt_key   = (sel_div, sel_dept)
        jt_opts  = dd_jt_map.get(jt_key, []) if sel_dept != "— select —" else []
        st.selectbox("Job Description *",
                      ["— select —"] + jt_opts + ["(other — type below)"],
                      key="nh_jobtitle", on_change=_on_jobtitle_change,
                      disabled=(sel_dept == "— select —"))
        if st.session_state.get("nh_jobtitle") == "(other — type below)":
            st.text_input("Custom Job Description *", key="nh_jobtitle_custom")

    with col_b:
        jp_opts = dd_jp_map.get(jt_key, []) if sel_dept != "— select —" else []
        st.selectbox("Job Property *", ["— select —"] + jp_opts,
                      key="nh_jobprop", disabled=(sel_dept == "— select —"))
        st.number_input("Number of Employees to Hire *",
                         min_value=1, max_value=50, value=1, step=1,
                         key="nh_quantity")
        st.date_input("Expected Hiring Date *", value=date.today(),
                       key="nh_hiring_date")
        st.text_input("Requested by (name/email)", key="nh_requested_by")
        st.text_area("Notes / justification", key="nh_notes", height=80)

    # ── Pay range section ───────────────────────────────────────────────────
    st.markdown("### 💼 Pay Range")
    st.caption("Select the reference level, pay zone and location to assign a "
               "salary band.  The **median (P50)** salary will be written to "
               "the New_Hirings sheet.  "
               "For reference levels below 8, a flat €1,300 is used.")

    from mb.pay_ranges import BELOW_8_LABEL   # local import to stay self-contained

    pr_a, pr_b, pr_c = st.columns(3)
    ranges_opts = cascade_options()

    with pr_a:
        st.selectbox("Reference Level *",
                      ["— select —"] + ranges_opts["levels"],
                      key="nh_ref_level", on_change=_on_ref_level_change)
    sel_level = st.session_state.get("nh_ref_level", "— select —")
    is_below_8 = (sel_level == BELOW_8_LABEL)

    with pr_b:
        if is_below_8:
            # Level "< 8" has no pay-zone concept → disable the dropdown
            st.selectbox("Pay Zone", ["(not applicable)"],
                          key="nh_pay_zone_disabled", disabled=True)
            # Clear any stale real selection so it doesn't bleed through
            st.session_state.pop("nh_pay_zone", None)
        else:
            zone_opts = (cascade_options(selected_level=sel_level)["zones"]
                          if sel_level != "— select —" else [])
            st.selectbox("Pay Zone *",
                          ["— select —"] + zone_opts,
                          key="nh_pay_zone", on_change=_on_pay_zone_change,
                          disabled=(sel_level == "— select —"))
    sel_zone = st.session_state.get("nh_pay_zone", "— select —")

    with pr_c:
        if is_below_8:
            st.selectbox("Location", ["(not applicable)"],
                          key="nh_pay_location_disabled", disabled=True)
            st.session_state.pop("nh_pay_location", None)
        else:
            loc_opts = (cascade_options(selected_level=sel_level,
                                          selected_zone=sel_zone)["locations"]
                         if (sel_level != "— select —" and sel_zone != "— select —")
                         else [])
            st.selectbox("Location *",
                          ["— select —"] + loc_opts,
                          key="nh_pay_location",
                          disabled=(sel_zone == "— select —"))
    sel_loc = st.session_state.get("nh_pay_location", "— select —")

    # Live preview of the selected band
    selected_range = None
    if is_below_8:
        selected_range = lookup_range(BELOW_8_LABEL, "", "")
    elif (sel_level != "— select —" and sel_zone != "— select —"
            and sel_loc != "— select —"):
        selected_range = lookup_range(sel_level, sel_zone, sel_loc)

    if selected_range:
        if selected_range.get("is_fixed"):
            st.info(f"💰 **Fixed median salary for level < 8: €{selected_range['p50']:,.2f}**  "
                    "(no Pay Zone / Location applies)")
        else:
            m1, m2, m3 = st.columns(3)
            p25 = selected_range["p25"]
            p50 = selected_range["p50"]
            p75 = selected_range["p75"]
            m1.metric("P25 (low)",       f"€{p25:,.2f}" if p25 is not None else "—")
            m2.metric("P50 (median) ⭐", f"€{p50:,.2f}" if p50 is not None else "—",
                      help="This is the value written to SharePoint as Median Salary.")
            m3.metric("P75 (high)",      f"€{p75:,.2f}" if p75 is not None else "—")

    st.markdown("---")

    jt_val = st.session_state.get("nh_jobtitle", "— select —")
    jt_ok  = (jt_val not in ("— select —", ""))
    if jt_val == "(other — type below)":
        jt_ok = bool(st.session_state.get("nh_jobtitle_custom", "").strip())

    form_valid = all([
        st.session_state.get("nh_division",   "— select —") not in ("— select —", ""),
        st.session_state.get("nh_department", "— select —") not in ("— select —", ""),
        jt_ok,
        st.session_state.get("nh_jobprop",    "— select —") not in ("— select —", ""),
        int(st.session_state.get("nh_quantity", 1)) >= 1,
        # Pay-range must resolve — either "< 8" OR a valid level+zone+location triple
        selected_range is not None,
    ])
    if not form_valid:
        st.info("👆 Complete all required fields (*) — including a Reference Level "
                "(and Pay Zone / Location if level ≥ 8).")

    qty = int(st.session_state.get("nh_quantity", 1))
    st.button(f"✅ Submit Hiring Request — writes {qty} row{'s' if qty > 1 else ''}",
              disabled=not form_valid, type="primary", on_click=_submit)

    if "nh_submit_error" in st.session_state:
        st.error(f"❌ Could not write to SharePoint: {st.session_state['nh_submit_error']}\n\n"
                 "Make sure the app has **Files.ReadWrite** permission.")

    if "nh_last_submission" in st.session_state:
        s = st.session_state["nh_last_submission"]
        r = s["rows"]
        st.success(
            f"✅ Submitted **{s['count']}** row{'s' if s['count'] > 1 else ''} "
            f"(Request ID `{s['request_id']}`): **{r.get('Job Description')}** — "
            f"**{r.get('Division')} / {r.get('Department')}** on **{r.get('Hiring Date')}**")
        with st.expander("View submitted record"):
            st.json(r)
        st.button("✖ Dismiss", key="nh_dismiss",
                   on_click=lambda: st.session_state.pop("nh_last_submission", None))

    st.markdown("---")
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        preview_clicked = st.button("👁️ Preview existing New_Hirings records")
    with col_p2:
        force_fresh = st.checkbox("Force fresh", value=True,
                                   help="Bypass cache for just-submitted rows.")

    if preview_clicked:
        with st.spinner("Loading New_Hirings sheet…"):
            try:
                if force_fresh:
                    resp = requests.get(
                        to_graph_url(main_file_url),
                        headers={"Authorization": f"Bearer {token}"},
                        allow_redirects=True, timeout=60)
                    resp.raise_for_status()
                    df_nh = pd.read_excel(io.BytesIO(resp.content),
                                           sheet_name="New_Hirings")
                else:
                    from mb.sharepoint import load_sp_excel
                    df_nh = load_sp_excel(token, main_file_url,
                                           sheet_name="New_Hirings",
                                           ttl_min=cache_ttl_min)
                if df_nh.empty:
                    st.info("📭 No entries yet in New_Hirings.")
                else:
                    st.success(f"✅ Loaded {len(df_nh)} record(s) from SharePoint.")
                    st.dataframe(df_nh, use_container_width=True, height=400)
            except Exception as e:
                st.warning(f"Could not load New_Hirings sheet: {e}")
