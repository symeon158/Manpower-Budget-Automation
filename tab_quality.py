"""Sidebar rendering — branded card + filters + per-scenario panels."""
from __future__ import annotations

import io
from datetime import date, datetime

import pandas as pd
import streamlit as st

from mb.config import (CONTRIB_RATE_MAP, DIM_COLS, YOUTH_AGE_THRESHOLD,
                       YOUTH_DISCOUNT_PP)
from mb.scenarios import Scenario
from mb.sharepoint import download_bytes, to_graph_url


def render_sidebar_styles() -> None:
    """All sidebar CSS (branded card + white-text overrides)."""
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def render_sidebar_filters() -> None:
    st.sidebar.header("🔎 Filters")
    opts = st.session_state.get("_filter_opts", {})
    rendered = False
    for dim in DIM_COLS:
        o = opts.get(dim, [])
        if o:
            st.sidebar.multiselect(dim, o, default=[], key=f"flt_{dim}")
            rendered = True
    if not rendered:
        st.sidebar.caption("⏳ Filters will appear once the data is loaded.")
    st.sidebar.markdown("---")


def render_excel_options(token: str, main_file_url: str) -> tuple[str | None, int]:
    """The 'Excel import options' expander. Returns (sheet_name, header_row)."""
    main_sheet = None
    main_hdr   = 0
    with st.sidebar.expander("Excel import options (MAIN)", expanded=False):
        try:
            raw_peek    = download_bytes(to_graph_url(main_file_url), token, 0)
            sheet_names = pd.ExcelFile(io.BytesIO(raw_peek)).sheet_names
            main_sheet  = st.selectbox("Sheet", sheet_names, index=0, key="ms")
            main_hdr    = st.number_input("Header row (0-based)", 0, 100, 0, key="mh")
        except Exception:
            pass
    return main_sheet, main_hdr


def render_scenario_panel(label: str, key_prefix: str, defaults: dict) -> Scenario:
    """
    Render a full set of parameter widgets for a scenario, return the
    validated Scenario object.
    """
    st.sidebar.header(f"{label} — Parameters")

    payroll_periods = st.sidebar.number_input(
        "Payroll Periods per Year",
        value=defaults.get("payroll_periods", 14),
        min_value=12, max_value=14, step=1,
        key=f"{key_prefix}_pp")
    projection_date = pd.to_datetime(st.sidebar.date_input(
        "Projection Date",
        value=defaults.get("projection_date", datetime(2025, 10, 1)),
        key=f"{key_prefix}_pd"))
    no_increase_cutoff = st.sidebar.date_input(
        "No Increases After",
        value=defaults.get("no_increase_cutoff", date(2025, 8, 1)),
        key=f"{key_prefix}_nic")
    effective_increase_date = st.sidebar.date_input(
        "Effective Date of Increases",
        value=defaults.get("effective_increase_date", date(2026, 5, 1)),
        key=f"{key_prefix}_eid")

    salary_increase_pct = st.sidebar.number_input(
        "Avg Salary Increase %",
        value=defaults.get("salary_increase_pct", 3.0), step=0.1,
        key=f"{key_prefix}_sip") / 100.0
    salary_increase_pct2 = st.sidebar.number_input(
        "Avg Salary Increase % (Gr 0.1)",
        value=defaults.get("salary_increase_pct2", 5.0), step=0.1,
        key=f"{key_prefix}_sip2") / 100.0

    st.sidebar.markdown(
        "<div style='color:#ffffff; font-size:0.85rem; margin-bottom:0.4rem;'>"
        "<b>Under-25 Contribution Discount</b></div>",
        unsafe_allow_html=True)
    apply_youth_discount = st.sidebar.checkbox(
        "Apply under-25 discount",
        value=defaults.get("apply_youth_discount", True),
        key=f"{key_prefix}_yth_on")
    youth_threshold = st.sidebar.number_input(
        "Age threshold (strictly under)",
        value=defaults.get("youth_threshold", YOUTH_AGE_THRESHOLD),
        min_value=18, max_value=35, step=1,
        key=f"{key_prefix}_yth_age")
    youth_discount_pp = st.sidebar.number_input(
        "Discount (percentage points)",
        value=defaults.get("youth_discount_pp", YOUTH_DISCOUNT_PP * 100),
        min_value=0.0, max_value=20.0, step=0.01,
        key=f"{key_prefix}_yth_pp") / 100.0

    # Contribution rate table override
    with st.sidebar.expander(f"{label} — Contribution rates", expanded=False):
        st.caption("Override per-code rates for this scenario. "
                   "Edit values, leave others untouched.")
        rate_map_default = defaults.get("contrib_rate_map", CONTRIB_RATE_MAP)
        custom_map: dict[str, float] = {}
        for code in sorted(rate_map_default.keys()):
            custom_map[code] = st.number_input(
                f"Code {code}",
                value=float(rate_map_default[code]),
                min_value=0.0, max_value=1.0, step=0.0001, format="%.4f",
                key=f"{key_prefix}_rate_{code}")

    try:
        return Scenario(
            name                    = label,
            projection_date         = projection_date,
            no_increase_cutoff      = no_increase_cutoff,
            effective_increase_date = effective_increase_date,
            payroll_periods         = int(payroll_periods),
            salary_increase_pct     = float(salary_increase_pct),
            salary_increase_pct2    = float(salary_increase_pct2),
            apply_youth_discount    = bool(apply_youth_discount),
            youth_threshold         = int(youth_threshold),
            youth_discount_pp       = float(youth_discount_pp),
            contrib_rate_map        = custom_map,
        )
    except Exception as e:
        st.sidebar.error(f"❌ {label} invalid: {e}")
        raise


def render_reset_button() -> None:
    st.sidebar.markdown("---")
    if st.sidebar.button("♻️ Reset & Reconnect", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            if k.startswith(("flt_", "nh_", "_")) or k in (
                    "sp_token", "df_cache_key", "last_loaded_ts"):
                st.session_state.pop(k, None)
        st.rerun()


_SIDEBAR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

[data-testid="stSidebar"] { background-color: #111827; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] label {
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important; font-weight: 700 !important;
    margin-top: 1rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
    color: #ffffff !important; opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="InlineHelpIcon"] svg {
    color: #ffffff !important; fill: #ffffff !important; opacity: 0.85 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] [data-baseweb="checkbox"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stThumbValue"],
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #ffffff !important;
}

.control-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px; padding: 16px; margin-bottom: 25px;
    font-family: 'Inter', sans-serif;
}
.cc-header { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.cc-icon { width:32px; height:32px; background:#1f2937; border:1px solid #3b82f6;
    border-radius:6px; display:flex; align-items:center; justify-content:center;
    color:#3b82f6; flex-shrink:0; }
.cc-system { font-size:9px; color:#60a5fa; text-transform:uppercase;
    letter-spacing:1.5px; font-weight:700; font-family:'JetBrains Mono', monospace; }
.cc-title  { font-size:16px; font-weight:700; color:#f3f4f6; margin:0; }
.cc-sep { height:1px; background:linear-gradient(90deg, rgba(59,130,246,.3), transparent); margin:12px 0; }
.cc-info { font-size:11px; color:#9ca3af; line-height:1.6; }
.cc-info b { color:#d1d5db; font-weight:600; }
.cc-status { margin-top:12px; display:inline-flex; align-items:center; padding:4px 10px;
    background:rgba(16, 185, 129, .1); border:1px solid rgba(16, 185, 129, .2);
    border-radius:6px; color:#34d399; font-size:10px; font-weight:700;
    font-family:'JetBrains Mono', monospace; letter-spacing:0.5px; }
.cc-dot { width:6px; height:6px; background:#10b981; border-radius:50%;
    margin-right:8px; box-shadow:0 0 8px #10b981;
    animation: ccpulse 2s ease-in-out infinite; }
@keyframes ccpulse { 0%,100% {opacity:1;transform:scale(1)} 50% {opacity:.4;transform:scale(.8)} }
</style>

<div class="control-card">
<div class="cc-header">
<div class="cc-icon">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
<line x1="3" y1="9" x2="21" y2="9"></line>
<line x1="9" y1="21" x2="9" y2="9"></line>
</svg>
</div>
<div><span class="cc-system">ALUMIL ERP-BI</span><br><span class="cc-title">Budget Control</span></div>
</div>
<div class="cc-sep"></div>
<div class="cc-info"><b>Module:</b> Workforce Planning<br><b>Environment:</b> Azure-Prod</div>
<div class="cc-status"><div class="cc-dot"></div>SYSTEM ACTIVE</div>
</div>
"""
