"""Budget & Analytics tab — with A vs B scenario comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mb.config import DIM_COLS
from mb.export import to_formatted_xlsx
from mb.scenarios import Scenario, ScenarioColumns, compare_scenarios, run_scenario


FINAL_ORDER_TEMPLATE = [
    "Company", "Hrms Id", "Surname", "Name", "Division", "Department",
    "Job Title", "Job Property", "Grade", "Date of Birth",
    "Hiring Date", "Retire Date", "Cost Center",
    "Monthly Gross Salary (Current)",
    "Κωδικός Κράτησης", "Contrib Rate Matched",
    "Contributions%", "Effective Contribution Rate", "Youth Discount Applied",
    "Monthly Employer's Contributions",
]


def _ordered_cols(df: pd.DataFrame, cols: ScenarioColumns) -> list[str]:
    wanted = FINAL_ORDER_TEMPLATE + [
        cols.months_col, cols.fy_months_col, cols.fy_gross_col,
        cols.fy_emp_cont_col, cols.total_cost_py,
        cols.ann_gross_col, cols.ann_emp_cont_col, cols.fy_cost_by,
        "Annual Training Cost", "Annual Meal Allowance/ Coupons Cost",
    ]
    return [c for c in wanted if c in df.columns]


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for dim in DIM_COLS:
        sel = st.session_state.get(f"flt_{dim}", [])
        if sel:
            out = out[out[dim].astype(str).str.strip().isin(sel)]
    return out


@st.cache_data(
    ttl=900,
    show_spinner=False,
    hash_funcs={Scenario: lambda s: s.signature()},
)
def _run_scenario_cached(scenario: Scenario, df_hash: int) -> tuple:
    """
    Run a scenario with caching.

    `hash_funcs` tells st.cache_data how to hash a Scenario object: we use
    its `.signature()` which is a stable MD5 of all its parameters. This is
    the canonical Streamlit pattern for caching on objects that contain
    types the default hasher can't handle (pd.Timestamp, dict, etc.).
    """
    df = st.session_state["_df_precontrib"]
    return run_scenario(df, scenario)


def _run_cached(s: Scenario) -> tuple[pd.DataFrame, ScenarioColumns]:
    """Run a scenario, using the result cache when inputs haven't changed."""
    df = st.session_state["_df_precontrib"]
    df_hash = int(pd.util.hash_pandas_object(df.index).sum())
    return _run_scenario_cached(s, df_hash)


def _fmt_money(v: float) -> str:
    return f"€{v:,.0f}"


def render_kpi_comparison(
    df_a: pd.DataFrame, cols_a: ScenarioColumns,
    df_b: pd.DataFrame, cols_b: ScenarioColumns,
    s_b: Scenario,
) -> None:
    """Single KPI row with A vs B deltas."""
    summary = compare_scenarios(df_a, cols_a, df_b, cols_b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Headcount", f"{summary['headcount_a']:,}",
              delta=f"{summary['headcount_b'] - summary['headcount_a']:+d} vs B",
              delta_color="off")
    c2.metric(f"💶 Total {s_b.budget_year} — A",
              _fmt_money(summary["total_cost_a"]))
    c3.metric(f"💶 Total {s_b.budget_year} — B",
              _fmt_money(summary["total_cost_b"]),
              delta=f"{_fmt_money(summary['delta_cost'])} ({summary['delta_pct']:+.2f}%)",
              delta_color="inverse")
    c4.metric("💰 Avg Monthly Salary",
              _fmt_money(summary["avg_sal_a"]),
              delta=_fmt_money(summary["avg_sal_b"] - summary["avg_sal_a"]),
              delta_color="off")

    if not summary["by_division"].empty:
        with st.expander("📊 Δ by Division", expanded=False):
            st.dataframe(summary["by_division"], use_container_width=True,
                         hide_index=True)


def render_tab_budget(scenario_a: Scenario, scenario_b: Scenario | None,
                      cache_ttl_min: int) -> None:
    l, r = st.columns([6, 2])
    with l:
        st.caption(f"📂 Data last loaded at **{st.session_state.get('last_loaded_ts', '—')}**"
                   f"  |  Cache: {cache_ttl_min} min")
    with r:
        if st.button("🔄 Refresh data now"):
            st.cache_data.clear()
            for k in ["df_cache_key", "_filter_opts", "_projections_sig",
                      "_df_data", "_df_full"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Run both scenarios
    try:
        df_a, cols_a = _run_cached(scenario_a)
    except Exception as e:
        st.error(f"❌ Scenario A failed: {e}"); return

    df_b, cols_b = None, None
    if scenario_b is not None:
        try:
            df_b, cols_b = _run_cached(scenario_b)
        except Exception as e:
            st.warning(f"⚠️ Scenario B failed: {e}")

    # Contribution coverage info
    if "Contributions%" in df_a.columns:
        matched = int(df_a["Contributions%"].notna().sum())
        total   = len(df_a)
        if matched < total:
            st.warning(f"⚠️ {total - matched} of {total} employees have a "
                       "Κωδικός Κράτησης not in the rate table — see Data Quality tab.")
        else:
            st.success(f"✅ All {total} employees matched to a contribution rate.")

    # Apply user-selected filters to both
    df_a_filt = _apply_filters(df_a)
    df_b_filt = _apply_filters(df_b) if df_b is not None else None

    # ── KPI comparison (or single-scenario fallback) ────────────────────────
    st.subheader(f"📊 Budget Summary — {scenario_a.proj_year} Projection / "
                 f"{scenario_a.budget_year} Budget")

    if scenario_b is not None and df_b_filt is not None:
        st.markdown(f"**Comparing:** `{scenario_a.name}` vs `{scenario_b.name}`")
        render_kpi_comparison(df_a_filt, cols_a, df_b_filt, cols_b, scenario_b)
    else:
        # Single-scenario KPIs
        _render_single_kpi(df_a_filt, cols_a, scenario_a)

    st.caption(f"Showing {len(df_a_filt):,} of {len(df_a):,} employees")

    # ── Data table toggle (A or B) ──────────────────────────────────────────
    st.markdown("---")
    if scenario_b is not None:
        which = st.radio("View data for:",
                          [scenario_a.name, scenario_b.name],
                          horizontal=True, key="which_scenario_view")
        df_view  = df_a_filt if which == scenario_a.name else df_b_filt
        cols_view = cols_a   if which == scenario_a.name else cols_b
    else:
        df_view, cols_view = df_a_filt, cols_a

    display = df_view[_ordered_cols(df_view, cols_view)].copy()

    # Totals row — sum ONLY calculated cost columns
    # (skip Grade, Contributions%, Effective Rate, raw salary etc.)
    cost_cols_for_total = [c for c in [
        cols_view.fy_gross_col,
        cols_view.fy_emp_cont_col,
        cols_view.total_cost_py,
        cols_view.ann_gross_col,
        cols_view.ann_emp_cont_col,
        cols_view.fy_cost_by,
        "Monthly Employer's Contributions",
        "Annual Training Cost",
        "Annual Meal Allowance/ Coupons Cost",
    ] if c in display.columns]

    totals = {c: "" for c in display.columns}
    for dim in DIM_COLS:
        if dim in display.columns:
            totals[dim] = "TOTAL"; break
    for c in cost_cols_for_total:
        totals[c] = round(float(pd.to_numeric(display[c], errors="coerce")
                                  .fillna(0).sum()), 2)
    display_with_totals = pd.concat([display, pd.DataFrame([totals])],
                                      ignore_index=True)

    st.markdown("### 📄 Employee Data")
    st.dataframe(display_with_totals, use_container_width=True, height=500)

    # ── Downloads ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            f"⬇️ Download {scenario_a.name} (XLSX)",
            data=to_formatted_xlsx(df_a_filt[_ordered_cols(df_a_filt, cols_a)],
                                    scenario_a.name),
            file_name=f"budget_{scenario_a.name}_{scenario_a.proj_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if scenario_b is not None and df_b_filt is not None:
        with c2:
            st.download_button(
                f"⬇️ Download {scenario_b.name} (XLSX)",
                data=to_formatted_xlsx(df_b_filt[_ordered_cols(df_b_filt, cols_b)],
                                        scenario_b.name),
                file_name=f"budget_{scenario_b.name}_{scenario_b.proj_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            # Side-by-side comparison export
            st.download_button(
                "⬇️ Download A vs B summary (XLSX)",
                data=_build_comparison_xlsx(df_a_filt, cols_a, df_b_filt, cols_b,
                                             scenario_a, scenario_b),
                file_name=f"comparison_{scenario_a.proj_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Charts ──────────────────────────────────────────────────────────────
    st.markdown("---")
    show_dash = st.checkbox(
        "📊 Show Visual Analytics Dashboard",
        value=True, key="show_dashboard",
        help="Uncheck to hide all charts and speed up reruns when you're "
             "iterating on filters or scenario parameters.")
    if show_dash:
        render_charts(df_a_filt, cols_a, scenario_a,
                       df_b_filt, cols_b, scenario_b)


def _render_single_kpi(df: pd.DataFrame, cols: ScenarioColumns, s: Scenario) -> None:
    c1, c2, c3, c4 = st.columns(4)
    hc  = df["Hrms Id"].nunique()  if "Hrms Id" in df.columns else 0
    tot = df[cols.fy_cost_by].sum() if cols.fy_cost_by in df.columns else 0
    py  = df[cols.total_cost_py].sum() if cols.total_cost_py in df.columns else 0
    avg = df["Monthly Gross Salary (Current)"].mean() if \
        "Monthly Gross Salary (Current)" in df.columns else 0
    c1.metric("👥 Headcount",              f"{hc:,}")
    c2.metric(f"💶 Payroll Cost {s.budget_year}", _fmt_money(tot))
    c3.metric(f"💶 Payroll Cost {s.proj_year}",   _fmt_money(py))
    c4.metric("💰 Avg Monthly Salary",     _fmt_money(avg))


def _build_comparison_xlsx(
    df_a, cols_a, df_b, cols_b, s_a: Scenario, s_b: Scenario
) -> bytes:
    """Build a multi-sheet workbook: Summary, A, B, Δ by Division."""
    import io
    buf = io.BytesIO()
    summary = compare_scenarios(df_a, cols_a, df_b, cols_b)

    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # Summary sheet
        summary_rows = pd.DataFrame([
            ["Metric", s_a.name, s_b.name, "Δ (€)", "Δ (%)"],
            ["Headcount", summary["headcount_a"], summary["headcount_b"],
             summary["headcount_b"] - summary["headcount_a"], ""],
            [f"Total Payroll Cost {s_a.budget_year}",
             summary["total_cost_a"], summary["total_cost_b"],
             summary["delta_cost"], f"{summary['delta_pct']:.2f}%"],
            ["Avg Monthly Salary",
             summary["avg_sal_a"], summary["avg_sal_b"],
             summary["avg_sal_b"] - summary["avg_sal_a"], ""],
        ])
        summary_rows.to_excel(writer, sheet_name="Summary",
                              index=False, header=False)

        # Detail sheets
        df_a[_ordered_cols(df_a, cols_a)].to_excel(
            writer, sheet_name=s_a.name[:31], index=False)
        df_b[_ordered_cols(df_b, cols_b)].to_excel(
            writer, sheet_name=s_b.name[:31], index=False)

        # Δ by Division
        if not summary["by_division"].empty:
            summary["by_division"].to_excel(
                writer, sheet_name="Delta by Division", index=False)

    return buf.getvalue()


def _grade_order(series: pd.Series) -> list:
    """Sort grades numerically with NaN/non-numeric pushed to the end."""
    return sorted(
        series.dropna().unique(),
        key=lambda g: (pd.to_numeric(str(g).replace(",", "."), errors="coerce") or 9999))


# ════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════
PALETTE_A = "#1F4E79"   # Alumil deep blue (Scenario A)
PALETTE_B = "#C00000"   # Brick red (Scenario B)
PALETTE_NEUTRAL = "#70AD47"  # Green for headcount / non-cost metrics


def _safe_chart(name: str):
    """Decorator that wraps a chart function and shows the error inline."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                st.error(f"❌ {name} chart failed: {e}")
        return wrapper
    return deco


# ────────────────────────────────────────────────────────────────────────────
# Section 1: Division Overview  (cost = comparison if B exists, headcount = A only)
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Division Budget")
def _chart_division_budget(df_a, cols_a, s_a, df_b, cols_b, s_b):
    if s_b is not None and df_b is not None:
        # Grouped bar A vs B
        div_a = df_a.groupby("Division")[cols_a.fy_cost_by].sum()
        div_b = df_b.groupby("Division")[cols_b.fy_cost_by].sum()
        comp = (pd.concat([div_a.rename(s_a.name), div_b.rename(s_b.name)], axis=1)
                .fillna(0)
                .sort_values(s_a.name, ascending=True)
                .reset_index())
        chart_h = max(380, len(comp) * 60 + 80)
        fig = go.Figure()
        fig.add_bar(y=comp["Division"], x=comp[s_a.name], name=s_a.name,
                     orientation="h", marker_color=PALETTE_A,
                     text=comp[s_a.name].apply(lambda v: f"€{v/1e6:.2f}M"),
                     textposition="outside",
                     hovertemplate=f"<b>%{{y}}</b><br>{s_a.name}: €%{{x:,.0f}}<extra></extra>")
        fig.add_bar(y=comp["Division"], x=comp[s_b.name], name=s_b.name,
                     orientation="h", marker_color=PALETTE_B,
                     text=comp[s_b.name].apply(lambda v: f"€{v/1e6:.2f}M"),
                     textposition="outside",
                     hovertemplate=f"<b>%{{y}}</b><br>{s_b.name}: €%{{x:,.0f}}<extra></extra>")
        fig.update_layout(
            title=f"Total Budget {s_a.budget_year} by Division — {s_a.name} vs {s_b.name}",
            barmode="group", xaxis_tickformat="€,.0f",
            xaxis_title="Total Budget (€)", yaxis_title=None,
            height=chart_h,
            legend=dict(orientation="h", y=-0.10),
            margin=dict(l=10, r=10, t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)
    else:
        ds = (df_a.groupby("Division")
              .agg(TotalCost=(cols_a.fy_cost_by, "sum"))
              .reset_index().sort_values("TotalCost", ascending=True))
        chart_h = max(380, len(ds) * 48 + 80)
        fig = go.Figure(go.Bar(
            y=ds["Division"], x=ds["TotalCost"],
            orientation="h", marker_color=PALETTE_A,
            text=ds["TotalCost"].apply(lambda v: f"€{v/1e6:.2f}M"),
            textposition="outside",
            hovertemplate="%{y}<br>Budget: €%{x:,.0f}<extra></extra>"))
        fig.update_layout(
            title=f"Total Budget {s_a.budget_year} by Division",
            xaxis=dict(title="Total Budget (€)", tickformat="€,.0f",
                       range=[0, ds["TotalCost"].max() * 1.18]),
            yaxis=dict(title=None),
            margin=dict(l=10, r=10, t=45, b=50),
            showlegend=False, height=chart_h)
        st.plotly_chart(fig, use_container_width=True)


@_safe_chart("Headcount by Division")
def _chart_headcount_by_division(df_a, s_a):
    dh = (df_a.groupby("Division")
          .agg(Headcount=("Hrms Id", "nunique"))
          .reset_index().sort_values("Headcount", ascending=True))
    chart_h = max(380, len(dh) * 48 + 80)
    fig = go.Figure(go.Bar(
        y=dh["Division"], x=dh["Headcount"],
        orientation="h", marker_color=PALETTE_NEUTRAL,
        text=dh["Headcount"], textposition="outside",
        hovertemplate="%{y}<br>Headcount: %{x}<extra></extra>"))
    fig.update_layout(
        title="Headcount by Division",
        xaxis=dict(title="Employees",
                   range=[0, dh["Headcount"].max() * 1.18]),
        yaxis=dict(title=None),
        margin=dict(l=10, r=10, t=45, b=50),
        showlegend=False, height=chart_h)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Section 2: Cost Composition
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Cost Waterfall")
def _chart_cost_waterfall(df_a, cols_a, s_a, df_b, cols_b, s_b):
    cost_components = {
        "Gross Salary":       cols_a.ann_gross_col,
        "Employer Contrib":   cols_a.ann_emp_cont_col,
        "Training":           "Annual Training Cost",
        "Meal Allowance":     "Annual Meal Allowance/ Coupons Cost",
    }
    avail = {k: v for k, v in cost_components.items() if v in df_a.columns}
    if len(avail) < 2:
        st.info("Not enough cost components present to render waterfall.")
        return

    totals = {k: float(df_a[v].sum()) for k, v in avail.items()}
    labels = list(totals.keys()) + ["Total Payroll Cost"]
    values = list(totals.values())
    grand  = sum(values)
    measures = ["relative"] * len(values) + ["total"]
    wf_vals  = values + [0]

    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        measure=measures, x=labels, y=wf_vals,
        text=[f"€{v/1e6:.2f}M" for v in values] + [f"€{grand/1e6:.2f}M"],
        textposition="outside",
        connector=dict(line=dict(color="#888", dash="dot")),
        increasing=dict(marker_color="#2E75B6"),
        decreasing=dict(marker_color="#C00000"),
        totals=dict(marker_color=PALETTE_A),
    ))

    title = f"Payroll Cost Waterfall — {s_a.budget_year} ({s_a.name})"
    if s_b is not None and df_b is not None and cols_b is not None:
        # Add Scenario B total as a marker for visual reference
        b_total = float(df_b[cols_b.fy_cost_by].sum())
        delta   = grand - b_total
        title  += f"  |  {s_b.name} total: €{b_total/1e6:.2f}M  (Δ {delta/1e6:+.2f}M)"

    fig.update_layout(
        title=title, yaxis_tickformat="€,.0f",
        yaxis_title="Annual Cost (€)",
        showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)


@_safe_chart("Salary Spread by Department")
def _chart_cost_per_employee(df_a, cols_a, s_a, df_b, cols_b, s_b):
    """
    Box-and-whisker plot of MONTHLY GROSS SALARIES per department.

    Reveals what "cost per employee" averages hide:
      • Internal salary inequity (long IQR vs short IQR)
      • Outliers (single very-high or very-low earners pulling the mean)
      • Median vs mean divergence (skewed distributions)

    Departments are ordered by median salary (highest at top).
    Limited to top-N departments by headcount to keep the chart readable.
    """
    if "Monthly Gross Salary (Current)" not in df_a.columns or \
       "Department" not in df_a.columns:
        st.info("Salary or Department column missing.")
        return

    sal_col = "Monthly Gross Salary (Current)"

    # Slider: which scenario's data and how many departments to show
    n_max = df_a["Department"].nunique()
    top_n = st.slider(
        "Show top N departments by headcount",
        min_value=5, max_value=int(n_max), value=min(20, int(n_max)),
        step=1, key="cpe_top_n",
        help="Higher N → more departments, but the chart gets crowded.")

    # Pick top-N departments by headcount
    top_depts = (df_a.groupby("Department")["Hrms Id"]
                 .nunique().nlargest(top_n).index.tolist())
    plot_df = df_a[df_a["Department"].isin(top_depts) & df_a[sal_col].notna()].copy()

    if plot_df.empty:
        st.info("No salary data to plot.")
        return

    # Sort by median (highest first)
    medians = (plot_df.groupby("Department")[sal_col].median()
               .sort_values(ascending=False))
    dept_order = medians.index.tolist()

    # Build per-department metadata for hover/annotation
    stats = (plot_df.groupby("Department")
             .agg(HC=("Hrms Id", "nunique"),
                  Median=(sal_col, "median"),
                  Mean=(sal_col, "mean"),
                  P25=(sal_col, lambda s: s.quantile(0.25)),
                  P75=(sal_col, lambda s: s.quantile(0.75)),
                  Min=(sal_col, "min"),
                  Max=(sal_col, "max"))
             .round(0))

    # Color each box by department's company-relative median z-score
    company_median = plot_df[sal_col].median()
    company_std    = plot_df[sal_col].std()
    z_scores = ((medians - company_median) / company_std).fillna(0)

    fig = go.Figure()
    for dept in dept_order:
        d = plot_df[plot_df["Department"] == dept]
        z = z_scores[dept]
        # Diverging color: blue (below median) → grey (at median) → red (above)
        if   z >  1.0: color = "#C00000"
        elif z >  0.3: color = "#E27A7A"
        elif z > -0.3: color = "#9CA3AF"
        elif z > -1.0: color = "#7AAEDC"
        else:          color = "#1F4E79"

        s = stats.loc[dept]
        hover = (f"<b>{dept}</b><br>"
                 f"Headcount: {int(s['HC'])}<br>"
                 f"Median: €{s['Median']:,.0f}<br>"
                 f"Mean:   €{s['Mean']:,.0f}<br>"
                 f"P25–P75: €{s['P25']:,.0f} – €{s['P75']:,.0f}<br>"
                 f"Range: €{s['Min']:,.0f} – €{s['Max']:,.0f}")

        fig.add_trace(go.Box(
            x=d[sal_col],
            name=dept,
            orientation="h",
            marker_color=color,
            line=dict(width=1.5),
            boxmean=True,                        # show mean as dashed line
            boxpoints="outliers",                 # only show outliers, not all points
            jitter=0.3,
            pointpos=0,
            hovertemplate=hover + "<extra></extra>",
        ))

    # Annotate company median as vertical reference line
    fig.add_vline(
        x=company_median, line_dash="dash", line_color="#666",
        annotation_text=f"Company median  €{company_median:,.0f}",
        annotation_position="top right",
        annotation_font_color="#666")

    # Sort y-axis (Plotly draws bottom→top, so reverse for "highest at top")
    fig.update_yaxes(categoryorder="array",
                      categoryarray=list(reversed(dept_order)))
    fig.update_layout(
        title=f"Monthly Salary Distribution by Department  "
              f"(top {len(dept_order)} by headcount, sorted by median)",
        xaxis_title="Monthly Gross Salary (€)",
        yaxis_title=None,
        xaxis_tickformat="€,.0f",
        showlegend=False,
        height=max(420, len(dept_order) * 30 + 100),
        margin=dict(l=10, r=10, t=60, b=50))

    st.plotly_chart(fig, use_container_width=True)

    # Optional: small caption explaining the color coding
    st.caption(
        "Color reflects how each department's median compares to the "
        "company median.  🟦 below average  •  ⚪ near average  •  🟥 above average. "
        "A wide box means high internal salary spread; outlier dots are "
        "individual employees beyond 1.5× IQR.")


# ────────────────────────────────────────────────────────────────────────────
# Section 3: Grade & Salary Analysis
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Salary Distribution")
def _chart_salary_box(df_a):
    grade_order = [str(g) for g in _grade_order(df_a["Grade"])]
    fig = px.box(
        df_a, x="Grade", y="Monthly Gross Salary (Current)",
        points="all", color="Grade",
        category_orders={"Grade": grade_order},
        title="Salary Distribution by Grade",
        labels={"Monthly Gross Salary (Current)": "Monthly Gross Salary (€)"})
    fig.update_layout(
        xaxis_tickangle=45 if len(grade_order) > 10 else 0,
        yaxis_tickformat="€,.0f",
        showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)


@_safe_chart("Headcount by Grade")
def _chart_headcount_by_grade(df_a):
    grade_order = [str(g) for g in _grade_order(df_a["Grade"])]
    gc = (df_a.groupby("Grade")["Hrms Id"]
          .count().reset_index(name="Headcount"))
    gc["Grade"] = gc["Grade"].astype(str)
    gc = gc.set_index("Grade").reindex(grade_order).reset_index()
    fig = px.bar(
        gc, x="Headcount", y="Grade", orientation="h",
        text="Headcount", color="Headcount",
        color_continuous_scale="Blues",
        title="Headcount by Grade",
        category_orders={"Grade": grade_order})
    fig.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Section 4: Salary Heatmap
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Salary Heatmap")
def _chart_salary_heatmap(df_a):
    grade_order = [str(g) for g in _grade_order(df_a["Grade"])]
    heat = (df_a.groupby(["Division", "Grade"])["Monthly Gross Salary (Current)"]
            .mean().round(0).reset_index())
    heat["Grade"] = heat["Grade"].astype(str)
    pivot = (heat.pivot(index="Division", columns="Grade",
                         values="Monthly Gross Salary (Current)")
             .reindex(columns=grade_order))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Blues",
        text=[[f"€{v:,.0f}" if pd.notna(v) else "" for v in row]
              for row in pivot.values],
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Avg Monthly<br>Salary (€)")))
    fig.update_layout(
        title="Average Monthly Salary — Grade × Division",
        xaxis_title="Grade", yaxis_title="Division",
        height=max(320, len(pivot.index) * 42 + 120))
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Section 5: Workforce Profile
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Department Scatter")
def _chart_department_scatter(df_a, cols_a, s_a):
    da = (df_a.groupby(["Department", "Division"])
          .agg(Headcount=("Hrms Id", "count"),
               Total_Budget=(cols_a.fy_cost_by, "sum"),
               Avg_Salary=(cols_a.ann_gross_col, "mean"))
          .reset_index())
    fig = px.scatter(
        da, x="Headcount", y="Avg_Salary",
        size="Total_Budget", color="Division",
        hover_name="Department", size_max=60, log_x=True,
        title=f"Department Analysis — {s_a.budget_year} (size = Total Budget)",
        labels={"Avg_Salary": "Avg Annual Gross Salary (€)",
                "Headcount": "Headcount (log scale)"})
    fig.update_layout(yaxis_tickformat="€,.0f",
                      legend=dict(title="Division", orientation="v"),
                      height=420)
    st.plotly_chart(fig, use_container_width=True)


@_safe_chart("Tenure Distribution")
def _chart_tenure(df_a):
    if "Hiring Date" not in df_a.columns:
        st.info("Hiring Date column not available for tenure chart.")
        return
    today    = pd.Timestamp.today()
    hd_clean = pd.to_datetime(df_a["Hiring Date"], errors="coerce")
    tenure   = ((today - hd_clean).dt.days / 365.25).dropna()
    if tenure.empty:
        st.info("No valid hiring dates for tenure chart.")
        return
    tf = df_a.loc[tenure.index].copy()
    tf["Tenure_Years"] = tenure.values
    fig = px.histogram(
        tf, x="Tenure_Years",
        color="Division" if "Division" in tf.columns else None,
        nbins=20, barmode="stack",
        title="Company-wide Tenure Distribution (stacked by Division)",
        labels={"Tenure_Years": "Years of Service", "count": "Employees"})
    med_t = tenure.median()
    fig.add_vline(x=med_t, line_dash="dash", line_color="red",
                  annotation_text=f"Median {med_t:.1f} yrs",
                  annotation_position="top right")
    fig.update_layout(xaxis_title="Years of Service",
                      yaxis_title="Number of Employees",
                      legend=dict(orientation="h", y=-0.25),
                      height=420)
    st.plotly_chart(fig, use_container_width=True)



# ────────────────────────────────────────────────────────────────────────────
# Section 7: Budget Treemap
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Budget Treemap")
def _chart_budget_treemap(df_a, cols_a, s_a):
    al = (df_a.groupby(["Company", "Division", "Department"])
          .agg(TotalCost=(cols_a.fy_cost_by, "sum"),
               HC=("Hrms Id", "nunique")).reset_index())
    av = (df_a.groupby(["Company", "Division"])
          .agg(TotalCost=(cols_a.fy_cost_by, "sum"),
               HC=("Hrms Id", "nunique")).reset_index())
    ac = (df_a.groupby(["Company"])
          .agg(TotalCost=(cols_a.fy_cost_by, "sum"),
               HC=("Hrms Id", "nunique")).reset_index())
    rl   = f"Total Budget {s_a.budget_year}"
    root = pd.DataFrame({"id": [rl], "parent": [""], "label": [rl],
                          "TotalCost": [ac["TotalCost"].sum()],
                          "HC": [ac["HC"].sum()], "level": ["Root"]})
    cp2 = ac.assign(id=lambda d: d["Company"], parent=rl,
                     label=lambda d: d["Company"], level="Company")[
        ["id", "parent", "label", "TotalCost", "HC", "level"]]
    dv2 = av.assign(id=lambda d: d["Company"] + " | " + d["Division"],
                     parent=lambda d: d["Company"],
                     label=lambda d: d["Division"], level="Division")[
        ["id", "parent", "label", "TotalCost", "HC", "level"]]
    dp3 = al.assign(id=lambda d: d["Company"] + " | " + d["Division"]
                                   + " | " + d["Department"],
                     parent=lambda d: d["Company"] + " | " + d["Division"],
                     label=lambda d: d["Department"], level="Department")[
        ["id", "parent", "label", "TotalCost", "HC", "level"]]
    nodes = pd.concat([root, cp2, dv2, dp3], ignore_index=True)
    fig = px.treemap(
        nodes, names="label", parents="parent", ids="id",
        values="TotalCost",
        title=f"Budget Breakdown — {s_a.budget_year}",
        custom_data=["HC", "TotalCost", "level"])
    fig.update_traces(
        branchvalues="total", textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>Level: %{customdata[2]}<br>"
                      "Total Cost: €%{customdata[1]:,.0f}<br>"
                      "Headcount: %{customdata[0]:.0f}<br>"
                      "Parent Share: %{percentParent:.2%}<extra></extra>")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Section 8: NEW — Salary increase impact (only meaningful with both scenarios)
# ────────────────────────────────────────────────────────────────────────────
@_safe_chart("Increase Impact")
def _chart_increase_impact(df_a, cols_a, s_a, df_b, cols_b, s_b):
    """
    Per-employee Δ between the two scenarios, ranked top-N most impacted.
    Helps identify where the budget swing comes from.
    """
    if s_b is None or df_b is None or cols_b is None:
        return
    a_per = df_a.set_index("Hrms Id")[cols_a.fy_cost_by]
    b_per = df_b.set_index("Hrms Id")[cols_b.fy_cost_by]
    delta = (b_per - a_per).dropna()
    if delta.empty:
        return
    top_n = 20
    delta_sorted = delta.reindex(
        delta.abs().sort_values(ascending=False).index).head(top_n)

    name_lookup = df_a.set_index("Hrms Id")
    labels = [
        f"{idx}  {name_lookup.loc[idx, 'Surname']} {name_lookup.loc[idx, 'Name']}"
        if idx in name_lookup.index else idx
        for idx in delta_sorted.index]

    fig = go.Figure(go.Bar(
        x=delta_sorted.values, y=labels, orientation="h",
        marker_color=[PALETTE_B if v > 0 else PALETTE_NEUTRAL
                      for v in delta_sorted.values],
        text=[f"€{v:+,.0f}" for v in delta_sorted.values],
        textposition="outside"))
    fig.update_layout(
        title=f"Top {top_n} employees — biggest Δ between {s_a.name} and {s_b.name}",
        xaxis_title=f"Δ {s_b.name} − {s_a.name}  (€/yr)",
        xaxis_tickformat="€,.0f",
        yaxis=dict(autorange="reversed"),
        height=max(420, top_n * 25 + 100),
        margin=dict(l=10, r=10, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
def render_charts(df_a: pd.DataFrame, cols_a: ScenarioColumns, s_a: Scenario,
                   df_b: pd.DataFrame | None, cols_b: ScenarioColumns | None,
                   s_b: Scenario | None) -> None:
    """Render the full visual analytics dashboard."""
    st.subheader("📊 Visual Analytics Dashboard")
    if df_a.empty:
        st.info("No data for the current filter selection.")
        return

    required = [cols_a.fy_cost_by, cols_a.ann_gross_col,
                "Monthly Gross Salary (Current)", "Hrms Id",
                "Division", "Department", "Company", "Grade"]
    missing = [c for c in required if c not in df_a.columns]
    if missing:
        st.warning(f"Dashboard missing columns: {', '.join(missing)}")
        return

    # Section 1
    st.markdown("#### 🏢 Division Overview")
    c1, c2 = st.columns(2)
    with c1:
        _chart_division_budget(df_a, cols_a, s_a, df_b, cols_b, s_b)
    with c2:
        _chart_headcount_by_division(df_a, s_a)

    # Section 2
    st.markdown("#### 💶 Cost Composition")
    c1, c2 = st.columns(2)
    with c1:
        _chart_cost_waterfall(df_a, cols_a, s_a, df_b, cols_b, s_b)
    with c2:
        _chart_cost_per_employee(df_a, cols_a, s_a, df_b, cols_b, s_b)

    # Section 3
    st.markdown("#### 🎓 Grade & Salary Analysis")
    c1, c2 = st.columns(2)
    with c1:
        _chart_salary_box(df_a)
    with c2:
        _chart_headcount_by_grade(df_a)

    # Section 4
    st.markdown("#### 🌡️ Salary Heatmap — Grade × Division")
    _chart_salary_heatmap(df_a)

    # Section 5
    st.markdown("#### 👥 Workforce Profile")
    c1, c2 = st.columns(2)
    with c1:
        _chart_department_scatter(df_a, cols_a, s_a)
    with c2:
        _chart_tenure(df_a)

    # Section 7
    st.markdown("#### 🗺️ Budget Hierarchy")
    _chart_budget_treemap(df_a, cols_a, s_a)

    # Section 8 — only when comparing
    if s_b is not None:
        st.markdown("#### 🎯 Per-Employee Δ Analysis (Scenario A vs B)")
        st.caption(
            "Identifies the employees driving the biggest budget swings between "
            "your two scenarios — useful for spotting where a salary-increase "
            "policy or a rate change concentrates its impact.")
        _chart_increase_impact(df_a, cols_a, s_a, df_b, cols_b, s_b)
