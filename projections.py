"""XLSX export with formatting."""
from __future__ import annotations

import io

import pandas as pd


def to_formatted_xlsx(df_in: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_in.to_excel(writer, index=False, sheet_name=sheet_name)
        wb = writer.book
        ws = writer.sheets[sheet_name]
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1F4E79",
                                  "font_color": "white", "border": 1,
                                  "align": "center", "valign": "vcenter"})
        cur_fmt = wb.add_format({"num_format": "#,##0.00", "border": 1})
        dt_fmt  = wb.add_format({"num_format": "dd/mm/yyyy", "border": 1})
        gen_fmt = wb.add_format({"border": 1})
        cur_kw  = ["salary", "cost", "budget", "contribution",
                   "allowance", "training"]
        dt_kw   = ["date"]
        for ci, cn in enumerate(df_in.columns):
            ws.write(0, ci, cn, hdr_fmt)
            cnl = cn.lower()
            fmt = cur_fmt if any(k in cnl for k in cur_kw) else (
                dt_fmt if any(k in cnl for k in dt_kw) else gen_fmt)
            ws.set_column(ci, ci, 18, fmt)
        ws.freeze_panes(1, 4)
        ws.autofilter(0, 0, len(df_in), len(df_in.columns) - 1)
    return buf.getvalue()
