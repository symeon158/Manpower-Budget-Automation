"""Microsoft Graph / SharePoint helpers."""
from __future__ import annotations

import base64
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=3500, show_spinner=False)
def get_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    resp = requests.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={"grant_type": "client_credentials",
              "client_id": client_id,
              "client_secret": client_secret,
              "scope": "https://graph.microsoft.com/.default"},
        timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def to_graph_url(file_url: str) -> str:
    if "graph.microsoft.com" in file_url:
        return file_url
    parsed  = urlparse(file_url)
    clean   = urlunparse(parsed._replace(query="", fragment=""))
    b64     = base64.urlsafe_b64encode(clean.encode()).decode().rstrip("=")
    encoded = "u!" + b64.replace("/", "_").replace("+", "-")
    return f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem/content"


@st.cache_data(ttl=120, show_spinner="⬇️ Downloading file from OneDrive…")
def download_bytes(graph_url: str, token: str, _ttl_bucket: int) -> bytes:
    resp = requests.get(graph_url, headers={"Authorization": f"Bearer {token}"},
                        allow_redirects=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def load_sp_excel(token: str, file_url: str, sheet_name=None,
                   header_row: int = 0, ttl_min: int = 2) -> pd.DataFrame:
    import io
    raw = download_bytes(to_graph_url(file_url), token, int(ttl_min))
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, header=header_row)


def get_driveitem_ids(token: str, file_url: str) -> tuple[str, str]:
    headers  = {"Authorization": f"Bearer {token}"}
    meta_url = to_graph_url(file_url).replace("/content", "")
    resp = requests.get(meta_url, headers=headers, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Could not resolve driveItem metadata: {resp.text[:300]}")
    meta     = resp.json()
    drive_id = (meta.get("parentReference", {}).get("driveId")
                or meta.get("remoteItem", {}).get("parentReference", {}).get("driveId"))
    item_id  = meta.get("id")
    if not drive_id or not item_id:
        raise RuntimeError(f"Could not extract driveId/itemId from: {meta}")
    return drive_id, item_id


def col_letter(n: int) -> str:
    """Convert 1-based column index to Excel letter (1→A, 27→AA)."""
    result = ""
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


def write_rows_to_sp(token: str, file_url: str, sheet_name: str, rows):
    """
    Write one OR MANY rows to a SharePoint-hosted .xlsx in a SINGLE Graph API call.
    Unchanged from v3 — behaviour-preserving.
    """
    if not rows:
        return
    if isinstance(rows, dict):
        rows = [rows]

    headers = {"Authorization": f"Bearer {token}",
               "Content-Type":  "application/json"}
    drive_id, item_id = get_driveitem_ids(token, file_url)
    base = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/workbook"

    # Ensure worksheet exists
    ws_list_resp = requests.get(f"{base}/worksheets", headers=headers, timeout=30)
    if not ws_list_resp.ok:
        raise RuntimeError(f"Could not list worksheets: {ws_list_resp.text[:300]}")
    existing = [s["name"] for s in ws_list_resp.json().get("value", [])]

    if sheet_name not in existing:
        create_resp = requests.post(
            f"{base}/worksheets", headers=headers,
            json={"name": sheet_name}, timeout=30)
        if not create_resp.ok:
            raise RuntimeError(f"Could not create sheet '{sheet_name}': {create_resp.text[:300]}")

    ws_base = f"{base}/worksheets/{sheet_name}"

    # Union of keys across rows
    col_order = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                col_order.append(k)

    hdr_resp = requests.get(f"{ws_base}/usedRange(valuesOnly=true)",
                             headers=headers, timeout=30)
    next_row_idx = 1
    if hdr_resp.ok:
        used   = hdr_resp.json()
        values = used.get("values", [])
        if values and values[0] and values[0][0] not in (None, ""):
            existing_headers = [str(v).strip() if v is not None else "" for v in values[0]]
            for col_name in col_order:
                if col_name not in existing_headers:
                    existing_headers.append(col_name)
                    new_col_idx = len(existing_headers)
                    cell_addr   = f"{col_letter(new_col_idx)}1"
                    requests.patch(f"{ws_base}/range(address='{cell_addr}')",
                                    headers=headers,
                                    json={"values": [[col_name]]}, timeout=30)
            col_order    = existing_headers
            next_row_idx = len(values) + 1
        else:
            addr = f"A1:{col_letter(len(col_order))}1"
            requests.patch(f"{ws_base}/range(address='{addr}')",
                            headers=headers,
                            json={"values": [list(col_order)]}, timeout=30)
            next_row_idx = 2
    else:
        addr = f"A1:{col_letter(len(col_order))}1"
        requests.patch(f"{ws_base}/range(address='{addr}')",
                        headers=headers,
                        json={"values": [list(col_order)]}, timeout=30)
        next_row_idx = 2

    data_matrix = [
        ["" if r.get(c) is None else str(r.get(c, "")) for c in col_order]
        for r in rows]
    last_col = col_letter(len(col_order))
    last_row_idx = next_row_idx + len(data_matrix) - 1
    addr = f"A{next_row_idx}:{last_col}{last_row_idx}"
    write_resp = requests.patch(f"{ws_base}/range(address='{addr}')",
                                 headers=headers,
                                 json={"values": data_matrix}, timeout=60)
    if not write_resp.ok:
        raise RuntimeError(
            f"Failed to write rows (HTTP {write_resp.status_code}): {write_resp.text[:300]}")
