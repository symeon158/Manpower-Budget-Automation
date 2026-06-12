"""Branded Alumil header."""
from __future__ import annotations

import streamlit as st


def render_header(last_loaded: str = "—") -> None:
    html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');
.block-container{{ padding-top:4.2rem !important; padding-left:1rem !important; padding-right:1rem !important; }}
.amb-header{{ position:relative; overflow:hidden; border-radius:18px; margin-bottom:1.4rem;
    border:1px solid rgba(255,255,255,.07);
    background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)),
                linear-gradient(135deg, #081426 0%, #0d1f39 38%, #163860 100%);
    box-shadow: 0 14px 38px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04); }}
.amb-header::before{{ content:''; position:absolute; left:0; top:0; bottom:0; width:5px;
    background:linear-gradient(180deg,#6fc2ff 0%,#2b7fc5 55%,#14538f 100%); }}
.amb-inner{{ position:relative; z-index:1; display:flex; align-items:center;
    justify-content:space-between; gap:24px; flex-wrap:wrap; padding:24px 28px 22px 32px; }}
.amb-left{{ display:flex; align-items:center; gap:18px; min-width:0; }}
.amb-logowrap{{ width:56px; height:56px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; border-radius:14px;
    background:linear-gradient(180deg, rgba(111,194,255,.14), rgba(111,194,255,.04));
    border:1px solid rgba(111,194,255,.14); box-shadow:inset 0 1px 0 rgba(255,255,255,.06); }}
.amb-logomark svg{{ width:34px; height:34px; }}
.amb-titles{{ display:flex; flex-direction:column; gap:4px; min-width:0; }}
.amb-company{{ font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:10px;
    letter-spacing:4px; text-transform:uppercase; color:rgba(111,194,255,.9); line-height:1; }}
.amb-title{{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:30px;
    line-height:1; letter-spacing:.4px; color:#ffffff; white-space:nowrap; }}
.amb-title span{{ color:#6fc2ff; }}
.amb-subtitle{{ font-family:'Barlow',sans-serif; font-weight:300; font-size:11px;
    letter-spacing:1.3px; text-transform:uppercase; color:rgba(255,255,255,.52); margin-top:2px; }}
.amb-right{{ display:flex; align-items:flex-end; justify-content:center; flex-direction:column;
    gap:8px; flex-shrink:0; }}
.amb-badge{{ display:inline-flex; align-items:center; gap:8px; min-height:32px; padding:6px 12px;
    border-radius:999px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
    font-family:'Barlow',sans-serif; font-size:11px; font-weight:500; color:rgba(255,255,255,.84);
    white-space:nowrap; backdrop-filter: blur(2px); }}
.amb-badge .dot{{ width:7px; height:7px; border-radius:50%; background:#3ddc84;
    box-shadow:0 0 8px rgba(61,220,132,.9); animation:ambpulse 2s ease-in-out infinite; flex-shrink:0; }}
.amb-badge .dot.grey{{ background:#8ea0b8; box-shadow:none; animation:none; }}
@keyframes ambpulse{{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(.82)}} }}
.amb-stripe{{ height:3px;
    background:linear-gradient(90deg,#1f6fb3 0%,#6fc2ff 28%,#1f6fb3 58%,transparent 100%); opacity:.72; }}
</style>
<div class='amb-header'>
<div class='amb-inner'>
<div class='amb-left'>
<div class='amb-logowrap'><div class='amb-logomark'>
<svg viewBox='0 0 48 48' fill='none' xmlns='http://www.w3.org/2000/svg'>
<polygon points='24,4 41,14 41,34 24,44 7,34 7,14' stroke='#6fc2ff' stroke-width='1.6' opacity='0.5'/>
<path d='M17 32L24 15L31 32' stroke='#6fc2ff' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>
<path d='M20 25H28' stroke='#6fc2ff' stroke-width='2' stroke-linecap='round'/>
</svg></div></div>
<div class='amb-titles'>
<div class='amb-company'>ALUMIL S.A.</div>
<div class='amb-title'>Manpower <span>Budget</span></div>
<div class='amb-subtitle'>Workforce Planning &amp; Analytics Platform</div>
</div></div>
<div class='amb-right'>
<div class='amb-badge'><span class='dot'></span>Microsoft 365 Connected</div>
<div class='amb-badge'><span class='dot grey'></span>Last sync&nbsp;&nbsp;{last_loaded}</div>
</div></div>
<div class='amb-stripe'></div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
