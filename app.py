"""
BVB Frauen — Performance Analytics
Streamlit App · run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
import os
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BVB Frauen · Performance Analytics",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# BVB theme
st.markdown("""
<style>
    /* BVB Yellow accents */
    [data-testid="stMetricValue"]     { color: #FDE000 !important; font-weight: 700; }
    [data-testid="stMetricDelta"]     { font-size: 0.85rem; }
    h1, h2, h3                        { color: #FDE000; }
    [data-testid="stSidebar"]         { background-color: #0f0f0f; }
    .stTabs [data-baseweb="tab"]      { color: #888; }
    .stTabs [aria-selected="true"]    { color: #FDE000; border-bottom: 2px solid #FDE000; }
    div[data-testid="metric-container"] { background: #111; border-radius: 10px; padding: 14px; border: 1px solid #1e1e1e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RAW_METRICS = {
    "cmj":       {"label": "CMJ Sprunghöhe", "unit": "cm",               "hib": True},
    "dj_rsi":    {"label": "DJ RSI",          "unit": "",                "hib": True},
    "t5":        {"label": "Sprint t5",       "unit": "s",               "hib": False},
    "t10":       {"label": "Sprint t10",      "unit": "s",               "hib": False},
    "t20":       {"label": "Sprint t20",      "unit": "s",               "hib": False},
    "t30":       {"label": "Sprint t30",      "unit": "s",               "hib": False},
    "agility":   {"label": "Agility",         "unit": "s",               "hib": False},
    "dribbling": {"label": "Dribbling",       "unit": "s",               "hib": False},
    "vo2max":    {"label": "VO2max",          "unit": "ml·kg⁻¹·min⁻¹",  "hib": True},
}
RADAR_METRICS = ["cmj", "dj_rsi", "t5", "t10", "t20", "agility", "dribbling", "vo2max"]
RADAR_LABELS  = ["CMJ", "DJ RSI", "t5", "t10", "t20", "Agility", "Dribbling", "VO2max"]

BVB_YELLOW = "#FDE000"
COLORS = ["#FDE000", "#60A5FA", "#4ADE80", "#FB923C", "#C084FC", "#F472B6"]

def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def cross_norm(df: pd.DataFrame, metric: str, sessions: list) -> list:
    """
    Normalise team-average raw values across sessions to a 75-128 scale
    so the radar shows meaningful differences between sessions.
    (Z-scores average to ~100 within each session so they can't be used here.)
    """
    info = RAW_METRICS.get(metric, {})
    hib  = info.get("hib", True)
    avgs = []
    for s in sessions:
        sess_df = df[(df["session"] == s) & df["cmj"].notna()]
        v = sess_df[metric].mean() if metric in sess_df.columns else np.nan
        avgs.append(v if pd.notna(v) else None)
    valid = [v for v in avgs if v is not None]
    if len(valid) < 2:
        return [100.0] * len(sessions)
    mn, mx = min(valid), max(valid)
    rng = mx - mn
    result = []
    for v in avgs:
        if v is None:
            result.append(85.0)
        elif rng == 0:
            result.append(100.0)
        else:
            # best session = 128, worst = 75
            norm = 75 + ((v - mn) / rng * 53) if hib else 75 + ((mx - v) / rng * 53)
            result.append(round(norm, 1))
    return result

SEED_DATA = [
    {"name":"Laura van der Laan","session":"Jan 26","dj_rsi":1.840,"cmj":29.8,"t5":1.02,"t10":1.88,"t20":3.28,"t30":4.65,"vo2max":52.53,"agility":16.07,"dribbling":10.63},
    {"name":"Mia Böger","session":"Jan 26","dj_rsi":1.817,"cmj":35.9,"t5":0.97,"t10":1.73,"t20":3.09,"t30":4.37,"vo2max":53.87,"agility":15.37,"dribbling":9.83},
    {"name":"Jenske Steenjwijk","session":"Jan 26","dj_rsi":1.711,"cmj":30.9,"t5":1.04,"t10":1.85,"t20":3.24,"t30":4.58,"vo2max":50.18,"agility":15.48,"dribbling":10.10},
    {"name":"Jasmin Jabbes","session":"Jan 26","dj_rsi":1.747,"cmj":33.4,"t5":1.08,"t10":1.85,"t20":3.19,"t30":4.46,"vo2max":60.93,"agility":15.34,"dribbling":9.88},
    {"name":"Noreen Günnewig","session":"Jan 26","dj_rsi":1.728,"cmj":32.3,"t5":1.12,"t10":1.93,"t20":3.46,"t30":4.80,"vo2max":51.86,"agility":15.89,"dribbling":10.26},
    {"name":"Nova Wicke","session":"Jan 26","dj_rsi":1.317,"cmj":28.9,"t5":1.03,"t10":1.82,"t20":3.19,"t30":4.47,"vo2max":44.46,"agility":15.65,"dribbling":10.43},
    {"name":"Lucy Wisniewski","session":"Jan 26","dj_rsi":1.623,"cmj":28.8,"t5":1.07,"t10":1.91,"t20":3.36,"t30":4.73,"vo2max":49.84,"agility":16.13,"dribbling":10.06},
    {"name":"Ronja Leubner","session":"Jan 26","dj_rsi":1.211,"cmj":35.6,"t5":1.03,"t10":1.82,"t20":3.20,"t30":4.56,"vo2max":51.86,"agility":15.38,"dribbling":10.33},
    {"name":"Nora Willeke","session":"Jan 26","dj_rsi":1.462,"cmj":32.8,"t5":1.06,"t10":1.85,"t20":3.25,"t30":4.57,"vo2max":57.90,"agility":15.44,"dribbling":9.80},
    {"name":"Celina Baum","session":"Jan 26","dj_rsi":1.511,"cmj":31.5,"t5":1.04,"t10":1.86,"t20":3.25,"t30":4.60,"vo2max":48.50,"agility":16.15,"dribbling":9.80},
    {"name":"Annika Enderle","session":"Jan 26","dj_rsi":1.671,"cmj":33.0,"t5":0.98,"t10":1.84,"t20":3.17,"t30":4.40,"vo2max":52.36,"agility":15.21,"dribbling":9.72},
    {"name":"Melanie Schuster","session":"Jan 26","dj_rsi":1.450,"cmj":29.0,"t5":1.12,"t10":1.99,"t20":3.44,"t30":4.84,"vo2max":49.17,"agility":15.83,"dribbling":9.87},
    {"name":"Caroline Blum","session":"Jan 26","dj_rsi":1.293,"cmj":30.8,"t5":1.09,"t10":2.03,"t20":3.52,"t30":4.97,"vo2max":49.50,"agility":16.14,"dribbling":10.63},
    {"name":"Paula Reimann","session":"Jan 26","dj_rsi":1.896,"cmj":29.9,"t5":1.11,"t10":1.96,"t20":3.43,"t30":4.84,"vo2max":51.18,"agility":16.37,"dribbling":9.91},
    {"name":"Yasu Woehrn","session":"Jan 26","dj_rsi":1.711,"cmj":35.5,"t5":1.05,"t10":1.89,"t20":3.34,"t30":4.69,"vo2max":50.85,"agility":15.38,"dribbling":9.66},
    {"name":"Laura van der Laan","session":"Mär 26","dj_rsi":1.798,"cmj":29.4,"t5":1.05,"t10":1.90,"t20":3.21,"t30":4.52,"vo2max":56.56,"agility":15.80,"dribbling":9.78},
    {"name":"Mia Böger","session":"Mär 26","dj_rsi":1.797,"cmj":37.2,"t5":1.01,"t10":1.71,"t20":3.01,"t30":4.26,"vo2max":None,"agility":14.83,"dribbling":9.60},
    {"name":"Jasmin Jabbes","session":"Mär 26","dj_rsi":1.763,"cmj":34.8,"t5":1.11,"t10":1.85,"t20":3.20,"t30":4.43,"vo2max":61.94,"agility":15.28,"dribbling":9.42},
    {"name":"Noreen Günnewig","session":"Mär 26","dj_rsi":1.784,"cmj":32.9,"t5":1.04,"t10":1.84,"t20":3.22,"t30":4.56,"vo2max":53.87,"agility":15.55,"dribbling":9.53},
    {"name":"Marina Scholz","session":"Mär 26","dj_rsi":2.129,"cmj":34.4,"t5":1.10,"t10":1.90,"t20":3.30,"t30":4.57,"vo2max":59.58,"agility":15.99,"dribbling":9.15},
    {"name":"Rita Schumacher","session":"Mär 26","dj_rsi":2.324,"cmj":39.0,"t5":0.97,"t10":1.68,"t20":3.00,"t30":4.19,"vo2max":54.88,"agility":14.82,"dribbling":8.65},
    {"name":"Sara Ito","session":"Mär 26","dj_rsi":1.023,"cmj":30.5,"t5":None,"t10":None,"t20":None,"t30":None,"vo2max":56.22,"agility":None,"dribbling":9.35},
    {"name":"Lucy Wisniewski","session":"Mär 26","dj_rsi":1.786,"cmj":31.4,"t5":1.13,"t10":1.91,"t20":3.32,"t30":4.65,"vo2max":56.22,"agility":16.05,"dribbling":9.97},
    {"name":"Ronja Leubner","session":"Mär 26","dj_rsi":1.487,"cmj":37.4,"t5":0.97,"t10":1.72,"t20":3.04,"t30":4.30,"vo2max":58.24,"agility":14.96,"dribbling":9.34},
    {"name":"Nora Willeke","session":"Mär 26","dj_rsi":1.768,"cmj":33.8,"t5":1.02,"t10":1.74,"t20":3.06,"t30":4.29,"vo2max":59.92,"agility":15.20,"dribbling":9.54},
    {"name":"Celina Baum","session":"Mär 26","dj_rsi":1.574,"cmj":32.6,"t5":1.07,"t10":1.86,"t20":3.25,"t30":4.56,"vo2max":58.24,"agility":16.00,"dribbling":9.28},
    {"name":"Annika Enderle","session":"Mär 26","dj_rsi":1.766,"cmj":34.3,"t5":0.99,"t10":1.81,"t20":3.16,"t30":4.42,"vo2max":54.88,"agility":15.23,"dribbling":9.10},
    {"name":"Frederike Kempe","session":"Mär 26","dj_rsi":2.238,"cmj":36.6,"t5":1.00,"t10":1.86,"t20":3.23,"t30":4.53,"vo2max":54.88,"agility":15.31,"dribbling":9.48},
    {"name":"Melanie Schuster","session":"Mär 26","dj_rsi":1.594,"cmj":34.3,"t5":1.09,"t10":1.89,"t20":3.26,"t30":4.53,"vo2max":52.19,"agility":15.77,"dribbling":9.84},
    {"name":"Caroline Blum","session":"Mär 26","dj_rsi":1.385,"cmj":33.7,"t5":1.12,"t10":1.91,"t20":3.42,"t30":4.81,"vo2max":50.18,"agility":15.89,"dribbling":9.90},
    {"name":"Paula Reimann","session":"Mär 26","dj_rsi":2.026,"cmj":31.5,"t5":1.07,"t10":1.80,"t20":3.29,"t30":4.63,"vo2max":53.87,"agility":15.76,"dribbling":9.38},
    {"name":"Yasu Woehrn","session":"Mär 26","dj_rsi":1.918,"cmj":43.3,"t5":1.02,"t10":1.81,"t20":3.18,"t30":4.44,"vo2max":58.91,"agility":15.30,"dribbling":9.32},
]

# ─────────────────────────────────────────────
# ENGINE (same logic as bvb_engine.py, adapted for Streamlit state)
# ─────────────────────────────────────────────

def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric, info in RAW_METRICS.items():
        z_col = f"{metric}_Z"
        df[z_col] = np.nan
        for sess in df["session"].unique():
            mask = df["session"] == sess
            vals = df.loc[mask, metric].dropna()
            if len(vals) < 2:
                continue
            m, sd = vals.mean(), vals.std()
            if sd == 0:
                continue
            sign = 1 if info["hib"] else -1
            df.loc[mask, z_col] = df.loc[mask, metric].apply(
                lambda v: round(100 + 10 * sign * (v - m) / sd, 1) if pd.notna(v) else np.nan
            )
    return df


def parse_excel(file, session_label: str) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    sheet = next(
        (s for s in xl.sheet_names if any(k in s.lower() for k in ["mär","mar","jan","sep","jul"])),
        xl.sheet_names[0]
    )
    raw = pd.read_excel(file, sheet_name=sheet, header=None)
    name_col, col_map = -1, {}
    for ri in range(min(5, len(raw))):
        for ci, cell in enumerate(raw.iloc[ri]):
            s = str(cell or "").lower().strip()
            if any(k in s for k in ["spieler", "name"]):          name_col = ci
            if "cmj" in s and any(k in s for k in ["best","höhe","jump"]): col_map["cmj"] = ci
            if "rsi" in s:                                         col_map["dj_rsi"] = ci
            if s == "t5" or s.startswith("t5 ") or s.startswith("t5("):   col_map["t5"] = ci
            if s == "t10" or s.startswith("t10 "):                col_map["t10"] = ci
            if s == "t20" or s.startswith("t20"):                 col_map["t20"] = ci
            if s == "t30" or s.startswith("t30"):                 col_map["t30"] = ci
            if ("agility" in s or "illinois" in s) and any(k in s for k in ["best","t1"]): col_map["agility"] = ci
            if "dribbling" in s and any(k in s for k in ["best","t1"]): col_map["dribbling"] = ci
            if "vo2" in s:                                         col_map["vo2max"] = ci
    if name_col == -1:
        return pd.DataFrame()
    records = []
    for _, row in raw.iloc[2:].iterrows():
        name = row.iloc[name_col]
        if not isinstance(name, str) or len(name.strip()) < 2:
            continue
        rec = {"name": name.strip(), "session": session_label}
        for key, ci in col_map.items():
            try:
                rec[key] = float(str(row.iloc[ci]).replace(",", "."))
            except:
                rec[key] = None
        for key in RAW_METRICS:
            if key not in rec:
                rec[key] = None
        if any(rec.get(k) is not None for k in RAW_METRICS):
            records.append(rec)
    return pd.DataFrame(records)


def get_df() -> pd.DataFrame:
    return st.session_state.df


def compute_injury_flags(df: pd.DataFrame, threshold: float = 8.0):
    sessions = sorted(df["session"].unique())
    if len(sessions) < 2:
        return []
    last, prev = sessions[-1], sessions[-2]
    flags = []
    for name in sorted(df[df["cmj"].notna()]["name"].unique()):
        j = df[(df["name"] == name) & (df["session"] == prev)]
        m = df[(df["name"] == name) & (df["session"] == last)]
        if j.empty or m.empty:
            continue
        j, m = j.iloc[0], m.iloc[0]
        drops, gains = [], []
        for metric, lbl in zip(RADAR_METRICS, RADAR_LABELS):
            zk = f"{metric}_Z"
            jz, mz = j.get(zk), m.get(zk)
            if pd.notna(jz) and pd.notna(mz):
                delta = mz - jz
                if delta < -threshold:
                    drops.append({"metric": lbl, "delta": round(delta, 1)})
                elif delta > threshold:
                    gains.append({"metric": lbl, "delta": round(delta, 1)})
        if drops:
            sev = "🔴 Hoch" if len(drops) >= 3 else "🟠 Mittel" if len(drops) >= 2 else "🟡 Niedrig"
            flags.append({"name": name, "drops": drops, "gains": gains, "severity": sev})
    return flags


def predict_next(df: pd.DataFrame, name: str) -> dict:
    hist = df[df["name"] == name].sort_values("session")
    if len(hist) < 2:
        return {}
    preds = {}
    xs = np.arange(len(hist))
    for metric, info in RAW_METRICS.items():
        vals = hist[metric].values
        valid_mask = ~pd.isna(vals)
        xi, yi = xs[valid_mask], vals[valid_mask].astype(float)
        if len(xi) < 2:
            preds[metric] = None
            continue
        slope, intercept, r, _, _ = stats.linregress(xi, yi)
        next_val = slope * len(hist) + intercept
        direction = "↑" if (slope > 0) == info["hib"] else "↓"
        confidence = "Hoch" if r**2 > 0.7 else "Mittel" if r**2 > 0.4 else "Niedrig"
        preds[metric] = {
            "value": round(next_val, 3),
            "slope": round(slope, 4),
            "r2": round(r**2, 3),
            "direction": direction,
            "confidence": confidence,
            "color": "#4ADE80" if direction == "↑" else "#F87171",
        }
    return preds


def compute_clusters(df: pd.DataFrame, session: str = None):
    if session is None:
        session = sorted(df["session"].unique())[-1]
    sess_df = df[(df["session"] == session) & df["cmj"].notna()].copy()
    feat_cols = ["cmj", "dj_rsi", "t5", "t20", "agility", "vo2max"]
    feat_df = sess_df[["name"] + feat_cols].dropna()
    if len(feat_df) < 4:
        return pd.DataFrame()
    X = feat_df[feat_cols].values.copy()
    for i, col in enumerate(feat_cols):
        if not RAW_METRICS[col]["hib"]:
            X[:, i] = -X[:, i]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    km = KMeans(n_clusters=min(4, len(feat_df)), random_state=42, n_init=10)
    labels = km.fit_predict(X_s)
    archetype_names = ["⚡ Explosiv-Kraft", "🏃 Schnelligkeit", "🫁 Ausdauer", "🎯 Allround"]
    archetype_colors = ["#FDE000", "#60A5FA", "#4ADE80", "#FB923C"]
    cluster_props = []
    for ci in range(km.n_clusters):
        mask = labels == ci
        if not mask.any():
            continue
        c = X_s[mask].mean(axis=0)
        power = (c[0] + c[1]) / 2
        speed = (c[2] + c[3] + c[4]) / 3
        endurance = c[5]
        spread = max(power, speed, endurance) - min(power, speed, endurance)
        if spread < 0.5:
            arch = 3
        elif power == max(power, speed, endurance):
            arch = 0
        elif speed == max(power, speed, endurance):
            arch = 1
        else:
            arch = 2
        cluster_props.append({"cluster_id": ci, "arch": arch})
    result = feat_df[["name"]].copy()
    result["cluster_id"] = labels
    result = result.merge(pd.DataFrame(cluster_props), on="cluster_id")
    result["archetype"] = result["arch"].map(lambda x: archetype_names[x])
    result["color"] = result["arch"].map(lambda x: archetype_colors[x])
    return result


def generate_commentary(df: pd.DataFrame, name: str) -> dict:
    sessions = sorted(df["session"].unique())
    last = sessions[-1]
    prev = sessions[-2] if len(sessions) > 1 else None
    last_rec = df[(df["name"] == name) & (df["session"] == last)]
    if last_rec.empty:
        return {}
    last_rec = last_rec.iloc[0]
    z_cols = [f"{m}_Z" for m in RADAR_METRICS]
    z_vals = [last_rec[zk] for zk in z_cols if pd.notna(last_rec.get(zk, np.nan))]
    avg_z = np.mean(z_vals) if z_vals else 100
    level = ("zur absoluten Spitzengruppe" if avg_z >= 115
             else "zur oberen Hälfte" if avg_z >= 108
             else "zum Teamdurchschnitt" if avg_z >= 95
             else "unter den Teamdurchschnitt")
    tone = ("herausragendes" if avg_z >= 115
            else "überdurchschnittliches" if avg_z >= 108
            else "solides" if avg_z >= 95
            else "entwicklungsfähiges")
    overall = f"{name} zeigt in der Session {last} ein {tone} Leistungsprofil und gehört {level} (Ø Z-Score: {avg_z:.1f})."
    team_avg = df[(df["session"] == last) & df["cmj"].notna()]
    strengths, weaknesses = [], []
    for metric, info in RAW_METRICS.items():
        zk = f"{metric}_Z"
        pz = last_rec.get(zk)
        if pd.isna(pz):
            continue
        if pz >= 110:
            strengths.append(f"{info['label']} (Z={pz:.0f})")
        elif pz < 92:
            weaknesses.append(f"{info['label']} (Z={pz:.0f})")
    improvements, declines = [], []
    if prev:
        prev_rec = df[(df["name"] == name) & (df["session"] == prev)]
        if not prev_rec.empty:
            prev_rec = prev_rec.iloc[0]
            for metric, info in RAW_METRICS.items():
                pv, mv = prev_rec.get(metric), last_rec.get(metric)
                if pd.isna(pv) or pd.isna(mv):
                    continue
                pct = ((mv - pv) / pv) * 100 if pv != 0 else 0
                improved = (pct > 0) == info["hib"]
                if abs(pct) > 3:
                    entry = f"{info['label']}: {pv:.2f} → {mv:.2f} ({'+' if pct > 0 else ''}{pct:.1f}%)"
                    (improvements if improved else declines).append(entry)
    flags = compute_injury_flags(df)
    flag = next((f for f in flags if f["name"] == name), None)
    injury_text = (
        f"⚠ {flag['severity']} — Rückgänge in: {', '.join(d['metric'] for d in flag['drops'])}. Belastungssteuerung überprüfen."
        if flag else "✓ Kein erhöhtes Verletzungsrisiko."
    )
    return {
        "overall": overall,
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "improvements": improvements[:4],
        "declines": declines[:4],
        "injury_risk": injury_text,
    }


# ─────────────────────────────────────────────
# CHART HELPERS (Plotly)
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111", plot_bgcolor="#111",
    font_color="#888", font_family="monospace",
    margin=dict(l=20, r=20, t=40, b=20),
)


def radar_chart(df: pd.DataFrame, names: list, sessions: list = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    all_sessions = sorted(df["session"].unique())
    if sessions is None:
        sessions = all_sessions
    palette = COLORS if len(names) > 1 else ["#FDE000", "#60A5FA", "#4ADE80"]
    for ni, name in enumerate(names):
        for si, sess in enumerate(sessions):
            rec = df[(df["name"] == name) & (df["session"] == sess)]
            if rec.empty:
                continue
            rec = rec.iloc[0]
            vals = [rec.get(f"{m}_Z", np.nan) for m in RADAR_METRICS]
            if all(pd.isna(v) for v in vals):
                continue
            vals_clean = [v if pd.notna(v) else 0 for v in vals]
            color = palette[ni % len(palette)] if len(names) > 1 else palette[si % len(palette)]
            label = f"{name} · {sess}" if len(names) > 1 else sess
            opacity = 1.0 if si == len(sessions) - 1 else 0.5
            fig.add_trace(go.Scatterpolar(
                r=vals_clean + [vals_clean[0]],
                theta=RADAR_LABELS + [RADAR_LABELS[0]],
                fill="toself",
                fillcolor=hex_rgba(color, 0.1),
                line=dict(color=color, width=2.5 if opacity == 1.0 else 1.5),
                opacity=opacity,
                name=label,
                hovertemplate="%{theta}: %{r:.1f}<extra>" + label + "</extra>",
            ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font_color="#FDE000", font_size=14),
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, range=[70, 135], showticklabels=False,
                            gridcolor="#1e1e1e", linecolor="#2a2a2a"),
            angularaxis=dict(gridcolor="#1e1e1e", linecolor="#2a2a2a",
                             tickfont=dict(color="#666", size=11)),
        ),
        legend=dict(font_color="#666", bgcolor="rgba(0,0,0,0)"),
        height=400,
    )
    return fig


def trend_chart(df: pd.DataFrame, name: str) -> go.Figure:
    hist = df[df["name"] == name].sort_values("session")
    sessions = hist["session"].tolist()
    metrics_to_plot = [
        ("cmj", "CMJ (cm)", "#FDE000"),
        ("vo2max", "VO2max", "#4ADE80"),
        ("t20", "Sprint t20 (s)", "#60A5FA"),
        ("agility", "Agility (s)", "#FB923C"),
        ("dj_rsi", "DJ RSI", "#C084FC"),
        ("dribbling", "Dribbling (s)", "#F472B6"),
    ]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[m[1] for m in metrics_to_plot],
                        vertical_spacing=0.18, horizontal_spacing=0.1)
    for idx, (metric, label, color) in enumerate(metrics_to_plot):
        row, col = idx // 3 + 1, idx % 3 + 1
        vals = hist[metric].values
        valid = [(i, v) for i, v in enumerate(vals) if pd.notna(v)]
        if not valid:
            continue
        xi, yi = zip(*valid)
        fig.add_trace(go.Scatter(
            x=[sessions[i] for i in xi], y=list(yi),
            mode="lines+markers+text",
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            text=[f"{v:.2f}" for v in yi],
            textposition="top center",
            textfont=dict(size=9, color=color),
            name=label, showlegend=False,
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ), row=row, col=col)
        if len(xi) >= 2:
            slope, intercept, _, _, _ = stats.linregress(list(xi), list(yi))
            trend_y = [slope * i + intercept for i in xi]
            fig.add_trace(go.Scatter(
                x=[sessions[i] for i in xi], y=trend_y,
                mode="lines", line=dict(color=color, width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)
        fig.update_xaxes(tickfont=dict(size=8, color="#555"), row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=8, color="#555"), row=row, col=col)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=dict(text=f"{name} — Leistungsentwicklung", font_color="#FDE000", font_size=13),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#666"
        ann.font.size = 10
    return fig


def comparison_bar(df: pd.DataFrame, name: str, session: str) -> go.Figure:
    rec = df[(df["name"] == name) & (df["session"] == session)]
    if rec.empty:
        return go.Figure()
    rec = rec.iloc[0]
    team = df[(df["session"] == session) & df["cmj"].notna()]
    labels, player_vals, team_vals = [], [], []
    for m in RADAR_METRICS:
        zk = f"{m}_Z"
        pz = rec.get(zk)
        tz = team[zk].mean() if zk in team.columns else np.nan
        if pd.notna(pz):
            labels.append(RAW_METRICS[m]["label"])
            player_vals.append(pz)
            team_vals.append(tz if pd.notna(tz) else 100)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=team_vals, name="Team Ø",
        orientation="h", marker_color="#333",
        hovertemplate="Team Ø: %{x:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=player_vals, name=name.split()[-1],
        orientation="h", marker_color="#FDE000",
        hovertemplate=f"{name}: %{{x:.1f}}<extra></extra>",
    ))
    fig.add_vline(x=100, line_color="#444", line_dash="dash", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        xaxis=dict(range=[70, 140], title="Z-Score (MW=100)", tickfont=dict(color="#555")),
        yaxis=dict(tickfont=dict(color="#888")),
        legend=dict(font_color="#666", bgcolor="rgba(0,0,0,0)"),
        height=320,
        title=dict(text=f"Vergleich mit Team Ø · {session}", font_color="#FDE000", font_size=13),
    )
    return fig


def ranked_bar_chart(df: pd.DataFrame, metric: str, sessions_to_show: list = None) -> go.Figure:
    """
    Ranked bar chart per metric — replicating the Säulendiagramme view.
    Players sorted by latest session value (descending for higher-is-better,
    ascending for lower-is-better). Team mean shown as horizontal reference lines.
    Players who did not participate show as 0 / are excluded.
    """
    info = RAW_METRICS[metric]
    all_sessions = sorted(df["session"].unique())
    if sessions_to_show is None:
        sessions_to_show = all_sessions[-2:] if len(all_sessions) >= 2 else all_sessions

    last_sess = sessions_to_show[-1]

    # Build player rows — only players who have data in at least one shown session
    player_data = []
    for name in sorted(df[df["cmj"].notna()]["name"].unique()):
        row = {"name": name}
        has_any = False
        for sess in sessions_to_show:
            rec = df[(df["name"] == name) & (df["session"] == sess)]
            v = rec.iloc[0][metric] if not rec.empty and pd.notna(rec.iloc[0].get(metric)) else None
            row[sess] = v
            if v is not None:
                has_any = True
        if has_any:
            player_data.append(row)

    if not player_data:
        return go.Figure()

    pdf = pd.DataFrame(player_data)

    # Sort by latest session value
    pdf["_sort"] = pdf[last_sess].fillna(-999 if info["hib"] else 999)
    pdf = pdf.sort_values("_sort", ascending=not info["hib"]).drop(columns="_sort")

    # Compute team means (only players with real data)
    means = {}
    for sess in sessions_to_show:
        col = df[(df["session"] == sess) & df[metric].notna()][metric]
        means[sess] = col.mean() if len(col) > 0 else None

    # Color palette per session
    sess_colors = {
        sessions_to_show[-1]: "#C0392B",   # dark red = latest (matches screenshot)
        sessions_to_show[0]:  "#E8A0A0",   # light pink = previous
    }
    if len(sessions_to_show) > 2:
        extra_colors = ["#60A5FA", "#4ADE80", "#FB923C"]
        for i, s in enumerate(sessions_to_show[:-2]):
            sess_colors[s] = extra_colors[i % len(extra_colors)]

    fig = go.Figure()

    # Bars — one group per session, in chronological order
    for sess in sessions_to_show:
        vals = [pdf[sess].iloc[i] if pd.notna(pdf[sess].iloc[i]) else 0
                for i in range(len(pdf))]
        # Replace 0 with None so bar is absent (not participated)
        bar_vals = [v if v > 0 else None for v in vals]
        fig.add_trace(go.Bar(
            name=sess,
            x=pdf["name"].tolist(),
            y=bar_vals,
            marker_color=sess_colors.get(sess, "#888"),
            marker_opacity=0.85,
            text=[f"{v:.2f}" if v else "" for v in bar_vals],
            textposition="outside",
            textfont=dict(size=8, color="#aaa"),
            hovertemplate=f"<b>%{{x}}</b><br>{sess}: %{{y:.2f}} {info['unit']}<extra></extra>",
        ))

    # Mean reference lines — one per session
    line_colors = {
        sessions_to_show[-1]: "#000000",   # black = latest mean (matches screenshot)
        sessions_to_show[0]:  "#888888",   # grey = previous mean
    }
    for sess in sessions_to_show:
        m = means.get(sess)
        if m is not None:
            fig.add_hline(
                y=m,
                line_color=line_colors.get(sess, "#555"),
                line_width=1.5,
                line_dash="solid",
                annotation_text=f"MW {sess}: {m:.2f}",
                annotation_font_color=line_colors.get(sess, "#555"),
                annotation_font_size=10,
                annotation_position="right",
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=460,
        title=dict(
            text=f"{info['label']}  <span style='font-size:13px;color:#555'>{info['unit']}</span>",
            font=dict(color="#FDE000", size=15),
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10, color="#777"),
            gridcolor="#1a1a1a",
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#555"),
            gridcolor="#1a1a1a",
            zeroline=False,
        ),
        legend=dict(
            font_color="#888",
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig


def team_radar_chart(df: pd.DataFrame) -> go.Figure:
    sessions = sorted(df["session"].unique())
    fig = go.Figure()
    palette = ["#60A5FA", "#FDE000", "#4ADE80", "#FB923C"]

    for si, sess in enumerate(sessions):
        # Use crossNorm so sessions are visually distinguishable.
        # Direct Z-score averages are always ~100 (by definition of Z),
        # which makes every session draw the same circle.
        vals = [cross_norm(df, m, sessions)[si] for m in RADAR_METRICS]
        # Build hover showing actual raw team averages
        sess_df = df[(df["session"] == sess) & df["cmj"].notna()]
        hover_vals = []
        for m in RADAR_METRICS:
            raw_avg = sess_df[m].mean() if m in sess_df.columns else None
            unit = RAW_METRICS[m]["unit"]
            hover_vals.append(
                f"{RAW_METRICS[m]['label']}: {raw_avg:.2f} {unit}"
                if raw_avg is not None and pd.notna(raw_avg) else
                f"{RAW_METRICS[m]['label']}: —"
            )
        color = palette[si % len(palette)]
        lw = 2.5 if si == len(sessions) - 1 else 1.5
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=RADAR_LABELS + [RADAR_LABELS[0]],
            fill="toself",
            fillcolor=hex_rgba(color, 0.12 if si == len(sessions)-1 else 0.06),
            line=dict(color=color, width=lw),
            name=sess,
            customdata=(hover_vals + [hover_vals[0]]),
            hovertemplate="%{customdata}<extra>" + sess + "</extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(
                visible=True, range=[65, 135],
                showticklabels=False,
                gridcolor="#222", linecolor="#2a2a2a",
            ),
            angularaxis=dict(
                gridcolor="#1e1e1e", linecolor="#2a2a2a",
                tickfont=dict(color="#666", size=11),
            ),
        ),
        legend=dict(font_color="#666", bgcolor="rgba(0,0,0,0)"),
        height=350,
        title=dict(
            text="Team Durchschnitt · Leistungsvergleich",
            font_color="#FDE000", font_size=13,
        ),
    )
    return fig


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf(df: pd.DataFrame, name: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            title=f"BVB Report – {name}")

    styles = getSampleStyleSheet()
    Y = colors.HexColor("#FDE000")
    G = colors.HexColor("#4ADE80")
    R = colors.HexColor("#F87171")
    DARK = colors.HexColor("#111111")
    DGRAY = colors.HexColor("#1a1a1a")
    MGRAY = colors.HexColor("#888888")
    LGRAY = colors.HexColor("#cccccc")

    def sty(size, color=LGRAY, bold=False, align=TA_LEFT, space_after=4):
        return ParagraphStyle("x", parent=styles["Normal"],
                               fontSize=size, textColor=color,
                               fontName="Helvetica-Bold" if bold else "Helvetica",
                               spaceAfter=space_after, leading=size*1.5, alignment=align)

    story = []
    sessions = sorted(df["session"].unique())
    last = sessions[-1]
    commentary = generate_commentary(df, name)
    preds = predict_next(df, name)
    clusters = compute_clusters(df)
    cluster = clusters[clusters["name"] == name].iloc[0] if not clusters.empty and name in clusters["name"].values else None

    # ── Header ──
    story.append(Paragraph("BVB FRAUEN · LEISTUNGSDIAGNOSTIK · " + last, sty(8, MGRAY)))
    story.append(Paragraph(name.upper(), sty(20, Y, bold=True, space_after=3)))
    if cluster is not None:
        story.append(Paragraph(f"Athletenprofil: {cluster['archetype']}", sty(9, MGRAY)))
    story.append(HRFlowable(width="100%", thickness=1, color=Y, spaceAfter=12))

    # ── Overall ──
    if commentary.get("overall"):
        story.append(Paragraph("GESAMTBEWERTUNG", sty(9, Y, bold=True, space_after=6)))
        story.append(Paragraph(commentary["overall"], sty(9, LGRAY, space_after=10)))

    # ── Radar chart (matplotlib) ──
    story.append(Paragraph("LEISTUNGSPROFIL (Z-Score Radar)", sty(9, Y, bold=True, space_after=6)))

    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#111", "font.family": "monospace"})
    angles = [n / float(len(RADAR_LABELS)) * 2 * np.pi for n in range(len(RADAR_LABELS))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), facecolor="#111")
    ax.set_facecolor("#111")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(70, 130)
    ax.set_rticks([80, 100, 120])
    ax.set_yticklabels(["80", "100", "120"], fontsize=7, color="#333")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=8, color="#777")
    ax.grid(color="#1e1e1e", linewidth=0.6)

    clrs = ["#FDE000", "#60A5FA", "#4ADE80"]
    for si, sess in enumerate(sessions[-3:]):
        rec = df[(df["name"] == name) & (df["session"] == sess)]
        if rec.empty:
            continue
        rec = rec.iloc[0]
        vals = [rec.get(f"{m}_Z", 100) or 100 for m in RADAR_METRICS]
        vals += vals[:1]
        c = clrs[si % len(clrs)]
        lw = 2.5 if sess == sessions[-1] else 1.5
        ax.plot(angles, vals, color=c, linewidth=lw, label=sess)
        ax.fill(angles, vals, color=c, alpha=0.08 if sess == sessions[-1] else 0.03)

    # Team average
    team_rec = df[(df["session"] == last) & df["cmj"].notna()]
    ta_vals = [team_rec[f"{m}_Z"].mean() if f"{m}_Z" in team_rec.columns else 100 for m in RADAR_METRICS]
    ta_vals += ta_vals[:1]
    ax.plot(angles, ta_vals, color="#333", linewidth=1, linestyle="--", label="Team Ø")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7,
              labelcolor="#666", framealpha=0)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=130, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    img_buf.seek(0)

    from reportlab.platypus import Image
    radar_img = Image(img_buf, width=7*cm, height=7*cm)

    # ── Metrics table ──
    header_row = ["Metrik", "Einheit"] + [s for s in sessions] + ["Team Ø"]
    tbl_data = [header_row]
    for metric, info in RAW_METRICS.items():
        row = [info["label"], info["unit"]]
        for sess in sessions:
            rec = df[(df["name"] == name) & (df["session"] == sess)]
            v = rec.iloc[0][metric] if not rec.empty and pd.notna(rec.iloc[0].get(metric)) else None
            row.append(f"{v:.2f}" if v is not None else "—")
        ta = team_rec[metric].mean() if metric in team_rec.columns else None
        row.append(f"{ta:.2f}" if ta is not None else "—")
        tbl_data.append(row)

    cw = [4*cm, 2*cm] + [2*cm]*len(sessions) + [2*cm]
    metrics_tbl = Table(tbl_data, colWidths=cw)
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DGRAY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), Y),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#111"), colors.HexColor("#161616")]),
        ("TEXTCOLOR",  (0, 1), (-1, -1), LGRAY),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    # Layout radar + table side by side
    side_table = Table([[radar_img, metrics_tbl]], colWidths=[7.5*cm, 10*cm])
    side_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(side_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Strengths / Weaknesses ──
    story.append(Paragraph("STÄRKEN & ENTWICKLUNGSFELDER", sty(9, Y, bold=True, space_after=6)))
    sw_data = [[Paragraph("✅ STÄRKEN", sty(8, G, bold=True)),
                Paragraph("⚠ ENTWICKLUNGSFELDER", sty(8, R, bold=True))]]
    strengths = commentary.get("strengths", [])
    weaknesses = commentary.get("weaknesses", [])
    for i in range(max(len(strengths), len(weaknesses), 1)):
        s = Paragraph(f"• {strengths[i]}", sty(8, G)) if i < len(strengths) else Paragraph("", sty(8))
        w = Paragraph(f"• {weaknesses[i]}", sty(8, R)) if i < len(weaknesses) else Paragraph("", sty(8))
        sw_data.append([s, w])
    sw_tbl = Table(sw_data, colWidths=[9*cm, 9*cm])
    sw_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#0d1a0d")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#1a0d0d")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#0a150a")),
        ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#150a0a")),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sw_tbl)
    story.append(Spacer(1, 0.3*cm))

    # ── Veränderungen ──
    if commentary.get("improvements") or commentary.get("declines"):
        story.append(Paragraph("VERÄNDERUNGEN", sty(9, Y, bold=True, space_after=6)))
        chg_data = [[Paragraph("↑ Verbesserungen", sty(8, G, bold=True)),
                     Paragraph("↓ Rückgänge", sty(8, R, bold=True))]]
        imp = commentary.get("improvements", [])
        dec = commentary.get("declines", [])
        for i in range(max(len(imp), len(dec), 1)):
            a = Paragraph(f"• {imp[i]}", sty(7.5, G)) if i < len(imp) else Paragraph("", sty(8))
            b = Paragraph(f"• {dec[i]}", sty(7.5, R)) if i < len(dec) else Paragraph("", sty(8))
            chg_data.append([a, b])
        chg_tbl = Table(chg_data, colWidths=[9*cm, 9*cm])
        chg_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0a150a")),
            ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#150a0a")),
        ]))
        story.append(chg_tbl)
        story.append(Spacer(1, 0.3*cm))

    # ── Injury Risk ──
    ir = commentary.get("injury_risk", "")
    story.append(Paragraph("VERLETZUNGSRISIKO", sty(9, Y, bold=True, space_after=6)))
    story.append(Paragraph(ir, sty(9, R if "⚠" in ir else G, space_after=10)))

    # ── Predictions ──
    story.append(Paragraph("PROGNOSE NÄCHSTE SESSION", sty(9, Y, bold=True, space_after=4)))
    story.append(Paragraph("Lineare Regression über alle verfügbaren Messpunkte:", sty(8, MGRAY, space_after=6)))
    pred_data = [["Metrik", "Prognose", "Trend", "R²", "Konfidenz"]]
    for metric, info in RAW_METRICS.items():
        p = preds.get(metric)
        if not p or p["value"] is None:
            continue
        pred_data.append([
            info["label"],
            f"{p['value']:.3f} {info['unit']}",
            p["direction"],
            f"{p['r2']:.3f}",
            p["confidence"],
        ])
    pred_tbl = Table(pred_data, colWidths=[4*cm, 3.5*cm, 1.5*cm, 2*cm, 2.5*cm])
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DGRAY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), Y),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#111"), colors.HexColor("#161616")]),
        ("TEXTCOLOR",  (0, 1), (-1, -1), LGRAY),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(pred_tbl)

    # ── Footer ──
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#2a2a2a"), spaceAfter=5))
    story.append(Paragraph(
        f"BVB Frauen · Performance Analytics · {last} · Automatisch generiert",
        sty(7, colors.HexColor("#444"), align=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    df_raw = pd.DataFrame(SEED_DATA)
    st.session_state.df = compute_z_scores(df_raw)

# Always read fresh from session_state — ensures post-upload reruns pick up new data
df       = get_df()
sessions  = sorted(df["session"].unique().tolist())
players   = sorted(df[df["cmj"].notna()]["name"].unique().tolist())
last_sess = sessions[-1]  if sessions  else None
prev_sess = sessions[-2]  if len(sessions) > 1 else None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🟡 BVB Analytics")
    st.caption("1. Mannschaft · Saison 25/26")
    st.divider()

    st.markdown("**Sessions**")
    for s in sessions:
        n = len(df[(df["session"] == s) & df["cmj"].notna()])
        st.caption(f"• {s} — {n} Spielerinnen")

    st.divider()
    st.markdown("**Neue Session laden**")
    new_sess_label = st.text_input("Session-Name", placeholder='z.B. "Jun 26"')
    uploaded = st.file_uploader("Messprotokoll Excel", type=["xlsx", "xls"])
    if st.button("📥 Laden", disabled=not (uploaded and new_sess_label.strip())):
        with st.spinner("Verarbeite..."):
            parsed = parse_excel(uploaded, new_sess_label.strip())
            if parsed.empty:
                st.error("Keine Daten erkannt. Prüfe Spaltenbezeichnungen.")
            else:
                existing = st.session_state.df[st.session_state.df["session"] != new_sess_label.strip()]
                combined = pd.concat([existing, parsed], ignore_index=True)
                st.session_state.df = compute_z_scores(combined)
                st.success(f"✓ {len(parsed)} Spielerinnen geladen")
                st.rerun()

    st.divider()
    # Remove uploaded (non-seed) sessions
    removable = [s for s in sessions if s not in {"Jan 26", "Mär 26"}]
    if removable:
        to_remove = st.selectbox("Session entfernen", ["—"] + removable)
        if st.button("🗑 Entfernen") and to_remove != "—":
            st.session_state.df = st.session_state.df[st.session_state.df["session"] != to_remove]
            st.session_state.df = compute_z_scores(st.session_state.df)
            st.rerun()

    st.divider()
    st.caption(f"v1.0 · {len(sessions)} Sessions · {len(players)} Spielerinnen")

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_overview, tab_player, tab_compare, tab_ranking, tab_saeulen, tab_flags, tab_pdf = st.tabs([
    "📊 Überblick",
    "👤 Spielerin",
    "⚡ Vergleich",
    "🏆 Rankings",
    "📊 Säulendiagramme",
    "🚨 Verletzungsrisiko",
    "📄 PDF Reports",
])

# ══════════════════════════════════════════════
# ÜBERBLICK
# ══════════════════════════════════════════════
with tab_overview:
    st.markdown(f"### Saison 25/26 · {last_sess}")

    # KPIs
    cols = st.columns(4)
    metrics_kpi = [
        ("Spielerinnen", len(df[(df["session"] == last_sess) & df["cmj"].notna()]), ""),
        ("Ø CMJ", df[(df["session"] == last_sess)]["cmj"].mean(), "cm"),
        ("Ø VO2max", df[(df["session"] == last_sess)]["vo2max"].mean(), ""),
        ("Ø Sprint t20", df[(df["session"] == last_sess)]["t20"].mean(), "s"),
    ]
    for i, (label, value, unit) in enumerate(metrics_kpi):
        with cols[i]:
            delta = None
            if prev_sess and label != "Spielerinnen":
                key = {"Ø CMJ": "cmj", "Ø VO2max": "vo2max", "Ø Sprint t20": "t20"}[label]
                prev_val = df[df["session"] == prev_sess][key].mean()
                delta_raw = value - prev_val
                delta = f"{'+' if delta_raw > 0 else ''}{delta_raw:.2f} seit {prev_sess}"
            st.metric(
                label=label,
                value=f"{value:.1f} {unit}" if isinstance(value, float) else str(value),
                delta=delta,
                delta_color="inverse" if label == "Ø Sprint t20" else "normal",
            )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(team_radar_chart(df), use_container_width=True)
    with col2:
        st.markdown("#### Meistverbesserte")
        if prev_sess:
            improved = []
            for name in players:
                j = df[(df["name"] == name) & (df["session"] == prev_sess)]
                m = df[(df["name"] == name) & (df["session"] == last_sess)]
                if j.empty or m.empty:
                    continue
                j, m = j.iloc[0], m.iloc[0]
                deltas = []
                for met in RADAR_METRICS:
                    zk = f"{met}_Z"
                    jz, mz = j.get(zk), m.get(zk)
                    if pd.notna(jz) and pd.notna(mz):
                        deltas.append(mz - jz)
                if deltas:
                    improved.append({"Spielerin": name, "Ø Δ Z": round(np.mean(deltas), 2)})
            if improved:
                imp_df = pd.DataFrame(improved).sort_values("Ø Δ Z", ascending=False).head(10)
                imp_df.index = range(1, len(imp_df) + 1)
                st.dataframe(
                    imp_df.style.background_gradient(cmap="RdYlGn", subset=["Ø Δ Z"]),
                    use_container_width=True
                )
        else:
            st.info("Lade eine zweite Session um Vergleiche zu sehen.")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Team Entwicklung")
        if prev_sess:
            trend_metrics = ["cmj", "dj_rsi", "t5", "t20", "agility", "vo2max"]
            trend_labels  = ["CMJ", "DJ RSI", "t5", "t20", "Agility", "VO2max"]
            deltas = []
            for m, l in zip(trend_metrics, trend_labels):
                v_last = df[df["session"] == last_sess][m].mean()
                v_prev = df[df["session"] == prev_sess][m].mean()
                if pd.notna(v_last) and pd.notna(v_prev) and v_prev != 0:
                    pct = ((v_last - v_prev) / v_prev) * 100
                    if not RAW_METRICS[m]["hib"]:
                        pct = -pct
                    deltas.append({"Metrik": l, "Verbesserung %": round(pct, 2)})
            if deltas:
                td = pd.DataFrame(deltas)
                fig_trend = px.bar(td, x="Metrik", y="Verbesserung %",
                                   color="Verbesserung %",
                                   color_continuous_scale=["#F87171", "#F87171", "#4ADE80", "#4ADE80"],
                                   color_continuous_midpoint=0)
                fig_trend.update_layout(**PLOTLY_LAYOUT, height=280,
                                        coloraxis_showscale=False,
                                        xaxis=dict(tickfont=dict(color="#666")),
                                        yaxis=dict(tickfont=dict(color="#555")))
                fig_trend.add_hline(y=0, line_color="#333")
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Zwei Sessions benötigt.")

    with col4:
        st.markdown("#### Athletenprofile")
        clusters = compute_clusters(df, last_sess)
        if not clusters.empty:
            for arch in clusters["archetype"].unique():
                group = clusters[clusters["archetype"] == arch]
                color = group.iloc[0]["color"]
                with st.expander(f"{arch} ({len(group)})", expanded=True):
                    for _, row in group.iterrows():
                        st.caption(f"• {row['name']}")

# ══════════════════════════════════════════════
# SPIELERIN DETAIL
# ══════════════════════════════════════════════
with tab_player:
    selected_player = st.selectbox("Spielerin auswählen", players, key="sel_player")
    selected_session = st.selectbox("Session", sessions, index=len(sessions)-1, key="sel_sess")

    if selected_player:
        st.divider()
        rec = df[(df["name"] == selected_player) & (df["session"] == selected_session)]
        if rec.empty:
            st.warning(f"Keine Daten für {selected_player} in {selected_session}")
        else:
            rec = rec.iloc[0]
            cluster_info = compute_clusters(df, last_sess)
            cl = cluster_info[cluster_info["name"] == selected_player]

            col_head, col_btn = st.columns([4, 1])
            with col_head:
                st.markdown(f"### {selected_player}")
                st.caption(f"Session: {selected_session} · Sessions gesamt: {', '.join(sessions)}")
                if not cl.empty:
                    arch = cl.iloc[0]["archetype"]
                    color = cl.iloc[0]["color"]
                    st.markdown(f'<span style="background:{color}22;color:{color};border:1px solid {color}44;padding:3px 12px;border-radius:12px;font-size:12px">{arch}</span>', unsafe_allow_html=True)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                player_sessions = sorted(df[df["name"] == selected_player]["session"].unique())
                st.plotly_chart(radar_chart(df, [selected_player], player_sessions,
                                           title=f"{selected_player} · Radar"), use_container_width=True)
            with c2:
                st.plotly_chart(comparison_bar(df, selected_player, selected_session),
                                use_container_width=True)

            st.plotly_chart(trend_chart(df, selected_player), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### Rohdaten")
                hist = df[df["name"] == selected_player].sort_values("session")
                raw_rows = []
                for metric, info in RAW_METRICS.items():
                    row = {"Metrik": info["label"], "Einheit": info["unit"]}
                    for s in sessions:
                        r = df[(df["name"] == selected_player) & (df["session"] == s)]
                        row[s] = round(r.iloc[0][metric], 3) if not r.empty and pd.notna(r.iloc[0].get(metric)) else None
                    ta = df[df["session"] == last_sess][metric].mean()
                    row["Team Ø"] = round(ta, 3) if pd.notna(ta) else None
                    raw_rows.append(row)
                raw_df = pd.DataFrame(raw_rows).set_index("Metrik")
                st.dataframe(raw_df, use_container_width=True)

            with c4:
                st.markdown("#### Auswertung")
                co = generate_commentary(df, selected_player)
                if co.get("overall"):
                    st.info(co["overall"])
                if co.get("strengths"):
                    st.success("**Stärken:** " + " · ".join(co["strengths"]))
                if co.get("weaknesses"):
                    st.warning("**Entwicklungsfelder:** " + " · ".join(co["weaknesses"]))
                ir = co.get("injury_risk", "")
                if "⚠" in ir:
                    st.error(ir)
                else:
                    st.success(ir)

            st.markdown("#### Prognose nächste Session")
            preds = predict_next(df, selected_player)
            pred_rows = []
            for metric, info in RAW_METRICS.items():
                p = preds.get(metric)
                if p and p["value"] is not None:
                    pred_rows.append({
                        "Metrik": info["label"],
                        "Prognose": f"{p['value']:.3f} {info['unit']}",
                        "Trend": p["direction"],
                        "R²": p["r2"],
                        "Konfidenz": p["confidence"],
                    })
            if pred_rows:
                pred_df = pd.DataFrame(pred_rows).set_index("Metrik")
                st.dataframe(
                    pred_df.style.applymap(
                        lambda v: "color: #4ADE80" if v == "↑" else ("color: #F87171" if v == "↓" else ""),
                        subset=["Trend"]
                    ),
                    use_container_width=True
                )

# ══════════════════════════════════════════════
# VERGLEICH
# ══════════════════════════════════════════════
with tab_compare:
    st.markdown("### Spielerinnen vergleichen")
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_players = st.multiselect(
            "Spielerinnen auswählen (max. 6)",
            players, max_selections=6,
            default=players[:3] if len(players) >= 3 else players,
        )
    with col_sel2:
        cmp_session = st.selectbox("Session", sessions, index=len(sessions)-1, key="cmp_sess")

    if len(selected_players) < 2:
        st.info("Mindestens 2 Spielerinnen auswählen.")
    else:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(radar_chart(df, selected_players, [cmp_session],
                                       title="Radar Vergleich"), use_container_width=True)
        with c2:
            cmp_metric = st.selectbox("Metrik", list(RAW_METRICS.keys()),
                                      format_func=lambda x: RAW_METRICS[x]["label"])
            cmp_data = []
            for name in selected_players:
                r = df[(df["name"] == name) & (df["session"] == cmp_session)]
                if not r.empty and pd.notna(r.iloc[0].get(cmp_metric)):
                    cmp_data.append({"Spielerin": name.split()[-1],
                                     "Wert": r.iloc[0][cmp_metric], "Voller Name": name})
            if cmp_data:
                cmp_df = pd.DataFrame(cmp_data)
                lb = not RAW_METRICS[cmp_metric]["hib"]
                cmp_df = cmp_df.sort_values("Wert", ascending=lb)
                fig_cmp = px.bar(cmp_df, x="Spielerin", y="Wert",
                                 color="Spielerin",
                                 color_discrete_sequence=COLORS,
                                 title=RAW_METRICS[cmp_metric]["label"],
                                 hover_data={"Voller Name": True})
                fig_cmp.update_layout(**PLOTLY_LAYOUT, height=330,
                                      showlegend=False,
                                      xaxis=dict(tickfont=dict(color="#666")),
                                      yaxis=dict(tickfont=dict(color="#555")))
                st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("#### Detaillierter Vergleich")
        rows = []
        for metric, info in RAW_METRICS.items():
            row = {"Metrik": info["label"]}
            vals = {}
            for name in selected_players:
                r = df[(df["name"] == name) & (df["session"] == cmp_session)]
                v = r.iloc[0][metric] if not r.empty and pd.notna(r.iloc[0].get(metric)) else None
                row[name.split()[-1]] = round(v, 3) if v is not None else None
                if v is not None:
                    vals[name] = v
            if vals:
                best = (max if info["hib"] else min)(vals.values())
                row["★ Best"] = [k.split()[-1] for k, v in vals.items() if v == best][0]
            rows.append(row)
        det_df = pd.DataFrame(rows).set_index("Metrik")
        st.dataframe(det_df, use_container_width=True)

        if len(sessions) > 1:
            st.markdown("#### Zeitverlauf")
            tl_metric = st.selectbox("Metrik", list(RAW_METRICS.keys()),
                                     format_func=lambda x: RAW_METRICS[x]["label"],
                                     key="tl_met")
            fig_tl = go.Figure()
            for i, name in enumerate(selected_players):
                tl_data = []
                for s in sessions:
                    r = df[(df["name"] == name) & (df["session"] == s)]
                    v = r.iloc[0][tl_metric] if not r.empty and pd.notna(r.iloc[0].get(tl_metric)) else None
                    tl_data.append(v)
                fig_tl.add_trace(go.Scatter(
                    x=sessions, y=tl_data,
                    mode="lines+markers", name=name,
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                    marker=dict(size=8),
                    connectgaps=True,
                ))
            fig_tl.update_layout(**PLOTLY_LAYOUT, height=280,
                                 xaxis=dict(tickfont=dict(color="#666")),
                                 yaxis=dict(tickfont=dict(color="#555")),
                                 legend=dict(font_color="#666"))
            st.plotly_chart(fig_tl, use_container_width=True)

# ══════════════════════════════════════════════
# RANKINGS
# ══════════════════════════════════════════════
with tab_ranking:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rk_metric = st.selectbox("Metrik", list(RAW_METRICS.keys()),
                                  format_func=lambda x: RAW_METRICS[x]["label"],
                                  key="rk_met")
    with col_r2:
        rk_session = st.selectbox("Session", sessions, index=len(sessions)-1, key="rk_sess")

    info = RAW_METRICS[rk_metric]
    rk_df = df[(df["session"] == rk_session) & df[rk_metric].notna()].copy()
    rk_df = rk_df.sort_values(rk_metric, ascending=not info["hib"])
    rk_df["#"] = range(1, len(rk_df) + 1)
    rk_df["Metrik"] = rk_df[rk_metric].round(3)

    # Add delta vs previous session
    if prev_sess:
        deltas = []
        for _, row in rk_df.iterrows():
            pr = df[(df["name"] == row["name"]) & (df["session"] == rk_session)]
            pp = df[(df["name"] == row["name"]) & (df["session"] == prev_sess)]
            if not pr.empty and not pp.empty:
                d = pr.iloc[0][rk_metric] - pp.iloc[0][rk_metric]
                deltas.append(round(d, 3))
            else:
                deltas.append(None)
        rk_df["Δ vs prev"] = deltas

    display_cols = ["#", "name", "Metrik"] + (["Δ vs prev"] if prev_sess else [])
    rk_display = rk_df[display_cols].rename(columns={"name": "Spielerin"}).set_index("#")
    st.dataframe(rk_display, use_container_width=True, height=600)


# ══════════════════════════════════════════════
# SÄULENDIAGRAMME
# ══════════════════════════════════════════════
with tab_saeulen:
    st.markdown("### Säulendiagramme — Ranked Bar Charts")
    st.caption("Spielerinnen sortiert nach bestem Wert in der neuesten Session. Horizontale Linien = Mannschaftsmittelwert.")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        saeulen_sessions = st.multiselect(
            "Sessions vergleichen",
            sessions,
            default=sessions[-2:] if len(sessions) >= 2 else sessions,
            key="saeul_sess",
        )
    with col_s2:
        saeulen_metric = st.selectbox(
            "Metrik",
            list(RAW_METRICS.keys()),
            format_func=lambda x: RAW_METRICS[x]["label"],
            key="saeul_met",
        )

    if not saeulen_sessions:
        st.info("Mindestens eine Session auswählen.")
    else:
        st.plotly_chart(
            ranked_bar_chart(df, saeulen_metric, saeulen_sessions),
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### Alle Metriken auf einen Blick")
        st.caption("Klicke eine Metrik an um den vollständigen Chart zu sehen.")

        # Grid of mini charts — 2 per row
        metrics_list = list(RAW_METRICS.keys())
        for i in range(0, len(metrics_list), 2):
            col_a, col_b = st.columns(2)
            with col_a:
                m = metrics_list[i]
                fig_mini = ranked_bar_chart(df, m, saeulen_sessions)
                fig_mini.update_layout(height=320, title_font_size=12,
                                       showlegend=False,
                                       xaxis_tickfont_size=7,
                                       margin=dict(l=10, r=10, t=40, b=60))
                st.plotly_chart(fig_mini, use_container_width=True)
            with col_b:
                if i + 1 < len(metrics_list):
                    m2 = metrics_list[i + 1]
                    fig_mini2 = ranked_bar_chart(df, m2, saeulen_sessions)
                    fig_mini2.update_layout(height=320, title_font_size=12,
                                            showlegend=False,
                                            xaxis_tickfont_size=7,
                                            margin=dict(l=10, r=10, t=40, b=60))
                    st.plotly_chart(fig_mini2, use_container_width=True)

# ══════════════════════════════════════════════
# VERLETZUNGSRISIKO
# ══════════════════════════════════════════════
with tab_flags:
    st.markdown("### Verletzungsrisiko · Leistungsabfälle")
    if not prev_sess:
        st.info("Zwei Sessions erforderlich für Vergleich.")
    else:
        st.caption(f"Z-Score Rückgang >8 Punkte zwischen {prev_sess} → {last_sess}")
        flags = compute_injury_flags(df)

        if not flags:
            st.success("✓ Keine signifikanten Leistungsabfälle erkannt.")
        else:
            for f in flags:
                with st.expander(f"{f['severity']} — {f['name']}"):
                    drops_str = ", ".join(f"{d['metric']} ({d['delta']:+.1f} Z)" for d in f["drops"])
                    st.error(f"↓ Rückgänge: {drops_str}")
                    if f["gains"]:
                        gains_str = ", ".join(f"{g['metric']} ({g['delta']:+.1f} Z)" for g in f["gains"])
                        st.success(f"↑ Verbesserungen: {gains_str}")

        st.divider()
        st.markdown("#### Δ Z-Score Heatmap")
        hm_rows = []
        for name in players:
            j = df[(df["name"] == name) & (df["session"] == prev_sess)]
            m = df[(df["name"] == name) & (df["session"] == last_sess)]
            if j.empty or m.empty:
                continue
            j, m = j.iloc[0], m.iloc[0]
            row = {"Spielerin": name}
            for metric, lbl in zip(RADAR_METRICS, RADAR_LABELS):
                zk = f"{metric}_Z"
                jz, mz = j.get(zk), m.get(zk)
                row[lbl] = round(mz - jz, 1) if pd.notna(jz) and pd.notna(mz) else None
            hm_rows.append(row)

        if hm_rows:
            hm_df = pd.DataFrame(hm_rows).set_index("Spielerin")
            st.dataframe(
                hm_df.style.background_gradient(cmap="RdYlGn", vmin=-20, vmax=20)
                           .format("{:.1f}", na_rep="—"),
                use_container_width=True,
                height=500,
            )

# ══════════════════════════════════════════════
# PDF REPORTS
# ══════════════════════════════════════════════
with tab_pdf:
    st.markdown("### PDF Leistungsberichte")
    st.caption("Vollständiger Bericht pro Spielerin mit Radar, Trends, Auswertung und Prognose.")
    st.divider()

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        pdf_player = st.selectbox("Spielerin", players, key="pdf_player")
    with col_p2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄 PDF generieren", type="primary"):
            with st.spinner(f"Generiere Bericht für {pdf_player}..."):
                try:
                    pdf_bytes = generate_pdf(df, pdf_player)
                    safe = pdf_player.replace(" ", "_")
                    st.download_button(
                        label=f"⬇ {pdf_player} — PDF herunterladen",
                        data=pdf_bytes,
                        file_name=f"BVB_Report_{safe}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✓ PDF bereit")
                except Exception as e:
                    st.error(f"Fehler: {e}")

    st.divider()
    st.markdown("#### Alle Spielerinnen auf einmal")
    if st.button("📦 Alle PDFs generieren (ZIP)"):
        import zipfile
        zip_buf = io.BytesIO()
        progress = st.progress(0)
        status = st.empty()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i, name in enumerate(players):
                status.text(f"Generiere {name}...")
                try:
                    pdf_bytes = generate_pdf(df, name)
                    safe = name.replace(" ", "_")
                    zf.writestr(f"BVB_Report_{safe}.pdf", pdf_bytes)
                except Exception as e:
                    st.warning(f"⚠ {name}: {e}")
                progress.progress((i + 1) / len(players))
        zip_buf.seek(0)
        st.download_button(
            label="⬇ Alle PDFs als ZIP herunterladen",
            data=zip_buf,
            file_name=f"BVB_Reports_{last_sess}.zip",
            mime="application/zip",
        )
        status.text("✓ Fertig!")
