"""
BVB Frauen — Performance Analytics
Streamlit App · run with: streamlit run app.py
"""

import streamlit as st
from database import BVBDatabase
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
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

# Historical reference MW/SD seit Jul 2023 (from Zusammenfassung_1_Mannschaft.xlsx)
# Used for player radar Z-scores — more meaningful than session-relative Z
HIST_MW = {
    "cmj": 31.780, "dj_rsi": 1.630, "t5": 1.082, "t10": 1.884,
    "t20": 3.329,  "t30": 4.660, "agility": 16.332,
    "dribbling": 10.331, "vo2max": 49.538,
}
HIST_SD = {
    "cmj": 4.160, "dj_rsi": 0.294, "t5": 0.046, "t10": 0.073,
    "t20": 0.132, "t30": 0.192, "agility": 0.663,
    "dribbling": 0.639, "vo2max": 4.378,
}

def hist_z(metric: str, value) -> float:
    """Z-score vs historical mean (Jul 2023–present). Same formula as Benedikt's Netzdiagramme."""
    if value is None or pd.isna(value):
        return None
    info = RAW_METRICS.get(metric, {})
    sign = 1 if info.get("hib", True) else -1
    mw   = HIST_MW.get(metric)
    sd   = HIST_SD.get(metric)
    if mw is None or sd is None or sd == 0:
        return None
    return round(100 + 10 * sign * (float(value) - mw) / sd, 1)

BVB_YELLOW = "#FDE000"
COLORS = ["#FDE000", "#60A5FA", "#4ADE80", "#FB923C", "#C084FC", "#F472B6"]

def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# All-time historical baselines from Zusammenfassung (Jul 2023 – Mär 2026)
# Used to anchor the team radar so sessions are comparable to real history
ALLTIME_MEAN = {
    "cmj": 31.780, "dj_rsi": 1.630, "t5": 1.082, "t10": 1.884,
    "t20": 3.329,  "agility": 16.332, "dribbling": 10.331, "vo2max": 49.538,
}
ALLTIME_SD = {
    "cmj": 4.160, "dj_rsi": 0.294, "t5": 0.046, "t10": 0.073,
    "t20": 0.132, "agility": 0.663, "dribbling": 0.639, "vo2max": 4.378,
}


def team_radar_z(df: pd.DataFrame, metric: str, sessions: list) -> list:
    """
    Compute Z-scores for each session's team average relative to the
    all-time historical mean (MW seit 2023 from Zusammenfassung).
    This gives each session a meaningful absolute position rather than
    collapsing all sessions to 100 (which per-session Z always does).
    """
    info = RAW_METRICS.get(metric, {})
    hib  = info.get("hib", True)
    sign = 1 if hib else -1
    mean = ALLTIME_MEAN.get(metric)
    sd   = ALLTIME_SD.get(metric)
    result = []
    for s in sessions:
        sess_df = df[(df["session"] == s) & df["cmj"].notna()]
        v = sess_df[metric].mean() if metric in sess_df.columns else None
        if v is None or pd.isna(v) or mean is None or sd is None or sd == 0:
            result.append(100.0)
        else:
            z = round(100 + 10 * sign * (v - mean) / sd, 1)
            result.append(z)
    return result

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
    """Always loads fresh from DB and recomputes Z-scores."""
    raw = st.session_state.db.load_dataframe()
    return compute_z_scores(raw)


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
            # Use historical Z-scores (vs MW seit 2023) — matches Benedikt's Netzdiagramme
            vals = [hist_z(m, rec.get(m)) for m in RADAR_METRICS]
            if all(v is None for v in vals):
                continue
            vals_clean = [v if v is not None else 100 for v in vals]
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
                hovertemplate="%{theta}: Z=%{r:.1f}<extra>" + label + "</extra>",
            ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font_color="#FDE000", font_size=14),
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, range=[70, 135], showticklabels=True,
                            tickvals=[80, 90, 100, 110, 120],
                            ticktext=["80", "90", "100", "110", "120"],
                            tickfont=dict(size=8, color="#444"),
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
        # Use historical Z-scores for meaningful comparison
        pz = hist_z(m, rec.get(m))
        if pz is not None:
            # Team average Z for this session
            team_metric_vals = pd.to_numeric(team[m], errors="coerce").dropna()
            if len(team_metric_vals) > 0:
                tz = np.mean([hist_z(m, v) for v in team_metric_vals if hist_z(m, v) is not None])
            else:
                tz = 100
            labels.append(RAW_METRICS[m]["label"])
            player_vals.append(pz)
            team_vals.append(tz if tz is not None else 100)
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
        vals = [team_radar_z(df, m, sessions)[si] for m in RADAR_METRICS]
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
                visible=True, range=[85, 125],
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
# ─────────────────────────────────────────────
# PDF ENGINE  (elite report generator)
# ─────────────────────────────────────────────
"""
BVB Frauen — Elite PDF Report Engine
======================================
Replaces generate_team_pdf() and generate_player_pdf().
Charts rendered via Matplotlib → PNG → embedded in ReportLab PDF.
"""

import io
import math
import tempfile
import os
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
BVB_Y   = colors.HexColor("#FDE000")
BVB_BK  = colors.HexColor("#0A0A0A")
BVB_DG  = colors.HexColor("#1A1A1A")
DGRAY   = colors.HexColor("#2D2D2D")
MGRAY   = colors.HexColor("#555555")
LGRAY   = colors.HexColor("#F0F0F0")
WHITE   = colors.white
GREEN   = colors.HexColor("#22C55E")
RED     = colors.HexColor("#EF4444")
AMBER   = colors.HexColor("#F59E0B")
BLUE    = colors.HexColor("#3B82F6")

W_PAGE  = A4[0]
H_PAGE  = A4[1]
MARGIN  = 1.5 * cm
W_BODY  = W_PAGE - 2 * MARGIN   # ≈ 18.2 cm

MPL_BG   = "#0A0A0A"
MPL_BG2  = "#111111"
MPL_GRID = "#222222"
MPL_TXT  = "#AAAAAA"
MPL_Y    = "#FDE000"
MPL_DARK = "#2D2D2D"

HIST_MW = {
    "cmj":31.780,"dj_rsi":1.630,"t5":1.082,"t10":1.884,
    "t20":3.329,"t30":4.660,"agility":16.332,
    "dribbling":10.331,"vo2max":49.538,
}
HIST_SD = {
    "cmj":4.160,"dj_rsi":0.294,"t5":0.046,"t10":0.073,
    "t20":0.132,"t30":0.192,"agility":0.663,
    "dribbling":0.639,"vo2max":4.378,
}
HIB = {
    "cmj":True,"dj_rsi":True,"t5":False,"t10":False,
    "t20":False,"t30":False,"agility":False,
    "dribbling":False,"vo2max":True,
}
METRIC_LABELS = {
    "cmj":"CMJ [cm]","dj_rsi":"DJ RSI","t5":"t5 [s]","t10":"t10 [s]",
    "t20":"t20 [s]","t30":"t30 [s]","agility":"Agility [s]",
    "dribbling":"Dribbling [s]","vo2max":"VO2max",
}
DEC_MAP = {
    "cmj":1,"dj_rsi":2,"t5":2,"t10":2,
    "t20":2,"t30":2,"agility":2,"dribbling":2,"vo2max":2,
}

# Radar axis groupings (6-axis radar)
RADAR_AXES = {
    "Sprint":     ["t5", "t10"],
    "Acceleration": ["t20", "t30"],
    "Power":      ["cmj"],
    "Reactivity": ["dj_rsi"],
    "Endurance":  ["vo2max"],
    "Agility & Technique": ["agility", "dribbling"],
}
RADAR_LABELS = list(RADAR_AXES.keys())


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmt(v, dec=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{dec}f}".replace(".", ",")


def hist_z(metric, value):
    if value is None or pd.isna(value):
        return None
    sign = 1 if HIB.get(metric, True) else -1
    mw, sd = HIST_MW.get(metric), HIST_SD.get(metric)
    if not mw or not sd or sd == 0:
        return None
    return round(100 + 10 * sign * (float(value) - mw) / sd, 1)


def z_to_badge(z):
    if z is None:
        return "—", "#555555"
    if z >= 112:
        return "ELITE", "#22C55E"
    if z >= 104:
        return "STRONG", "#3B82F6"
    if z >= 96:
        return "AVERAGE", "#888888"
    if z >= 88:
        return "DEVELOPING", "#F59E0B"
    return "FOCUS AREA", "#EF4444"


def axis_z(metrics_list, player_row):
    """Mean hist_z for a group of metrics for one player row."""
    vals = [hist_z(m, player_row.get(m)) for m in metrics_list]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def player_axis_scores(row):
    """Returns dict of {axis_label: z_score} for the 6-axis radar."""
    return {ax: axis_z(metrics, row) for ax, metrics in RADAR_AXES.items()}


def style():
    """Return shared ParagraphStyle factory dict."""
    return {
        "cover_title": ParagraphStyle("ct", fontName="Helvetica-Bold",
            fontSize=28, textColor=BVB_Y, leading=34, alignment=TA_CENTER),
        "cover_sub": ParagraphStyle("cs", fontName="Helvetica",
            fontSize=11, textColor=colors.HexColor("#CCCCCC"), leading=16,
            alignment=TA_CENTER),
        "cover_meta": ParagraphStyle("cm", fontName="Helvetica",
            fontSize=8.5, textColor=MGRAY, leading=13, alignment=TA_CENTER),
        "section": ParagraphStyle("sec", fontName="Helvetica-Bold",
            fontSize=12, textColor=BVB_Y, leading=16, spaceBefore=12, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontName="Helvetica-Bold",
            fontSize=9.5, textColor=colors.HexColor("#1A1A1A"), leading=13,
            spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontName="Helvetica",
            fontSize=8.5, textColor=colors.HexColor("#2A2A2A"), leading=12.5),
        "small": ParagraphStyle("sm", fontName="Helvetica",
            fontSize=7.5, textColor=MGRAY, leading=10.5),
        "insight": ParagraphStyle("ins", fontName="Helvetica-Oblique",
            fontSize=8.5, textColor=colors.HexColor("#1A1A1A"), leading=12.5),
        "caption": ParagraphStyle("cap", fontName="Helvetica",
            fontSize=7, textColor=MGRAY, leading=9, alignment=TA_CENTER),
    }


# ── CHART FUNCTIONS ───────────────────────────────────────────────────────────

def chart_radar(player_scores: dict, team_scores: dict,
                title="", figsize=(4.2, 4.2)) -> bytes:
    """6-axis radar: player (yellow) vs team average (grey)."""
    labels = RADAR_LABELS
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    p_vals = [player_scores.get(ax, 100) or 100 for ax in labels] + \
             [player_scores.get(labels[0], 100) or 100]
    t_vals = [team_scores.get(ax, 100) or 100 for ax in labels] + \
             [team_scores.get(labels[0], 100) or 100]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG2)

    ax.plot(angles, t_vals, '--', lw=1.4, color="#666666", zorder=2)
    ax.fill(angles, t_vals, alpha=0.10, color="#666666")
    ax.plot(angles, p_vals, 'o-', lw=2.5, color=MPL_Y, zorder=3, markersize=4)
    ax.fill(angles, p_vals, alpha=0.22, color=MPL_Y)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#CCCCCC", fontweight="bold")
    ax.set_ylim(70, 130)
    ax.set_yticks([80, 90, 100, 110, 120])
    ax.set_yticklabels(["80", "90", "100", "110", "120"],
                       size=6, color="#555555")
    ax.spines["polar"].set_color("#333333")
    ax.grid(color=MPL_GRID, linewidth=0.7)

    legend_elements = [
        mpatches.Patch(facecolor=MPL_Y, alpha=0.6, label="Spielerin"),
        mpatches.Patch(facecolor="#666666", alpha=0.4, label="Team Ø"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.45, 1.18), fontsize=7.5,
              framealpha=0, labelcolor="white")

    if title:
        ax.set_title(title, color=MPL_Y, size=8.5, pad=18, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=MPL_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_hbar(players: list, values: list, metric_label: str,
               team_avg=None, highlight_idx=None,
               figsize=None, max_n=17, dec=2) -> bytes:
    """Horizontal bar chart — ranked players."""
    players = players[:max_n]
    values  = values[:max_n]
    n = len(players)
    if figsize is None:
        figsize = (7.2, max(2.0, n * 0.40))

    bar_colors = []
    for i, v in enumerate(values):
        if i == highlight_idx:
            bar_colors.append(MPL_Y)
        elif i == 0:
            bar_colors.append("#22C55E")
        elif i == n - 1:
            bar_colors.append("#EF4444")
        else:
            bar_colors.append(MPL_DARK)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG2)

    bars = ax.barh([p[:22] for p in players[::-1]],
                   values[::-1],
                   color=bar_colors[::-1],
                   height=0.65, edgecolor="none")

    if team_avg is not None:
        label_avg = f"Team Ø: {team_avg:.2f}".replace(".", ",")
        ax.axvline(team_avg, color=MPL_Y, lw=1.2, ls="--",
                   alpha=0.75, label=label_avg)
        ax.legend(fontsize=7.5, framealpha=0, labelcolor="white",
                  loc="lower right")

    val_range = max(values) - min(values) if max(values) != min(values) else 1
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + val_range * 0.012,
                bar.get_y() + bar.get_height() / 2,
                fmt(val, dec),
                va="center", ha="left", color="#CCCCCC", size=7.5, fontweight="bold")

    ax.set_xlabel(metric_label, color=MPL_TXT, size=8.5)
    ax.tick_params(colors=MPL_TXT, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.xaxis.label.set_color(MPL_TXT)
    ax.set_xlim(right=max(values) + val_range * 0.16)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight",
                facecolor=MPL_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_trend(player_name: str, df: pd.DataFrame,
                metrics=None, figsize=(7.0, 2.6)) -> bytes:
    """Multi-metric Z-score trend across sessions."""
    if metrics is None:
        metrics = ["cmj", "vo2max", "t20", "agility"]

    p_df = df[df["name"] == player_name].sort_values("session")
    sessions = p_df["session"].tolist()
    if len(sessions) < 2:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG2)

    palette = [MPL_Y, "#22C55E", "#3B82F6", "#F59E0B",
               "#EF4444", "#A855F7", "#EC4899", "#14B8A6"]
    plotted = 0
    for i, m in enumerate(metrics):
        zvals = [hist_z(m, row[m]) for _, row in p_df.iterrows()]
        if all(v is None for v in zvals):
            continue
        ax.plot(sessions, [v or np.nan for v in zvals],
                "o-", lw=2, color=palette[i % len(palette)],
                label=METRIC_LABELS.get(m, m), markersize=5)
        plotted += 1

    ax.axhline(100, color="#444444", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(70, 130)
    ax.set_yticks([80, 90, 100, 110, 120])
    ax.set_yticklabels(["80", "90", "100↑", "110", "120"], size=7, color=MPL_TXT)
    ax.tick_params(colors=MPL_TXT, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(axis="y", color=MPL_GRID, lw=0.5)
    if plotted > 0:
        ax.legend(fontsize=6.5, framealpha=0, labelcolor="white",
                  loc="upper left", ncol=4)
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight",
                facecolor=MPL_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_percentile_bar(player_name: str, df: pd.DataFrame,
                         session: str, figsize=(7.0, 3.2)) -> bytes:
    """Show player Z-score per metric vs 100 baseline — percentile style."""
    p_row = df[(df["name"] == player_name) & (df["session"] == session)]
    if p_row.empty:
        return None
    row = p_row.iloc[0]

    metrics_to_show = ["cmj", "dj_rsi", "t5", "t20", "vo2max"]
    labels, zscores, bar_colors = [], [], []

    for m in metrics_to_show:
        z = hist_z(m, row.get(m))
        if z is None:
            continue
        labels.append(METRIC_LABELS.get(m, m))
        zscores.append(z)
        if z >= 112:
            bar_colors.append("#22C55E")
        elif z >= 104:
            bar_colors.append("#3B82F6")
        elif z >= 96:
            bar_colors.append("#555555")
        elif z >= 88:
            bar_colors.append("#F59E0B")
        else:
            bar_colors.append("#EF4444")

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG2)

    x = np.arange(len(labels))
    bars = ax.bar(x, zscores, color=bar_colors, width=0.58, edgecolor="none")
    ax.axhline(100, color="#666666", lw=1.2, ls="--", alpha=0.8,
               label="Team Ø (Z=100)")

    for bar, z in zip(bars, zscores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{z:.0f}", ha="center", va="bottom",
                color="#DDDDDD", size=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=8, color=MPL_TXT, rotation=18, ha="right")
    ax.set_ylim(70, max(132, max(zscores) + 12))
    ax.set_ylabel("Z-Score", color=MPL_TXT, size=8.5)
    ax.tick_params(colors=MPL_TXT, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(axis="y", color=MPL_GRID, lw=0.5, alpha=0.6)
    ax.legend(fontsize=7.5, framealpha=0, labelcolor="white", loc="upper right")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight",
                facecolor=MPL_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── REPORTLAB HELPERS ──────────────────────────────────────────────────────────

def img_flowable(png_bytes: bytes, width_cm=16.0) -> Image:
    w = width_cm * cm
    img = Image(io.BytesIO(png_bytes))
    aspect = img.imageWidth / img.imageHeight
    return Image(io.BytesIO(png_bytes), width=w, height=w / aspect)


def rl_rule(story, color=BVB_Y, thickness=1.5):
    story.append(HRFlowable(width="100%", thickness=thickness,
                             color=color, spaceAfter=5, spaceBefore=3))


def rl_section(story, text, S):
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(text.upper(), S["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BVB_Y,
                             spaceAfter=6, spaceBefore=2))


def rl_sub(story, text, S):
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(text, S["sub"]))


def styled_table(rows, col_widths, header_rows=1, zebra=True,
                 best_row=None, worst_row=None):
    t = Table(rows, colWidths=col_widths, repeatRows=header_rows)
    cmds = [
        ("BACKGROUND",   (0, 0), (-1, header_rows - 1), colors.HexColor("#1A1A1A")),
        ("TEXTCOLOR",    (0, 0), (-1, header_rows - 1), BVB_Y),
        ("FONTNAME",     (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LINEBELOW",    (0, header_rows - 1), (-1, header_rows - 1), 1.5, BVB_Y),
        ("GRID",         (0, header_rows), (-1, -1), 0.25, colors.HexColor("#E8E8E8")),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]
    if zebra:
        for i in range(header_rows, len(rows)):
            bg = WHITE if i % 2 == 0 else colors.HexColor("#F9F9F9")
            cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    if best_row is not None:
        cmds.append(("BACKGROUND", (0, best_row), (-1, best_row),
                     colors.HexColor("#DCFCE7")))
        cmds.append(("TEXTCOLOR",  (0, best_row), (-1, best_row),
                     colors.HexColor("#166534")))
    if worst_row is not None:
        cmds.append(("BACKGROUND", (0, worst_row), (-1, worst_row),
                     colors.HexColor("#FEE2E2")))
        cmds.append(("TEXTCOLOR",  (0, worst_row), (-1, worst_row),
                     colors.HexColor("#991B1B")))
    t.setStyle(TableStyle(cmds))
    return t


def score_card_table(cards: list) -> Table:
    """
    Render a row of premium metric cards matching website style.
    cards = list of (label, value, badge_text, badge_color)
    """
    S = style()
    CARD_W = W_BODY / len(cards)
    cells = []
    for label, value, badge, badge_color in cards:
        # Value colour: use badge colour; grey "AVERAGE" cards get dark value text
        val_color = badge_color if badge_color not in ("#888888", "#555555") else "#0A0A0A"
        cell_content = [
            Paragraph(f'<font size="6.5" color="#888888">{label}</font>', S["caption"]),
            Spacer(1, 5),
            Paragraph(f'<b><font size="18" color="{val_color}">{value}</font></b>',
                      S["caption"]),
            Spacer(1, 4),
            Paragraph(f'<font size="6.5" color="{badge_color}"><b>{badge}</b></font>',
                      S["caption"]),
        ]
        cells.append(cell_content)

    t = Table([cells], colWidths=[CARD_W] * len(cards))
    cmds = [
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("LINEAFTER",    (0, 0), (-2, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("LINEBELOW",    (0, 0), (-1, -1), 2.5, BVB_Y),
    ]
    t.setStyle(TableStyle(cmds))
    return t


def cover_page_elements(title: str, subtitle: str,
                        session: str, n_players: int, S: dict) -> list:
    elems = []

    # Premium header band — dark background with BVB branding text
    brand_style = ParagraphStyle("brand", fontName="Helvetica-Bold",
        fontSize=11, textColor=BVB_Y, leading=16, alignment=TA_CENTER)
    dept_style  = ParagraphStyle("dept",  fontName="Helvetica",
        fontSize=8,  textColor=colors.HexColor("#888888"), leading=12, alignment=TA_CENTER)

    header_inner = Table(
        [[Paragraph("BVB FRAUEN", brand_style)],
         [Paragraph("Performance Diagnostics · Sports Science Department", dept_style)]],
        colWidths=[W_BODY],
    )
    header_inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BVB_BK),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("LINEABOVE",     (0, 0), (-1, 0),  4, BVB_Y),
        ("LINEBELOW",     (0, -1),(-1, -1), 1, colors.HexColor("#333333")),
    ]))
    elems.append(header_inner)
    elems.append(Spacer(1, 0.5 * cm))

    # BVB yellow accent line
    elems.append(HRFlowable(width="100%", thickness=3.5, color=BVB_Y,
                             spaceAfter=14, spaceBefore=0))

    elems.append(Paragraph(title, S["cover_title"]))
    elems.append(Spacer(1, 0.25 * cm))
    elems.append(Paragraph(subtitle, S["cover_sub"]))
    elems.append(Spacer(1, 0.35 * cm))
    elems.append(HRFlowable(width="55%", thickness=0.75, color=DGRAY,
                             spaceAfter=10, spaceBefore=0))
    elems.append(Paragraph(
        f"Session: <b>{session}</b>   ·   {n_players} Spielerinnen getestet   ·   "
        f"{datetime.now().strftime('%d.%m.%Y')}",
        S["cover_meta"]))
    elems.append(Spacer(1, 0.2 * cm))
    elems.append(Paragraph(
        "BVB Frauen · Performance Diagnostics · Sports Science Department",
        S["cover_meta"]))
    return elems


def auto_insights(df: pd.DataFrame, session: str, player: str = None) -> list:
    """Generate bullet-point insights from the data."""
    sdf = df[df["session"] == session]
    if player:
        sdf = sdf[sdf["name"] == player]
    insights = []

    METRICS_CHECKED = [
        ("cmj",      "Sprungkraft (CMJ)",       True),
        ("dj_rsi",   "Reaktivkraft (RSI)",       True),
        ("t20",      "Sprint 20m",               False),
        ("agility",  "Agility",                  False),
        ("vo2max",   "Aerobe Kapazität (VO2max)", True),
    ]

    if player:
        row = sdf.iloc[0] if not sdf.empty else None
        if row is None:
            return []
        for m, label, _ in METRICS_CHECKED:
            z = hist_z(m, row.get(m))
            if z is None:
                continue
            badge, _ = z_to_badge(z)
            if z >= 108:
                insights.append(f"✦ {label}: überdurchschnittlich stark (Z={z:.0f}) — {badge}")
            elif z <= 92:
                insights.append(f"■ {label}: Entwicklungspotenzial (Z={z:.0f}) — Trainingsempfehlung")
    else:
        # Team insights
        for m, label, hib in METRICS_CHECKED:
            col = pd.to_numeric(sdf[m], errors="coerce").dropna()
            if len(col) == 0:
                continue
            team_z = np.mean([hist_z(m, v) for v in col if hist_z(m, v) is not None])
            if team_z is None:
                continue
            if team_z >= 105:
                insights.append(f"✦ {label}: Team liegt über historischem Schnitt (Ø Z={team_z:.0f})")
            elif team_z <= 95:
                insights.append(f"■ {label}: Teamschnitt unter Referenzwert (Ø Z={team_z:.0f}) — Fokus empfohlen")

    return insights[:5]


def training_recommendations(player: str, df: pd.DataFrame, session: str) -> list:
    """Generate training focus suggestions based on Z-scores."""
    p_row = df[(df["name"] == player) & (df["session"] == session)]
    if p_row.empty:
        return []
    row = p_row.iloc[0]

    recs = []
    metric_focus = {
        "cmj":      "Pliometrisches Training: Box Jumps, Depth Drops, Squat Jumps",
        "dj_rsi":   "Reaktivkraft: Drop Jumps, Ankle Stiffness Drills, Bounding",
        "t5":       "Beschleunigung: Start-Explosivität, Hip Drive, Sled Pushes",
        "t20":      "Maximale Sprintgeschwindigkeit: Fliegende Sprints, Wickets",
        "agility":  "Agilität: COD-Drills, Reaktions-Shuttles, T-Test Varianten",
        "dribbling":"Technische Schnelligkeit: Ball-Drills unter Zeitdruck",
        "vo2max":   "Aerobe Basis: Intervalltraining 90–95% HFmax, Tempoläufe",
    }
    for m, rec in metric_focus.items():
        z = hist_z(m, row.get(m))
        if z is not None and z < 95:
            recs.append(f"• {METRIC_LABELS.get(m, m)}: {rec}")

    return recs[:3] if recs else ["• Leistungsprofil ausgeglichen — allgemeines Erhaltungstraining empfohlen"]


# ── FOOTER CANVAS ──────────────────────────────────────────────────────────────

def make_footer_canvas(session_label: str):
    def footer(canvas, doc):
        canvas.saveState()
        # Black footer band
        canvas.setFillColor(BVB_BK)
        canvas.rect(0, 0, W_PAGE, 1.1 * cm, fill=1, stroke=0)
        # Yellow accent line above footer
        canvas.setFillColor(BVB_Y)
        canvas.rect(0, 1.1 * cm, W_PAGE, 0.12 * cm, fill=1, stroke=0)
        # Footer text
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawString(MARGIN, 0.38 * cm,
            f"BVB Frauen · Performance Diagnostics · Session: {session_label}")
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(colors.HexColor("#BBBBBB"))
        canvas.drawRightString(
            W_PAGE - MARGIN, 0.38 * cm,
            f"Seite {doc.page}  ·  Generiert am {datetime.now().strftime('%d.%m.%Y')}")
        canvas.restoreState()
    return footer


# ── TEAM PDF ───────────────────────────────────────────────────────────────────

def generate_team_pdf(df: pd.DataFrame) -> bytes:
    S = style()
    buf = io.BytesIO()
    ALL_SESSIONS = sorted(df["session"].unique().tolist())
    last_sess = ALL_SESSIONS[-1]
    last_df   = df[df["session"] == last_sess]
    n_players = last_df["name"].nunique()
    HIB_MAP   = HIB

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=1.6 * cm,
    )
    story = []

    # ── COVER ──────────────────────────────────────────────────────────────
    story += cover_page_elements(
        "Leistungsdiagnostik", "BVB Frauen · Team Performance Report",
        last_sess, n_players, S,
    )
    story.append(Spacer(1, 0.6 * cm))

    # Executive summary score cards
    def best_player(metric):
        col = pd.to_numeric(last_df[metric], errors="coerce")
        idx = col.idxmax() if HIB_MAP.get(metric, True) else col.idxmin()
        if pd.isna(idx):
            return "—", None
        row = last_df.loc[idx]
        return row["name"].split()[-1], col[idx]

    sprint_name, sprint_val = best_player("t20")
    jump_name,   jump_val   = best_player("cmj")
    vo2_name,    vo2_val    = best_player("vo2max")
    rsi_name,    rsi_val    = best_player("dj_rsi")

    # Overall team fitness score (mean of all hist_z)
    all_z = []
    for m in ["cmj", "dj_rsi", "t5", "t20", "agility", "dribbling", "vo2max"]:
        col = pd.to_numeric(last_df[m], errors="coerce").dropna()
        all_z += [hist_z(m, v) for v in col if hist_z(m, v) is not None]
    team_fitness = round(np.mean(all_z), 1) if all_z else 100
    fitness_badge, fitness_color = z_to_badge(team_fitness)

    cards = [
        ("BEST SPRINT",    fmt(sprint_val, 2) + "s", sprint_name, "#3B82F6"),
        ("BEST JUMP",      fmt(jump_val, 1) + "cm",  jump_name,   "#22C55E"),
        ("BEST VO2MAX",    fmt(vo2_val, 1),           vo2_name,    "#A855F7"),
        ("TEAM FITNESS",   f"Z = {team_fitness:.0f}", fitness_badge, fitness_color),
    ]
    story.append(score_card_table(cards))
    story.append(PageBreak())

    # ── PAGE 2: TEAM RADAR + TEAM INSIGHTS ────────────────────────────────
    rl_section(story, "Squad Benchmark Summary", S)

    # Build team average axis scores
    team_axis = {}
    for ax, metrics in RADAR_AXES.items():
        z_vals = []
        for m in metrics:
            col = pd.to_numeric(last_df[m], errors="coerce").dropna()
            z_vals += [hist_z(m, v) for v in col if hist_z(m, v) is not None]
        team_axis[ax] = round(np.mean(z_vals), 1) if z_vals else 100

    # Elite benchmark = 108 across all axes (top 20%)
    elite_axis = {ax: 108 for ax in RADAR_LABELS}

    radar_bytes = chart_radar(team_axis, elite_axis,
                              title="Team Ø vs Elite-Referenz (Z=108)")
    story.append(img_flowable(radar_bytes, width_cm=12))
    story.append(Paragraph("Radar basiert auf historischen Z-Scores (Referenz: MW seit Jul 2023). "
                            "Außenring = Elite-Niveau (Z≥108).", S["caption"]))
    story.append(Spacer(1, 0.3 * cm))

    # Team insights
    rl_sub(story, "Auto-Insights", S)
    insights = auto_insights(df, last_sess)
    for ins in insights:
        story.append(Paragraph(ins, S["insight"]))
    story.append(Spacer(1, 0.2 * cm))

    # Session stats table
    rl_sub(story, "Mannschaft · Statistische Übersicht", S)
    for sess in ALL_SESSIONS:
        s_df = df[df["session"] == sess]
        story.append(Paragraph(f"<b>{sess}</b>", S["body"]))
        story.append(Spacer(1, 0.1 * cm))

        METRIC_GROUPS = [
            ("Sprint", ["t5","t10","t20","t30"]),
            ("Power",  ["cmj","dj_rsi"]),
            ("Fitness",["agility","dribbling","vo2max"]),
        ]
        col_w = [3.2*cm, 2.5*cm, 2.3*cm, 2.0*cm, 2.3*cm, 2.3*cm]
        hrow = [Paragraph(f"<b>{h}</b>", S["body"])
                for h in ["Gruppe · Metrik", "Messgröße", "Mittelwert", "SD", "Min", "Max"]]
        rows = [hrow]
        for grp, metrics in METRIC_GROUPS:
            first = True
            for m in metrics:
                col = pd.to_numeric(s_df[m], errors="coerce").dropna()
                dec = DEC_MAP.get(m, 2)
                rows.append([
                    Paragraph(f"<b>{grp}</b>" if first else "", S["body"]),
                    Paragraph(METRIC_LABELS.get(m, m), S["body"]),
                    Paragraph(fmt(col.mean(), dec) if len(col) else "—", S["body"]),
                    Paragraph(fmt(col.std(),  dec) if len(col) else "—", S["body"]),
                    Paragraph(fmt(col.min(),  dec) if len(col) else "—", S["body"]),
                    Paragraph(fmt(col.max(),  dec) if len(col) else "—", S["body"]),
                ])
                first = False

        story.append(styled_table(rows, col_w))
        story.append(Spacer(1, 0.3 * cm))

    # ── PAGE 3: RANKINGS ───────────────────────────────────────────────────
    story.append(PageBreak())
    rl_section(story, "Performance Rankings", S)

    key_metrics = [
        ("cmj",       "Explosive Power Leaderboard · CMJ"),
        ("t20",       "Sprint Performance Rankings · 20m"),
        ("vo2max",    "Aerobic Capacity Analysis · VO2max"),
        ("agility",   "Agility Rankings"),
        ("dribbling", "Technical Speed Rankings · Dribbling"),
        ("dj_rsi",    "Reactive Strength Index · DJ RSI"),
    ]

    for m, chart_title in key_metrics:
        hib = HIB_MAP.get(m, True)
        all_vals = {}
        for _, row in df[df[m].notna()].iterrows():
            v = pd.to_numeric(row[m], errors="coerce")
            if pd.notna(v):
                p = row["name"]
                s = row["session"]
                if p not in all_vals:
                    all_vals[p] = {}
                all_vals[p][s] = v

        if not all_vals:
            continue

        sorted_players = sorted(
            all_vals.keys(),
            key=lambda p: all_vals[p].get(last_sess,
                float("-inf") if hib else float("inf")),
            reverse=hib,
        )

        vals_last = [all_vals[p].get(last_sess) for p in sorted_players
                     if all_vals[p].get(last_sess) is not None]
        players_with_val = [p for p in sorted_players
                            if all_vals[p].get(last_sess) is not None]

        if not vals_last:
            continue

        team_avg = np.mean(vals_last)
        bar_bytes = chart_hbar(players_with_val, vals_last,
                                METRIC_LABELS.get(m, m),
                                team_avg=team_avg,
                                dec=DEC_MAP.get(m, 2))

        # Table alongside
        col_w_r = [4.5*cm] + [min(2.6*cm, (W_BODY-4.5*cm)/len(ALL_SESSIONS))] * len(ALL_SESSIONS)
        hrow = [Paragraph("<b>Spielerin</b>", S["body"])] + \
               [Paragraph(f"<b>{s}</b>", S["body"]) for s in ALL_SESSIONS]
        t_rows = [hrow]
        for p in sorted_players:
            row_data = [Paragraph(p, S["body"])]
            for s in ALL_SESSIONS:
                v = all_vals[p].get(s)
                dec = DEC_MAP.get(m, 2)
                row_data.append(Paragraph(fmt(v, dec) if v is not None else "X", S["body"]))
            t_rows.append(row_data)

        best_r  = 1 if len(t_rows) > 1 else None
        worst_r = len(t_rows) - 1 if len(t_rows) > 2 else None

        story.append(KeepTogether([
            Paragraph(chart_title, S["sub"]),
            img_flowable(bar_bytes, width_cm=16),
            Spacer(1, 0.15 * cm),
            styled_table(t_rows, col_w_r, best_row=best_r, worst_row=worst_r),
            Spacer(1, 0.5 * cm),
        ]))

    # ── PAGE 4: INDIVIDUAL CARDS ────────────────────────────────────────────
    story.append(PageBreak())
    rl_section(story, "Individuelle Leistungsprofile", S)

    # Team axis scores for comparison in player cards
    all_players_sorted = sorted(df["name"].unique())
    for player in all_players_sorted:
        p_df = df[df["name"] == player].sort_values("session")
        if p_df.empty:
            continue
        p_sessions = p_df["session"].tolist()
        last_p_sess = p_sessions[-1]
        p_last_row  = p_df[p_df["session"] == last_p_sess].iloc[0]

        p_axis = player_axis_scores(p_last_row)
        t_axis = team_axis

        # Score cards
        overall_z = np.mean([v for v in p_axis.values() if v is not None] or [100])
        badge, bc  = z_to_badge(overall_z)
        cmj_v  = p_last_row.get("cmj")
        t20_v  = p_last_row.get("t20")
        vo2_v  = p_last_row.get("vo2max")

        rsi_v = p_last_row.get("dj_rsi")
        p_cards = [
            ("OVERALL",    f"Z = {overall_z:.0f}", badge, bc),
            ("CMJ",        fmt(cmj_v, 1) + " cm" if cmj_v else "—",
             *z_to_badge(hist_z("cmj", cmj_v))),
            ("SPRINT 20m", fmt(t20_v, 2) + " s" if t20_v else "—",
             *z_to_badge(hist_z("t20", t20_v))),
            ("VO2MAX",     fmt(vo2_v, 1) if vo2_v else "—",
             *z_to_badge(hist_z("vo2max", vo2_v))),
            ("DJ RSI",     fmt(rsi_v, 2) if rsi_v else "—",
             *z_to_badge(hist_z("dj_rsi", rsi_v))),
        ]

        radar_bytes = chart_radar(p_axis, t_axis,
                                  title=f"{player} vs Team Ø")

        dec_map = DEC_MAP
        col_w_p = [3.8*cm] + [min(2.8*cm, (W_BODY-3.8*cm)/len(p_sessions))] * len(p_sessions)
        hrow = [Paragraph("<b>Test</b>", S["body"])] + \
               [Paragraph(f"<b>{s}</b>", S["body"]) for s in p_sessions]
        p_rows = [hrow]
        for grp, metrics in [
            ("Sprint", ["t5","t10","t20","t30"]),
            ("Power",  ["cmj","dj_rsi"]),
            ("Fitness",["agility","dribbling","vo2max"]),
        ]:
            first = True
            for mk in metrics:
                r_data = [Paragraph(
                    f"<b>{grp}</b> · {METRIC_LABELS.get(mk,mk)}" if first
                    else METRIC_LABELS.get(mk, mk), S["body"])]
                first = False
                for s in p_sessions:
                    rec = p_df[p_df["session"] == s]
                    v = pd.to_numeric(rec.iloc[0].get(mk), errors="coerce") \
                        if not rec.empty else None
                    r_data.append(Paragraph(
                        fmt(v, dec_map.get(mk, 2)) if v is not None and pd.notna(v) else "X",
                        S["body"]))
                p_rows.append(r_data)

        insights_p = auto_insights(df, last_p_sess, player)
        recs_p     = training_recommendations(player, df, last_p_sess)

        block = [
            Paragraph(f"<b>{player}</b>", S["section"]),
            Paragraph(f"Sessions: {' · '.join(p_sessions)}  ·  Letzte: {last_p_sess}",
                      S["small"]),
            Spacer(1, 0.1 * cm),
            score_card_table(p_cards),
            Spacer(1, 0.15 * cm),
        ]

        # Radar + table side by side
        radar_img = img_flowable(radar_bytes, width_cm=8.5)
        tbl = styled_table(p_rows, col_w_p)
        side_table = Table(
            [[radar_img, tbl]],
            colWidths=[9 * cm, W_BODY - 9 * cm],
        )
        side_table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ]))
        block.append(side_table)
        block.append(Spacer(1, 0.15 * cm))

        if insights_p:
            block.append(Paragraph("<b>Performance-Analyse</b>", S["sub"]))
            for ins in insights_p:
                block.append(Paragraph(ins, S["insight"]))

        if recs_p:
            block.append(Spacer(1, 0.1 * cm))
            block.append(Paragraph("<b>Trainingsempfehlungen</b>", S["sub"]))
            for rec in recs_p:
                block.append(Paragraph(rec, S["body"]))

        block.append(Spacer(1, 0.45 * cm))
        block.append(HRFlowable(width="100%", thickness=0.75,
                                 color=colors.HexColor("#CCCCCC"),
                                 spaceAfter=2, spaceBefore=2))
        story.append(KeepTogether(block))

    doc.build(story,
              onFirstPage=make_footer_canvas(last_sess),
              onLaterPages=make_footer_canvas(last_sess))
    buf.seek(0)
    return buf.read()


# ── PLAYER PDF ─────────────────────────────────────────────────────────────────

def generate_player_pdf(df: pd.DataFrame, player: str) -> bytes:
    S = style()
    buf = io.BytesIO()
    ALL_SESSIONS = sorted(df["session"].unique().tolist())
    last_sess = ALL_SESSIONS[-1]

    p_df = df[df["name"] == player].sort_values("session")
    if p_df.empty:
        buf.write(b"No data")
        return buf.getvalue()

    p_sessions  = p_df["session"].tolist()
    last_p_sess = p_sessions[-1]
    p_last_row  = p_df[p_df["session"] == last_p_sess].iloc[0]
    last_df     = df[df["session"] == last_sess]

    # Build comparison scores
    p_axis = player_axis_scores(p_last_row)
    team_axis = {}
    for ax, metrics in RADAR_AXES.items():
        z_vals = []
        for m in metrics:
            col = pd.to_numeric(last_df[m], errors="coerce").dropna()
            z_vals += [hist_z(m, v) for v in col if hist_z(m, v) is not None]
        team_axis[ax] = round(np.mean(z_vals), 1) if z_vals else 100

    overall_z = np.mean([v for v in p_axis.values() if v is not None] or [100])
    badge, bc  = z_to_badge(overall_z)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=1.6 * cm,
    )
    story = []

    # ── COVER / HEADER ──────────────────────────────────────────────────────
    story += cover_page_elements(
        player, "Individual Athletic Profile · BVB Frauen",
        last_p_sess, len(p_sessions), S,
    )
    story.append(Spacer(1, 0.4 * cm))

    # Score cards row 1
    cmj_v  = p_last_row.get("cmj")
    t5_v   = p_last_row.get("t5")
    t20_v  = p_last_row.get("t20")
    vo2_v  = p_last_row.get("vo2max")
    rsi_v  = p_last_row.get("dj_rsi")
    agl_v  = p_last_row.get("agility")

    cards = [
        ("OVERALL SCORE", f"Z = {overall_z:.0f}", badge, bc),
        ("CMJ HEIGHT",   fmt(cmj_v,1)+" cm" if cmj_v else "—",
         *z_to_badge(hist_z("cmj", cmj_v))),
        ("SPRINT 20m",   fmt(t20_v,2)+" s" if t20_v else "—",
         *z_to_badge(hist_z("t20", t20_v))),
        ("VO2MAX",       fmt(vo2_v,1) if vo2_v else "—",
         *z_to_badge(hist_z("vo2max", vo2_v))),
        ("DJ RSI",       fmt(rsi_v,2) if rsi_v else "—",
         *z_to_badge(hist_z("dj_rsi", rsi_v))),
    ]
    story.append(score_card_table(cards))
    story.append(Spacer(1, 0.3 * cm))

    # ── RADAR + PERCENTILE BAR ───────────────────────────────────────────────
    rl_section(story, "Individual Athletic Profile", S)

    radar_bytes = chart_radar(p_axis, team_axis,
                              title=f"{player.split()[-1]} vs Team Ø (Z-Score)")
    pct_bytes   = chart_percentile_bar(player, df, last_p_sess)

    if pct_bytes:
        side = Table(
            [[img_flowable(radar_bytes, 8.5),
              img_flowable(pct_bytes, 8.5)]],
            colWidths=[9*cm, W_BODY-9*cm],
        )
        side.setStyle(TableStyle([
            ("VALIGN", (0,0),(-1,-1), "TOP"),
            ("LEFTPADDING",(0,0),(-1,-1), 0),
            ("RIGHTPADDING",(0,0),(-1,-1), 0),
        ]))
        story.append(side)
    else:
        story.append(img_flowable(radar_bytes, 10))

    story.append(Paragraph(
        "Links: Sechsachsen-Radar (Spielerin gelb, Team Ø grau). "
        "Rechts: Z-Score je Metrik (grün=stark, rot=Entwicklungsfeld).",
        S["caption"]))
    story.append(Spacer(1, 0.2 * cm))

    # ── TREND (if multiple sessions) ─────────────────────────────────────────
    if len(p_sessions) >= 2:
        rl_section(story, "Performance Trend", S)
        trend_bytes = chart_trend(player, df)
        if trend_bytes:
            story.append(img_flowable(trend_bytes, 16))
            story.append(Paragraph(
                "Z-Score-Entwicklung über alle Sessions (Referenz: historischer Mannschaftsdurchschnitt).",
                S["caption"]))
            story.append(Spacer(1, 0.2 * cm))

    # ── DATA TABLE ─────────────────────────────────────────────────────────
    rl_section(story, "Messdaten · Vollständige Historie", S)

    col_w_p = [3.8*cm] + [min(2.8*cm, (W_BODY-3.8*cm)/len(p_sessions))]*len(p_sessions)
    hrow = [Paragraph("<b>Test</b>", S["body"])] + \
           [Paragraph(f"<b>{s}</b>", S["body"]) for s in p_sessions]
    p_rows = [hrow]

    for grp, metrics in [
        ("Sprint",  ["t5","t10","t20","t30"]),
        ("Power",   ["cmj","dj_rsi"]),
        ("Fitness", ["agility","dribbling","vo2max"]),
    ]:
        first = True
        for mk in metrics:
            r_data = [Paragraph(
                f"<b>{grp}</b> · {METRIC_LABELS.get(mk,mk)}" if first
                else METRIC_LABELS.get(mk, mk), S["body"])]
            first = False
            for s in p_sessions:
                rec = p_df[p_df["session"] == s]
                v = pd.to_numeric(rec.iloc[0].get(mk), errors="coerce") \
                    if not rec.empty else None
                r_data.append(Paragraph(
                    fmt(v, DEC_MAP.get(mk, 2)) if v is not None and pd.notna(v) else "X",
                    S["body"]))
            p_rows.append(r_data)

    story.append(styled_table(p_rows, col_w_p))
    story.append(Spacer(1, 0.3 * cm))

    # ── AUTO INSIGHTS ─────────────────────────────────────────────────────────
    rl_section(story, "Performance-Analyse", S)
    insights = auto_insights(df, last_p_sess, player)
    if insights:
        for ins in insights:
            story.append(Paragraph(ins, S["insight"]))
    else:
        story.append(Paragraph("Ausgeglichenes Leistungsprofil in dieser Session.", S["body"]))

    story.append(Spacer(1, 0.2 * cm))

    # ── TRAINING RECOMMENDATIONS ───────────────────────────────────────────
    rl_section(story, "Trainingsempfehlungen", S)
    recs = training_recommendations(player, df, last_p_sess)
    for rec in recs:
        story.append(Paragraph(rec, S["body"]))

    doc.build(story,
              onFirstPage=make_footer_canvas(last_p_sess),
              onLaterPages=make_footer_canvas(last_p_sess))
    buf.seek(0)
    return buf.read()



# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
# ── Database init ────────────────────────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = BVBDatabase()
    st.session_state.db.init()   # creates tables + seeds if empty

db = st.session_state.db

# Load fresh from DB on every run — persistent across restarts
df        = get_df()
sessions  = db.get_sessions()  # chronological order from DB
players   = sorted(df[df["cmj"].notna()]["name"].unique().tolist())
last_sess = sessions[-1] if sessions else None
prev_sess = sessions[-2] if len(sessions) > 1 else None

# ── Empty database guard ─────────────────────────────────────────────
if not sessions:
    st.title("🟡 BVB Frauen · Performance Analytics")
    st.divider()
    st.info(
        "**Datenbank ist leer.** Lade deine erste Session hoch um zu starten."
    )
    st.markdown("### 📂 Erste Session laden")
    new_sess_label = st.text_input("Session-Name", placeholder='z.B. "Jan 26"')
    uploaded_first = st.file_uploader(
        "Messprotokoll Excel", type=["xlsx", "xls"], key="first_upload"
    )
    if st.button("📥 Laden", disabled=not (uploaded_first and new_sess_label.strip())):
        with st.spinner("Verarbeite..."):
            parsed = parse_excel(uploaded_first, new_sess_label.strip())
            if parsed.empty:
                st.error("Keine Daten erkannt. Stelle sicher dass die Datei "
                         "das BVB Frauen Messprotokoll Format hat.")
            else:
                db.upsert_session_from_df(parsed, new_sess_label.strip())
                st.success(f"✓ {len(parsed)} Spielerinnen geladen!")
                st.rerun()
    st.stop()  # Don't render the rest of the app until data exists

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
                # Save directly to DB — persists across restarts
                db.upsert_session_from_df(parsed, new_sess_label.strip())
                st.success(f"✓ {len(parsed)} Spielerinnen in DB gespeichert")
                st.rerun()

    st.divider()
    # Remove uploaded (non-seed) sessions
    if sessions:
        to_remove = st.selectbox("Session entfernen", ["—"] + sessions)
        if st.button("🗑 Entfernen") and to_remove != "—":
            db.delete_session(to_remove)
            st.rerun()

    st.divider()
    st.caption(f"v1.0 · {len(sessions)} Sessions · {len(players)} Spielerinnen")
    
    # Show Supabase fix reminder if RSI looks wrong
    if not df.empty and "dj_rsi" in df.columns:
        sample_rsi = df["dj_rsi"].dropna()
        if len(sample_rsi) > 0 and sample_rsi.mean() > 10:
            st.warning(
                "⚠️ RSI-Werte scheinen falsch (>10). "
                "Bitte in Supabase SQL ausführen:\n"
                "```sql\nUPDATE measurements\n"
                "SET dj_rsi = ROUND((dvj_hoehe::numeric/100)/(dvj_kontaktzeit::numeric/1000),3)\n"
                "WHERE dvj_kontaktzeit IS NOT NULL AND dvj_hoehe IS NOT NULL;\n```"
            )

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_overview, tab_player, tab_compare, tab_ranking, tab_saeulen, tab_flags, tab_pdf, tab_squad = st.tabs([
    "📊 Überblick",
    "👤 Spielerin",
    "⚡ Vergleich",
    "🏆 Rankings",
    "📊 Säulendiagramme",
    "🚨 Verletzungsrisiko",
    "📄 PDF Reports",
    "👥 Spielerinnen",
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
        st.plotly_chart(team_radar_chart(df), width='stretch', key="team_radar")
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
                    width='stretch'
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
                st.plotly_chart(fig_trend, width='stretch', key="team_trend")
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
            cl = cluster_info[cluster_info["name"] == selected_player] if not cluster_info.empty and "name" in cluster_info.columns else pd.DataFrame()

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
                                           title=f"{selected_player} · Radar"), width='stretch', key=f"player_radar_{selected_player}")
            with c2:
                st.plotly_chart(comparison_bar(df, selected_player, selected_session),
                                width='stretch', key=f"player_comp_{selected_player}_{selected_session}")

            st.plotly_chart(trend_chart(df, selected_player), width='stretch', key=f"player_trend_{selected_player}")

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
                st.dataframe(raw_df, width='stretch')

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
                    pred_df.style.map(
                        lambda v: "color: #4ADE80" if v == "↑" else ("color: #F87171" if v == "↓" else ""),
                        subset=["Trend"]
                    ),
                    width='stretch'
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
                                       title="Radar Vergleich"), width='stretch', key="cmp_radar")
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
                st.plotly_chart(fig_cmp, width='stretch', key=f"cmp_bar_{cmp_metric}")

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
        st.dataframe(det_df, width='stretch')

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
            st.plotly_chart(fig_tl, width='stretch', key=f"cmp_trend_{tl_metric}")

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
    rk_df[rk_metric] = pd.to_numeric(rk_df[rk_metric], errors="coerce")
    rk_df = rk_df[rk_df[rk_metric].notna()]
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
    st.dataframe(rk_display, width='stretch', height=600)


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
            width='stretch',
            key=f"saeul_main_{saeulen_metric}",
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
                                       xaxis_tickfont_size=7)
                fig_mini.update_layout(margin=dict(l=10, r=10, t=40, b=60))
                st.plotly_chart(fig_mini, width='stretch', key=f"saeul_mini_{m}")
            with col_b:
                if i + 1 < len(metrics_list):
                    m2 = metrics_list[i + 1]
                    fig_mini2 = ranked_bar_chart(df, m2, saeulen_sessions)
                    fig_mini2.update_layout(height=320, title_font_size=12,
                                            showlegend=False,
                                            xaxis_tickfont_size=7)
                    fig_mini2.update_layout(margin=dict(l=10, r=10, t=40, b=60))
                    st.plotly_chart(fig_mini2, width='stretch', key=f"saeul_mini_{m2}")

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
                width='stretch',
                height=500,
            )

# ══════════════════════════════════════════════
# PDF REPORTS
# ══════════════════════════════════════════════
with tab_pdf:
    st.markdown("### 📄 PDF Reports")

    # ── FULL TEAM REPORT ──────────────────────────────────────────────
    st.markdown("#### Mannschaftsbericht")
    st.caption(
        "Vollständiger Bericht im Format der Auswertung Leistungsdiagnostik: "
        "Übersicht je Session · Ranglisten je Disziplin · Individuelle Profile aller Spielerinnen."
    )

    if st.button("📥 Mannschaftsbericht generieren", type="primary"):
        with st.spinner("Erstelle PDF..."):
            try:
                pdf_bytes = generate_team_pdf(df)
                st.download_button(
                    label="⬇️ Mannschaftsbericht herunterladen",
                    data=pdf_bytes,
                    file_name=f"BVB_Frauen_Leistungsdiagnostik_{last_sess.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    key="dl_team",
                )
                st.success("✓ PDF erstellt — Download bereit!")
            except Exception as e:
                st.error(f"Fehler beim Generieren: {e}")

    st.divider()

    # ── INDIVIDUAL PLAYER REPORTS ─────────────────────────────────────
    st.markdown("#### Individuelle Spielerinnen-Reports")
    st.caption("Einzelbericht pro Spielerin mit vollständiger Messhistorie.")

    col_dl, col_all = st.columns([2, 1])
    with col_dl:
        pdf_player = st.selectbox("Spielerin auswählen", players, key="pdf_player_sel")
    with col_all:
        st.markdown("")
        st.markdown("")
        gen_single = st.button("📥 Spielerin-PDF", key="gen_single_pdf")

    if gen_single and pdf_player:
        with st.spinner(f"Erstelle PDF für {pdf_player}..."):
            try:
                pdf_bytes = generate_player_pdf(df, pdf_player)
                st.download_button(
                    label=f"⬇️ {pdf_player} herunterladen",
                    data=pdf_bytes,
                    file_name=f"BVB_{pdf_player.replace(' ','_')}_{last_sess.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    key="dl_single",
                )
                st.success(f"✓ PDF für {pdf_player} erstellt!")
            except Exception as e:
                st.error(f"Fehler: {e}")

    st.divider()

    # ── ALL PLAYERS AS ZIP ────────────────────────────────────────────
    st.markdown("#### Alle Spielerinnen als ZIP")
    st.caption("Generiert für jede Spielerin eine eigene PDF-Datei und packt sie in ein ZIP.")

    if st.button("📦 Alle Spielerinnen-PDFs als ZIP", key="gen_zip"):
        import zipfile, io as _io
        with st.spinner("Erstelle PDFs für alle Spielerinnen..."):
            try:
                zip_buf = _io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    progress = st.progress(0)
                    for i, p in enumerate(players):
                        pdf_b = generate_player_pdf(df, p)
                        zf.writestr(
                            f"BVB_{p.replace(' ','_')}_{last_sess.replace(' ','_')}.pdf",
                            pdf_b,
                        )
                        progress.progress((i + 1) / len(players))
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ ZIP herunterladen",
                    data=zip_buf.read(),
                    file_name=f"BVB_Spielerinnen_{last_sess.replace(' ','_')}.zip",
                    mime="application/zip",
                    key="dl_zip",
                )
                st.success(f"✓ {len(players)} PDFs im ZIP!")
            except Exception as e:
                st.error(f"Fehler: {e}")


# ══════════════════════════════════════════════
# SPIELERINNEN MANAGEMENT
# ══════════════════════════════════════════════
with tab_squad:
    st.markdown("### Spielerinnen verwalten")
    st.caption("Kader ändert sich automatisch — neue Spielerinnen erscheinen nach dem ersten Upload, "
               "ehemalige werden hier entfernt.")

    # ── Current squad overview ──────────────────
    st.markdown("#### Aktueller Kader")

    # Build per-player session summary
    squad_rows = []
    for name in players:
        p_sessions = db.get_player_sessions(name)
        last_rec = df[(df["name"] == name)].sort_values("session").iloc[-1] if not df[df["name"]==name].empty else None
        squad_rows.append({
            "Spielerin":    name,
            "Sessions":     len(p_sessions),
            "Erste Session": p_sessions[0] if p_sessions else "—",
            "Letzte Session": p_sessions[-1] if p_sessions else "—",
            "Im Kader seit": p_sessions[0] if p_sessions else "—",
        })

    if squad_rows:
        squad_df = pd.DataFrame(squad_rows).set_index("Spielerin")
        st.dataframe(squad_df, width='stretch')
    else:
        st.info("Noch keine Spielerinnen. Lade eine Session um zu starten.")
    st.caption(f"**{len(players)} Spielerinnen** mit Messdaten · "
               f"Spielerinnen werden automatisch hinzugefügt wenn sie in einem Upload erscheinen.")

    st.divider()

    col_a, col_b = st.columns(2)

    # ── Rename player ───────────────────────────
    with col_a:
        st.markdown("#### Spielerin umbenennen")
        st.caption("Z.B. bei Heirat, Tippfehler im Namen oder Spitzname im Excel")
        rename_from = st.selectbox("Spielerin", ["— auswählen —"] + players, key="rename_from")
        rename_to   = st.text_input("Neuer Name", key="rename_to",
                                     placeholder="Korrekter vollständiger Name")
        if st.button("✏️ Umbenennen", disabled=(rename_from == "— auswählen —" or not rename_to.strip())):
            db.rename_player(rename_from, rename_to.strip())
            st.success(f"✓ '{rename_from}' → '{rename_to.strip()}'")
            st.rerun()

    # ── Remove player ───────────────────────────
    with col_b:
        st.markdown("#### Spielerin entfernen")
        st.caption("Entfernt die Spielerin und ALLE ihre Messdaten dauerhaft. "
                   "Nur verwenden wenn die Spielerin den Verein verlassen hat.")
        remove_player = st.selectbox("Spielerin", ["— auswählen —"] + players, key="remove_player")
        if remove_player != "— auswählen —":
            p_sess = db.get_player_sessions(remove_player)
            st.warning(f"⚠ Löscht **{len(p_sess)} Session(en)** mit Daten: {', '.join(p_sess)}")
        confirm = st.checkbox("Ich bestätige die dauerhafte Löschung", key="confirm_delete")
        if st.button("🗑 Dauerhaft entfernen",
                     disabled=(remove_player == "— auswählen —" or not confirm),
                     type="primary"):
            db.delete_player(remove_player)
            st.success(f"✓ '{remove_player}' und alle Messdaten gelöscht.")
            st.rerun()

    st.divider()

    # ── DB maintenance ──────────────────────────
    st.markdown("#### Datenbank-Wartung")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Duplikate bereinigen**")
        st.caption("Entfernt Spieler-Einträge ohne Messdaten "
                   "(entstehen bei fehlgeschlagenen Uploads).")
        if st.button("🧹 Bereinigen"):
            n = db.clean_orphan_players()
            st.success(f"✓ {n} verwaiste Einträge entfernt.")
            st.rerun()

    with col_d:
        st.markdown("**Datenbank zurücksetzen**")
        st.caption("⚠ Löscht ALLE Daten und seedet neu. Nur bei korruptem Zustand verwenden.")
        confirm_reset = st.checkbox("Ich verstehe dass alle Daten gelöscht werden", key="confirm_reset")
        if st.button("💣 Zurücksetzen & neu seeden",
                     disabled=not confirm_reset, type="primary"):
            with st.spinner("Setze zurück..."):
                db.reset_and_reseed()
            st.success("✓ Datenbank zurückgesetzt und neu geseedet.")
            st.rerun()
