# BVB Frauen — Performance Analytics

## Schnellstart (2 Minuten)

### 1. Pakete installieren
```bash
pip install -r requirements.txt
```

### 2. App starten
```bash
streamlit run app.py
```
Browser öffnet automatisch → `http://localhost:8501`

---

## Online teilen (kostenlos)

1. Kostenloses Konto auf [streamlit.io/cloud](https://streamlit.io/cloud)
2. Code auf GitHub hochladen (privates Repo reicht)
3. "New app" → Repo auswählen → Deploy
4. Fertig — öffentliche URL wie `bvb-analytics.streamlit.app` zum Teilen

---

## Struktur

```
bvb_system/
├── app.py              ← Streamlit App (alles drin)
├── bvb_engine.py       ← Python Engine (für Jupyter / server)
├── server.py           ← Flask API (optional)
├── BVB_Analytics.ipynb ← Jupyter Notebook
└── requirements.txt
```

## Features

| Tab | Inhalt |
|-----|--------|
| 📊 Überblick | KPIs, Team Radar, Meistverbesserte, Cluster |
| 👤 Spielerin | Radar, Trends, Rohdaten, Auswertung, Prognose |
| ⚡ Vergleich | Bis zu 6 Spielerinnen, Zeitverlauf |
| 🏆 Rankings | Alle Metriken, alle Sessions |
| 🚨 Verletzungsrisiko | Automatische Flags, Z-Score Heatmap |
| 📄 PDF Reports | Einzeln oder alle als ZIP |

## Neue Session laden

Sidebar → Session-Name eingeben → Excel (Messprotokoll) hochladen → Laden klicken.
Z-Scores werden automatisch neu berechnet.
