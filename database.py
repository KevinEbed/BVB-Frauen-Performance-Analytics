"""
BVB Frauen — Database Layer
============================
Works with:
  - SQLite  (local development — default, zero setup)
  - PostgreSQL via Supabase (cloud deployment — set DATABASE_URL in .streamlit/secrets.toml)

Usage:
  db = BVBDatabase()          # auto-detects SQLite or PostgreSQL
  db.init()                   # create tables if not exists + seed data
  df = db.load_dataframe()    # returns full pd.DataFrame for the app
  db.upsert_session(df, label, date)   # save a new/updated session
  db.delete_session(label)    # remove a session
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path

# ─────────────────────────────────────────────────────────
# SEED DATA  (Jan 26 + Mär 26)
# ─────────────────────────────────────────────────────────
SEED_RECORDS = [
    # Jan 26
    {"name":"Laura van der Laan","session":"Jan 26","date":"2026-01-01",
     "cmj_1":28.8,"cmj_2":29.8,"cmj_3":27.7,
     "dvj_kontaktzeit":181,"dvj_hoehe":33.3,
     "sprint_t5_r1":1.08,"sprint_t10_r1":1.91,"sprint_t20_r1":3.31,"sprint_t30_r1":4.67,
     "sprint_t5_r2":1.02,"sprint_t10_r2":1.88,"sprint_t20_r2":3.28,"sprint_t30_r2":4.65,
     "agility_r1":16.07,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.63,
     "yoyo_level":10,"yoyo_shuttles":10,"hf_max":None},
    {"name":"Mia Böger","session":"Jan 26","date":"2026-01-01",
     "cmj_1":34.3,"cmj_2":35.9,"cmj_3":34.6,
     "dvj_kontaktzeit":175,"dvj_hoehe":31.8,
     "sprint_t5_r1":0.97,"sprint_t10_r1":1.73,"sprint_t20_r1":3.09,"sprint_t30_r1":4.37,
     "sprint_t5_r2":0.97,"sprint_t10_r2":1.73,"sprint_t20_r2":3.09,"sprint_t30_r2":4.37,
     "agility_r1":15.37,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.83,
     "yoyo_level":11,"yoyo_shuttles":2,"hf_max":None},
    {"name":"Jenske Steenjwijk","session":"Jan 26","date":"2026-01-01",
     "cmj_1":30.3,"cmj_2":30.9,"cmj_3":29.6,
     "dvj_kontaktzeit":187,"dvj_hoehe":32.0,
     "sprint_t5_r1":1.04,"sprint_t10_r1":1.85,"sprint_t20_r1":3.24,"sprint_t30_r1":4.58,
     "sprint_t5_r2":1.04,"sprint_t10_r2":1.85,"sprint_t20_r2":3.24,"sprint_t30_r2":4.58,
     "agility_r1":15.48,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.10,
     "yoyo_level":9,"yoyo_shuttles":12,"hf_max":None},
    {"name":"Jasmin Jabbes","session":"Jan 26","date":"2026-01-01",
     "cmj_1":33.4,"cmj_2":31.6,"cmj_3":33.4,
     "dvj_kontaktzeit":170,"dvj_hoehe":29.7,
     "sprint_t5_r1":1.11,"sprint_t10_r1":1.84,"sprint_t20_r1":3.24,"sprint_t30_r1":4.46,
     "sprint_t5_r2":1.08,"sprint_t10_r2":1.85,"sprint_t20_r2":3.19,"sprint_t30_r2":4.46,
     "agility_r1":15.34,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.88,
     "yoyo_level":13,"yoyo_shuttles":12,"hf_max":None},
    {"name":"Noreen Günnewig","session":"Jan 26","date":"2026-01-01",
     "cmj_1":30.0,"cmj_2":32.3,"cmj_3":30.3,
     "dvj_kontaktzeit":180,"dvj_hoehe":31.1,
     "sprint_t5_r1":1.12,"sprint_t10_r1":1.93,"sprint_t20_r1":3.46,"sprint_t30_r1":4.80,
     "sprint_t5_r2":1.12,"sprint_t10_r2":1.93,"sprint_t20_r2":3.46,"sprint_t30_r2":4.80,
     "agility_r1":15.89,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.26,
     "yoyo_level":10,"yoyo_shuttles":6,"hf_max":None},
    {"name":"Nova Wicke","session":"Jan 26","date":"2026-01-01",
     "cmj_1":28.5,"cmj_2":28.9,"cmj_3":28.0,
     "dvj_kontaktzeit":214,"dvj_hoehe":28.2,
     "sprint_t5_r1":1.03,"sprint_t10_r1":1.82,"sprint_t20_r1":3.19,"sprint_t30_r1":4.47,
     "sprint_t5_r2":1.03,"sprint_t10_r2":1.82,"sprint_t20_r2":3.19,"sprint_t30_r2":4.47,
     "agility_r1":15.65,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.43,
     "yoyo_level":8,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Lucy Wisniewski","session":"Jan 26","date":"2026-01-01",
     "cmj_1":28.5,"cmj_2":28.8,"cmj_3":28.0,
     "dvj_kontaktzeit":177,"dvj_hoehe":28.7,
     "sprint_t5_r1":1.07,"sprint_t10_r1":1.91,"sprint_t20_r1":3.36,"sprint_t30_r1":4.73,
     "sprint_t5_r2":1.07,"sprint_t10_r2":1.91,"sprint_t20_r2":3.36,"sprint_t30_r2":4.73,
     "agility_r1":16.13,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.06,
     "yoyo_level":9,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Ronja Leubner","session":"Jan 26","date":"2026-01-01",
     "cmj_1":35.6,"cmj_2":35.0,"cmj_3":33.8,
     "dvj_kontaktzeit":294,"dvj_hoehe":35.6,
     "sprint_t5_r1":1.03,"sprint_t10_r1":1.82,"sprint_t20_r1":3.20,"sprint_t30_r1":4.56,
     "sprint_t5_r2":1.03,"sprint_t10_r2":1.82,"sprint_t20_r2":3.20,"sprint_t30_r2":4.56,
     "agility_r1":15.38,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.33,
     "yoyo_level":10,"yoyo_shuttles":10,"hf_max":None},
    {"name":"Nora Willeke","session":"Jan 26","date":"2026-01-01",
     "cmj_1":32.8,"cmj_2":32.0,"cmj_3":31.5,
     "dvj_kontaktzeit":225,"dvj_hoehe":32.9,
     "sprint_t5_r1":1.06,"sprint_t10_r1":1.85,"sprint_t20_r1":3.25,"sprint_t30_r1":4.57,
     "sprint_t5_r2":1.06,"sprint_t10_r2":1.85,"sprint_t20_r2":3.25,"sprint_t30_r2":4.57,
     "agility_r1":15.44,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.80,
     "yoyo_level":12,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Celina Baum","session":"Jan 26","date":"2026-01-01",
     "cmj_1":31.5,"cmj_2":30.5,"cmj_3":30.0,
     "dvj_kontaktzeit":208,"dvj_hoehe":31.4,
     "sprint_t5_r1":1.04,"sprint_t10_r1":1.86,"sprint_t20_r1":3.25,"sprint_t30_r1":4.60,
     "sprint_t5_r2":1.04,"sprint_t10_r2":1.86,"sprint_t20_r2":3.25,"sprint_t30_r2":4.60,
     "agility_r1":16.15,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.80,
     "yoyo_level":9,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Annika Enderle","session":"Jan 26","date":"2026-01-01",
     "cmj_1":33.0,"cmj_2":32.0,"cmj_3":31.8,
     "dvj_kontaktzeit":198,"dvj_hoehe":33.1,
     "sprint_t5_r1":0.98,"sprint_t10_r1":1.84,"sprint_t20_r1":3.17,"sprint_t30_r1":4.40,
     "sprint_t5_r2":0.98,"sprint_t10_r2":1.84,"sprint_t20_r2":3.17,"sprint_t30_r2":4.40,
     "agility_r1":15.21,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.72,
     "yoyo_level":10,"yoyo_shuttles":10,"hf_max":None},
    {"name":"Melanie Schuster","session":"Jan 26","date":"2026-01-01",
     "cmj_1":29.0,"cmj_2":28.5,"cmj_3":28.0,
     "dvj_kontaktzeit":200,"dvj_hoehe":29.0,
     "sprint_t5_r1":1.12,"sprint_t10_r1":1.99,"sprint_t20_r1":3.44,"sprint_t30_r1":4.84,
     "sprint_t5_r2":1.12,"sprint_t10_r2":1.99,"sprint_t20_r2":3.44,"sprint_t30_r2":4.84,
     "agility_r1":15.83,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.87,
     "yoyo_level":9,"yoyo_shuttles":6,"hf_max":None},
    {"name":"Caroline Blum","session":"Jan 26","date":"2026-01-01",
     "cmj_1":30.8,"cmj_2":29.5,"cmj_3":29.0,
     "dvj_kontaktzeit":238,"dvj_hoehe":30.8,
     "sprint_t5_r1":1.09,"sprint_t10_r1":2.03,"sprint_t20_r1":3.52,"sprint_t30_r1":4.97,
     "sprint_t5_r2":1.09,"sprint_t10_r2":2.03,"sprint_t20_r2":3.52,"sprint_t30_r2":4.97,
     "agility_r1":16.14,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.63,
     "yoyo_level":9,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Paula Reimann","session":"Jan 26","date":"2026-01-01",
     "cmj_1":29.9,"cmj_2":29.0,"cmj_3":28.5,
     "dvj_kontaktzeit":158,"dvj_hoehe":29.9,
     "sprint_t5_r1":1.11,"sprint_t10_r1":1.96,"sprint_t20_r1":3.43,"sprint_t30_r1":4.84,
     "sprint_t5_r2":1.11,"sprint_t10_r2":1.96,"sprint_t20_r2":3.43,"sprint_t30_r2":4.84,
     "agility_r1":16.37,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.91,
     "yoyo_level":10,"yoyo_shuttles":6,"hf_max":None},
    {"name":"Yasu Woehrn","session":"Jan 26","date":"2026-01-01",
     "cmj_1":35.5,"cmj_2":34.5,"cmj_3":34.0,
     "dvj_kontaktzeit":207,"dvj_hoehe":35.5,
     "sprint_t5_r1":1.05,"sprint_t10_r1":1.89,"sprint_t20_r1":3.34,"sprint_t30_r1":4.69,
     "sprint_t5_r2":1.05,"sprint_t10_r2":1.89,"sprint_t20_r2":3.34,"sprint_t30_r2":4.69,
     "agility_r1":15.38,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.66,
     "yoyo_level":10,"yoyo_shuttles":8,"hf_max":None},
    # Mär 26
    {"name":"Laura van der Laan","session":"Mär 26","date":"2026-03-01",
     "cmj_1":29.4,"cmj_2":29.3,"cmj_3":28.8,
     "dvj_kontaktzeit":193,"dvj_hoehe":34.7,
     "sprint_t5_r1":None,"sprint_t10_r1":None,"sprint_t20_r1":None,"sprint_t30_r1":None,
     "sprint_t5_r2":None,"sprint_t10_r2":None,"sprint_t20_r2":None,"sprint_t30_r2":None,
     "agility_r1":None,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":None,
     "yoyo_level":None,"yoyo_shuttles":None,"hf_max":None},
    {"name":"Mia Böger","session":"Mär 26","date":"2026-03-01",
     "cmj_1":37.2,"cmj_2":36.8,"cmj_3":35.0,
     "dvj_kontaktzeit":177,"dvj_hoehe":31.8,
     "sprint_t5_r1":None,"sprint_t10_r1":None,"sprint_t20_r1":None,"sprint_t30_r1":None,
     "sprint_t5_r2":None,"sprint_t10_r2":None,"sprint_t20_r2":None,"sprint_t30_r2":None,
     "agility_r1":None,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":None,
     "yoyo_level":None,"yoyo_shuttles":None,"hf_max":None},
    {"name":"Jasmin Jabbes","session":"Mär 26","date":"2026-03-01",
     "cmj_1":33.1,"cmj_2":34.0,"cmj_3":34.8,
     "dvj_kontaktzeit":186,"dvj_hoehe":32.8,
     "sprint_t5_r1":1.11,"sprint_t10_r1":1.85,"sprint_t20_r1":3.20,"sprint_t30_r1":4.43,
     "sprint_t5_r2":1.11,"sprint_t10_r2":1.85,"sprint_t20_r2":3.20,"sprint_t30_r2":4.43,
     "agility_r1":15.28,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.42,
     "yoyo_level":13,"yoyo_shuttles":14,"hf_max":None},
    {"name":"Noreen Günnewig","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.9,"cmj_2":32.1,"cmj_3":32.0,
     "dvj_kontaktzeit":185,"dvj_hoehe":33.0,
     "sprint_t5_r1":1.04,"sprint_t10_r1":1.84,"sprint_t20_r1":3.22,"sprint_t30_r1":4.56,
     "sprint_t5_r2":1.04,"sprint_t10_r2":1.84,"sprint_t20_r2":3.22,"sprint_t30_r2":4.56,
     "agility_r1":15.55,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.53,
     "yoyo_level":11,"yoyo_shuttles":2,"hf_max":None},
    {"name":"Marina Scholz","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.5,"cmj_2":32.1,"cmj_3":34.4,
     "dvj_kontaktzeit":162,"dvj_hoehe":34.5,
     "sprint_t5_r1":1.10,"sprint_t10_r1":1.90,"sprint_t20_r1":3.30,"sprint_t30_r1":4.57,
     "sprint_t5_r2":1.10,"sprint_t10_r2":1.90,"sprint_t20_r2":3.30,"sprint_t30_r2":4.57,
     "agility_r1":15.99,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.15,
     "yoyo_level":12,"yoyo_shuttles":6,"hf_max":None},
    {"name":"Rita Schumacher","session":"Mär 26","date":"2026-03-01",
     "cmj_1":38.4,"cmj_2":36.3,"cmj_3":39.0,
     "dvj_kontaktzeit":168,"dvj_hoehe":39.0,
     "sprint_t5_r1":0.97,"sprint_t10_r1":1.68,"sprint_t20_r1":3.00,"sprint_t30_r1":4.19,
     "sprint_t5_r2":0.97,"sprint_t10_r2":1.68,"sprint_t20_r2":3.00,"sprint_t30_r2":4.19,
     "agility_r1":14.82,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":8.65,
     "yoyo_level":11,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Sara Ito","session":"Mär 26","date":"2026-03-01",
     "cmj_1":29.3,"cmj_2":29.4,"cmj_3":30.5,
     "dvj_kontaktzeit":298,"dvj_hoehe":30.5,
     "sprint_t5_r1":None,"sprint_t10_r1":None,"sprint_t20_r1":None,"sprint_t30_r1":None,
     "sprint_t5_r2":None,"sprint_t10_r2":None,"sprint_t20_r2":None,"sprint_t30_r2":None,
     "agility_r1":None,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.35,
     "yoyo_level":11,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Lucy Wisniewski","session":"Mär 26","date":"2026-03-01",
     "cmj_1":30.5,"cmj_2":31.4,"cmj_3":30.3,
     "dvj_kontaktzeit":176,"dvj_hoehe":31.4,
     "sprint_t5_r1":1.13,"sprint_t10_r1":1.91,"sprint_t20_r1":3.32,"sprint_t30_r1":4.65,
     "sprint_t5_r2":1.13,"sprint_t10_r2":1.91,"sprint_t20_r2":3.32,"sprint_t30_r2":4.65,
     "agility_r1":16.05,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.97,
     "yoyo_level":11,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Ronja Leubner","session":"Mär 26","date":"2026-03-01",
     "cmj_1":37.8,"cmj_2":37.4,"cmj_3":37.2,
     "dvj_kontaktzeit":251,"dvj_hoehe":37.3,
     "sprint_t5_r1":0.97,"sprint_t10_r1":1.72,"sprint_t20_r1":3.04,"sprint_t30_r1":4.30,
     "sprint_t5_r2":0.97,"sprint_t10_r2":1.72,"sprint_t20_r2":3.04,"sprint_t30_r2":4.30,
     "agility_r1":14.96,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.34,
     "yoyo_level":12,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Nora Willeke","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.1,"cmj_2":33.8,"cmj_3":33.8,
     "dvj_kontaktzeit":191,"dvj_hoehe":33.8,
     "sprint_t5_r1":1.02,"sprint_t10_r1":1.74,"sprint_t20_r1":3.06,"sprint_t30_r1":4.29,
     "sprint_t5_r2":1.02,"sprint_t10_r2":1.74,"sprint_t20_r2":3.06,"sprint_t30_r2":4.29,
     "agility_r1":15.20,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.54,
     "yoyo_level":13,"yoyo_shuttles":0,"hf_max":None},
    {"name":"Celina Baum","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.6,"cmj_2":31.1,"cmj_3":30.2,
     "dvj_kontaktzeit":207,"dvj_hoehe":32.6,
     "sprint_t5_r1":1.07,"sprint_t10_r1":1.86,"sprint_t20_r1":3.25,"sprint_t30_r1":4.56,
     "sprint_t5_r2":1.07,"sprint_t10_r2":1.86,"sprint_t20_r2":3.25,"sprint_t30_r2":4.56,
     "agility_r1":16.00,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.28,
     "yoyo_level":12,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Annika Enderle","session":"Mär 26","date":"2026-03-01",
     "cmj_1":34.3,"cmj_2":32.5,"cmj_3":33.7,
     "dvj_kontaktzeit":194,"dvj_hoehe":34.3,
     "sprint_t5_r1":0.99,"sprint_t10_r1":1.81,"sprint_t20_r1":3.16,"sprint_t30_r1":4.42,
     "sprint_t5_r2":0.99,"sprint_t10_r2":1.81,"sprint_t20_r2":3.16,"sprint_t30_r2":4.42,
     "agility_r1":15.23,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.10,
     "yoyo_level":11,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Frederike Kempe","session":"Mär 26","date":"2026-03-01",
     "cmj_1":36.6,"cmj_2":35.9,"cmj_3":35.5,
     "dvj_kontaktzeit":164,"dvj_hoehe":36.7,
     "sprint_t5_r1":1.00,"sprint_t10_r1":1.86,"sprint_t20_r1":3.23,"sprint_t30_r1":4.53,
     "sprint_t5_r2":1.00,"sprint_t10_r2":1.86,"sprint_t20_r2":3.23,"sprint_t30_r2":4.53,
     "agility_r1":15.31,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.48,
     "yoyo_level":11,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Melanie Schuster","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.4,"cmj_2":32.5,"cmj_3":34.3,
     "dvj_kontaktzeit":215,"dvj_hoehe":34.3,
     "sprint_t5_r1":1.09,"sprint_t10_r1":1.89,"sprint_t20_r1":3.26,"sprint_t30_r1":4.53,
     "sprint_t5_r2":1.09,"sprint_t10_r2":1.89,"sprint_t20_r2":3.26,"sprint_t30_r2":4.53,
     "agility_r1":15.77,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.84,
     "yoyo_level":10,"yoyo_shuttles":8,"hf_max":None},
    {"name":"Caroline Blum","session":"Mär 26","date":"2026-03-01",
     "cmj_1":32.4,"cmj_2":33.3,"cmj_3":33.7,
     "dvj_kontaktzeit":244,"dvj_hoehe":33.8,
     "sprint_t5_r1":1.12,"sprint_t10_r1":1.91,"sprint_t20_r1":3.42,"sprint_t30_r1":4.81,
     "sprint_t5_r2":1.12,"sprint_t10_r2":1.91,"sprint_t20_r2":3.42,"sprint_t30_r2":4.81,
     "agility_r1":15.89,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.90,
     "yoyo_level":10,"yoyo_shuttles":4,"hf_max":None},
    {"name":"Paula Reimann","session":"Mär 26","date":"2026-03-01",
     "cmj_1":29.9,"cmj_2":31.5,"cmj_3":30.9,
     "dvj_kontaktzeit":155,"dvj_hoehe":31.5,
     "sprint_t5_r1":1.07,"sprint_t10_r1":1.80,"sprint_t20_r1":3.29,"sprint_t30_r1":4.63,
     "sprint_t5_r2":1.07,"sprint_t10_r2":1.80,"sprint_t20_r2":3.29,"sprint_t30_r2":4.63,
     "agility_r1":15.76,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.38,
     "yoyo_level":11,"yoyo_shuttles":2,"hf_max":None},
    {"name":"Yasu Woehrn","session":"Mär 26","date":"2026-03-01",
     "cmj_1":41.7,"cmj_2":42.0,"cmj_3":43.3,
     "dvj_kontaktzeit":225,"dvj_hoehe":43.2,
     "sprint_t5_r1":1.02,"sprint_t10_r1":1.81,"sprint_t20_r1":3.18,"sprint_t30_r1":4.44,
     "sprint_t5_r2":1.02,"sprint_t10_r2":1.81,"sprint_t20_r2":3.18,"sprint_t30_r2":4.44,
     "agility_r1":15.30,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":9.32,
     "yoyo_level":13,"yoyo_shuttles":6,"hf_max":None},
    {"name":"Deniese Lichatschow","session":"Mär 26","date":"2026-03-01",
     "cmj_1":34.7,"cmj_2":37.9,"cmj_3":36.8,
     "dvj_kontaktzeit":186,"dvj_hoehe":37.9,
     "sprint_t5_r1":1.13,"sprint_t10_r1":1.92,"sprint_t20_r1":3.33,"sprint_t30_r1":4.65,
     "sprint_t5_r2":1.13,"sprint_t10_r2":1.92,"sprint_t20_r2":3.33,"sprint_t30_r2":4.65,
     "agility_r1":16.40,"agility_r2":None,"dribbling_r1":None,"dribbling_r2":10.03,
     "yoyo_level":10,"yoyo_shuttles":10,"hf_max":None},
]


# ─────────────────────────────────────────────────────────
# DATABASE CLASS
# ─────────────────────────────────────────────────────────

class BVBDatabase:
    """
    Persistent database for BVB Frauen performance data.
    Auto-detects SQLite (local) or PostgreSQL (Supabase cloud).
    """

    def __init__(self):
        # Check both environment variable and Streamlit secrets
        self.db_url = os.environ.get("DATABASE_URL", "")

        # Also check st.secrets if running in Streamlit
        if not self.db_url:
            try:
                import streamlit as st
                self.db_url = st.secrets.get("DATABASE_URL", "")
            except Exception:
                pass

        self.use_postgres = bool(self.db_url) and (
            self.db_url.startswith("postgresql") or
            self.db_url.startswith("postgres")
        )

        if self.use_postgres:
            self.db_path = None
        else:
            # SQLite — store next to database.py
            self.db_path = Path(__file__).parent / "bvb_data.db"

    def _connect(self):
        if self.use_postgres:
            try:
                import psycopg2
                return psycopg2.connect(self.db_url)
            except ImportError:
                raise ImportError(
                    "psycopg2-binary is required for PostgreSQL. "
                    "Add it to requirements.txt"
                )
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def _ph(self):
        """Return the right placeholder — ? for SQLite, %s for PostgreSQL."""
        return "%s" if self.use_postgres else "?"

    # ── Schema ────────────────────────────────────────────

    def init(self):
        """Create tables if they don't exist, then seed if empty."""
        conn = self._connect()
        cur  = conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                label        TEXT    NOT NULL UNIQUE,
                session_date TEXT,
                created_at   TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS players (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL UNIQUE,
                created_at TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id          INTEGER NOT NULL REFERENCES players(id),
                session_id         INTEGER NOT NULL REFERENCES sessions(id),
                -- Raw attempts
                cmj_1              REAL, cmj_2 REAL, cmj_3 REAL,
                dvj_kontaktzeit    REAL, dvj_hoehe REAL,
                sprint_t5_r1       REAL, sprint_t10_r1 REAL,
                sprint_t20_r1      REAL, sprint_t30_r1 REAL,
                sprint_t5_r2       REAL, sprint_t10_r2 REAL,
                sprint_t20_r2      REAL, sprint_t30_r2 REAL,
                agility_r1         REAL, agility_r2 REAL,
                dribbling_r1       REAL, dribbling_r2 REAL,
                yoyo_level         REAL, yoyo_shuttles REAL,
                hf_max             REAL,
                -- Best / computed values
                cmj_best           REAL,
                dj_rsi             REAL,
                t5                 REAL, t10 REAL, t20 REAL, t30 REAL,
                agility            REAL,
                dribbling          REAL,
                vo2max             REAL,
                created_at         TEXT DEFAULT (datetime('now')),
                UNIQUE(player_id, session_id)
            );
        """) if not self.use_postgres else self._init_postgres(cur)

        conn.commit()
        conn.close()
        self._seed_if_empty()

    def _init_postgres(self, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           SERIAL PRIMARY KEY,
                label        TEXT   NOT NULL UNIQUE,
                session_date DATE,
                created_at   TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS players (
                id         SERIAL PRIMARY KEY,
                name       TEXT   NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS measurements (
                id                 SERIAL PRIMARY KEY,
                player_id          INTEGER NOT NULL REFERENCES players(id),
                session_id         INTEGER NOT NULL REFERENCES sessions(id),
                cmj_1 REAL, cmj_2 REAL, cmj_3 REAL,
                dvj_kontaktzeit REAL, dvj_hoehe REAL,
                sprint_t5_r1 REAL, sprint_t10_r1 REAL,
                sprint_t20_r1 REAL, sprint_t30_r1 REAL,
                sprint_t5_r2 REAL, sprint_t10_r2 REAL,
                sprint_t20_r2 REAL, sprint_t30_r2 REAL,
                agility_r1 REAL, agility_r2 REAL,
                dribbling_r1 REAL, dribbling_r2 REAL,
                yoyo_level REAL, yoyo_shuttles REAL,
                hf_max REAL,
                cmj_best REAL, dj_rsi REAL,
                t5 REAL, t10 REAL, t20 REAL, t30 REAL,
                agility REAL, dribbling REAL, vo2max REAL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, session_id)
            );
        """)

    # ── Seed ──────────────────────────────────────────────

    def _seed_if_empty(self):
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        count = cur.fetchone()[0]
        conn.close()
        if count == 0:
            for rec in SEED_RECORDS:
                self.upsert_record(rec)

    # ── Write ──────────────────────────────────────────────

    def upsert_record(self, rec: dict):
        """
        Insert or update a single player-session measurement.
        rec must have: name, session, date + all measurement fields.
        Best values are computed here so they're always consistent.
        """
        # Compute best values
        cmj_1    = rec.get("cmj_1") or 0
        cmj_2    = rec.get("cmj_2") or 0
        cmj_3    = rec.get("cmj_3") or 0
        cmj_best = max(cmj_1, cmj_2, cmj_3) or None

        kz = rec.get("dvj_kontaktzeit")
        hh = rec.get("dvj_hoehe")
        dj_rsi = round(hh / kz * 1000, 3) if kz and hh and kz > 0 else None

        def best_time(*vals):
            v = [x for x in vals if x is not None and x > 0]
            return min(v) if v else None

        t5  = best_time(rec.get("sprint_t5_r1"),  rec.get("sprint_t5_r2"))
        t10 = best_time(rec.get("sprint_t10_r1"), rec.get("sprint_t10_r2"))
        t20 = best_time(rec.get("sprint_t20_r1"), rec.get("sprint_t20_r2"))
        t30 = best_time(rec.get("sprint_t30_r1"), rec.get("sprint_t30_r2"))
        agility   = best_time(rec.get("agility_r1"),   rec.get("agility_r2"))
        dribbling = best_time(rec.get("dribbling_r1"), rec.get("dribbling_r2"))

        lv = rec.get("yoyo_level")
        sh = rec.get("yoyo_shuttles")
        vo2max = round(lv * 4.225 + sh * 0.682, 2) if lv is not None and sh is not None else None

        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()

        # Upsert session
        cur.execute(
            f"INSERT INTO sessions (label, session_date) VALUES ({ph},{ph}) "
            f"ON CONFLICT(label) DO UPDATE SET session_date=excluded.session_date",
            (rec["session"], rec.get("date"))
        ) if self.use_postgres else cur.execute(
            "INSERT OR IGNORE INTO sessions (label, session_date) VALUES (?,?)",
            (rec["session"], rec.get("date"))
        )

        # Upsert player
        cur.execute(
            f"INSERT INTO players (name) VALUES ({ph}) ON CONFLICT(name) DO NOTHING",
            (rec["name"],)
        ) if self.use_postgres else cur.execute(
            "INSERT OR IGNORE INTO players (name) VALUES (?)", (rec["name"],)
        )

        cur.execute(f"SELECT id FROM sessions WHERE label={ph}", (rec["session"],))
        sess_id   = cur.fetchone()[0]
        cur.execute(f"SELECT id FROM players  WHERE name={ph}",  (rec["name"],))
        player_id = cur.fetchone()[0]

        vals = (
            player_id, sess_id,
            rec.get("cmj_1"), rec.get("cmj_2"), rec.get("cmj_3"),
            rec.get("dvj_kontaktzeit"), rec.get("dvj_hoehe"),
            rec.get("sprint_t5_r1"), rec.get("sprint_t10_r1"),
            rec.get("sprint_t20_r1"), rec.get("sprint_t30_r1"),
            rec.get("sprint_t5_r2"), rec.get("sprint_t10_r2"),
            rec.get("sprint_t20_r2"), rec.get("sprint_t30_r2"),
            rec.get("agility_r1"), rec.get("agility_r2"),
            rec.get("dribbling_r1"), rec.get("dribbling_r2"),
            rec.get("yoyo_level"), rec.get("yoyo_shuttles"),
            rec.get("hf_max"),
            cmj_best, dj_rsi, t5, t10, t20, t30, agility, dribbling, vo2max,
        )

        if self.use_postgres:
            cur.execute("""
                INSERT INTO measurements
                  (player_id,session_id,cmj_1,cmj_2,cmj_3,dvj_kontaktzeit,dvj_hoehe,
                   sprint_t5_r1,sprint_t10_r1,sprint_t20_r1,sprint_t30_r1,
                   sprint_t5_r2,sprint_t10_r2,sprint_t20_r2,sprint_t30_r2,
                   agility_r1,agility_r2,dribbling_r1,dribbling_r2,
                   yoyo_level,yoyo_shuttles,hf_max,
                   cmj_best,dj_rsi,t5,t10,t20,t30,agility,dribbling,vo2max)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(player_id,session_id) DO UPDATE SET
                  cmj_1=excluded.cmj_1, cmj_2=excluded.cmj_2, cmj_3=excluded.cmj_3,
                  dvj_kontaktzeit=excluded.dvj_kontaktzeit, dvj_hoehe=excluded.dvj_hoehe,
                  sprint_t5_r1=excluded.sprint_t5_r1, sprint_t10_r1=excluded.sprint_t10_r1,
                  sprint_t20_r1=excluded.sprint_t20_r1, sprint_t30_r1=excluded.sprint_t30_r1,
                  sprint_t5_r2=excluded.sprint_t5_r2, sprint_t10_r2=excluded.sprint_t10_r2,
                  sprint_t20_r2=excluded.sprint_t20_r2, sprint_t30_r2=excluded.sprint_t30_r2,
                  agility_r1=excluded.agility_r1, agility_r2=excluded.agility_r2,
                  dribbling_r1=excluded.dribbling_r1, dribbling_r2=excluded.dribbling_r2,
                  yoyo_level=excluded.yoyo_level, yoyo_shuttles=excluded.yoyo_shuttles,
                  hf_max=excluded.hf_max, cmj_best=excluded.cmj_best,
                  dj_rsi=excluded.dj_rsi, t5=excluded.t5, t10=excluded.t10,
                  t20=excluded.t20, t30=excluded.t30, agility=excluded.agility,
                  dribbling=excluded.dribbling, vo2max=excluded.vo2max
            """, vals)
        else:
            cur.execute("""
                INSERT INTO measurements
                  (player_id,session_id,cmj_1,cmj_2,cmj_3,dvj_kontaktzeit,dvj_hoehe,
                   sprint_t5_r1,sprint_t10_r1,sprint_t20_r1,sprint_t30_r1,
                   sprint_t5_r2,sprint_t10_r2,sprint_t20_r2,sprint_t30_r2,
                   agility_r1,agility_r2,dribbling_r1,dribbling_r2,
                   yoyo_level,yoyo_shuttles,hf_max,
                   cmj_best,dj_rsi,t5,t10,t20,t30,agility,dribbling,vo2max)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(player_id,session_id) DO UPDATE SET
                  cmj_1=excluded.cmj_1, cmj_2=excluded.cmj_2, cmj_3=excluded.cmj_3,
                  dvj_kontaktzeit=excluded.dvj_kontaktzeit, dvj_hoehe=excluded.dvj_hoehe,
                  sprint_t5_r1=excluded.sprint_t5_r1, sprint_t10_r1=excluded.sprint_t10_r1,
                  sprint_t20_r1=excluded.sprint_t20_r1, sprint_t30_r1=excluded.sprint_t30_r1,
                  sprint_t5_r2=excluded.sprint_t5_r2, sprint_t10_r2=excluded.sprint_t10_r2,
                  sprint_t20_r2=excluded.sprint_t20_r2, sprint_t30_r2=excluded.sprint_t30_r2,
                  agility_r1=excluded.agility_r1, agility_r2=excluded.agility_r2,
                  dribbling_r1=excluded.dribbling_r1, dribbling_r2=excluded.dribbling_r2,
                  yoyo_level=excluded.yoyo_level, yoyo_shuttles=excluded.yoyo_shuttles,
                  hf_max=excluded.hf_max, cmj_best=excluded.cmj_best,
                  dj_rsi=excluded.dj_rsi, t5=excluded.t5, t10=excluded.t10,
                  t20=excluded.t20, t30=excluded.t30, agility=excluded.agility,
                  dribbling=excluded.dribbling, vo2max=excluded.vo2max
            """, vals)

        conn.commit()
        conn.close()

    def delete_session(self, label: str):
        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()
        cur.execute(f"SELECT id FROM sessions WHERE label={ph}", (label,))
        row = cur.fetchone()
        if row:
            cur.execute(f"DELETE FROM measurements WHERE session_id={ph}", (row[0],))
            cur.execute(f"DELETE FROM sessions WHERE id={ph}", (row[0],))
        conn.commit()
        conn.close()

    def upsert_session_from_df(self, df: pd.DataFrame, label: str,
                                session_date: str = None):
        """
        Bulk-save a parsed DataFrame (from Excel upload) into the database.
        df must have columns: name + all metric columns.
        """
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["session"] = label
            rec["date"]    = session_date or label
            # Map app column names → db column names
            mapping = {
                "cmj": "cmj_best", "dj_rsi": "dj_rsi",
                "t5": "t5", "t10": "t10", "t20": "t20", "t30": "t30",
                "agility": "agility", "dribbling": "dribbling", "vo2max": "vo2max",
            }
            # Fill missing raw attempts with None
            for col in ["cmj_1","cmj_2","cmj_3","dvj_kontaktzeit","dvj_hoehe",
                        "sprint_t5_r1","sprint_t10_r1","sprint_t20_r1","sprint_t30_r1",
                        "sprint_t5_r2","sprint_t10_r2","sprint_t20_r2","sprint_t30_r2",
                        "agility_r1","agility_r2","dribbling_r1","dribbling_r2",
                        "yoyo_level","yoyo_shuttles","hf_max"]:
                if col not in rec:
                    rec[col] = None
            self.upsert_record(rec)

    # ── Read ──────────────────────────────────────────────

    def load_dataframe(self) -> pd.DataFrame:
        """
        Load all measurements joined with player names and session labels.
        Returns a DataFrame in the same format the app expects.
        """
        conn = self._connect()
        query = """
            SELECT
                p.name        AS name,
                s.label       AS session,
                m.cmj_1, m.cmj_2, m.cmj_3,
                m.dvj_kontaktzeit, m.dvj_hoehe,
                m.sprint_t5_r1,  m.sprint_t10_r1,
                m.sprint_t20_r1, m.sprint_t30_r1,
                m.sprint_t5_r2,  m.sprint_t10_r2,
                m.sprint_t20_r2, m.sprint_t30_r2,
                m.agility_r1, m.agility_r2,
                m.dribbling_r1, m.dribbling_r2,
                m.yoyo_level, m.yoyo_shuttles, m.hf_max,
                m.cmj_best  AS cmj,
                m.dj_rsi,
                m.t5, m.t10, m.t20, m.t30,
                m.agility, m.dribbling, m.vo2max
            FROM measurements m
            JOIN players  p ON p.id = m.player_id
            JOIN sessions s ON s.id = m.session_id
            ORDER BY s.session_date, p.name
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_sessions(self) -> list:
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("SELECT label FROM sessions ORDER BY session_date, label")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    def get_players(self) -> list:
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("SELECT name FROM players ORDER BY name")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    def get_session_stats(self) -> pd.DataFrame:
        """Returns player count per session for the sidebar."""
        conn = self._connect()
        query = """
            SELECT s.label, COUNT(*) as n_players
            FROM measurements m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.cmj_best IS NOT NULL
            GROUP BY s.label
            ORDER BY s.session_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
