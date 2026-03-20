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
SEED_RECORDS = []  # Empty — all data comes from Excel uploads

class BVBDatabase:
    """
    Persistent database for BVB Frauen performance data.
    Auto-detects SQLite (local) or PostgreSQL (Supabase cloud).
    """

    def __init__(self):
        # Check both environment variable and Streamlit secrets
        self.db_url = os.environ.get("DATABASE_URL", "").strip()

        # Also check st.secrets if running in Streamlit
        if not self.db_url:
            try:
                import streamlit as st
                self.db_url = str(st.secrets.get("DATABASE_URL", "")).strip()
            except Exception:
                pass

        # Clean up the URL
        if self.db_url:
            # psycopg2 needs "postgresql://" not "postgres://"
            if self.db_url.startswith("postgres://"):
                self.db_url = "postgresql://" + self.db_url[len("postgres://"):]

            # Add sslmode=require for Supabase if not already present
            if "supabase" in self.db_url and "sslmode" not in self.db_url:
                sep = "&" if "?" in self.db_url else "?"
                self.db_url += f"{sep}sslmode=require"

        self.use_postgres = bool(self.db_url) and             self.db_url.startswith("postgresql")

        if self.use_postgres:
            self.db_path = None
        else:
            # SQLite — store next to database.py
            self.db_path = Path(__file__).parent / "bvb_data.db"

    def _connect(self):
        if self.use_postgres:
            try:
                import psycopg2
                return psycopg2.connect(
                    self.db_url,
                    connect_timeout=10,
                )
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
        """Create tables if they don't exist. Tables are committed before returning."""
        conn = self._connect()
        cur  = conn.cursor()

        if self.use_postgres:
            self._init_postgres(cur)
            conn.commit()  # commit immediately so tables exist for subsequent queries
        else:
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
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(player_id, session_id)
                );
            """)

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
        """Database starts empty — populated by Excel uploads only."""
        self.clean_orphan_players()

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

        # Upsert session — date must be a real ISO date (YYYY-MM-DD) or None
        # Never pass the label string as the date
        raw_date = rec.get("date")
        if raw_date and raw_date != rec["session"]:
            session_date = raw_date  # only use it if it looks like a real date
        else:
            session_date = None  # PostgreSQL DATE column requires real date or NULL

        if self.use_postgres:
            cur.execute(
                f"INSERT INTO sessions (label, session_date) VALUES ({ph},{ph}) "
                f"ON CONFLICT(label) DO UPDATE SET session_date=COALESCE(excluded.session_date, sessions.session_date)",
                (rec["session"], session_date)
            )
        else:
            cur.execute(
                "INSERT OR IGNORE INTO sessions (label, session_date) VALUES (?,?)",
                (rec["session"], session_date)
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
        session_date must be a real ISO date string (YYYY-MM-DD) or None.
        """
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["session"] = label
            rec["date"]    = session_date  # None is fine — stored as NULL
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
        Returns empty DataFrame if tables don't exist yet.
        """
        conn = self._connect()
        try:
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
        except Exception:
            # Tables don't exist yet — return empty DataFrame with correct columns
            conn.close()
            return pd.DataFrame(columns=[
                "name", "session", "cmj_1", "cmj_2", "cmj_3",
                "dvj_kontaktzeit", "dvj_hoehe",
                "sprint_t5_r1", "sprint_t10_r1", "sprint_t20_r1", "sprint_t30_r1",
                "sprint_t5_r2", "sprint_t10_r2", "sprint_t20_r2", "sprint_t30_r2",
                "agility_r1", "agility_r2", "dribbling_r1", "dribbling_r2",
                "yoyo_level", "yoyo_shuttles", "hf_max",
                "cmj", "dj_rsi", "t5", "t10", "t20", "t30",
                "agility", "dribbling", "vo2max"
            ])

    def get_sessions(self) -> list:
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("SELECT label FROM sessions ORDER BY session_date, label")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    def get_players(self) -> list:
        """Only players who have at least one measurement — no ghost records."""
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("""
            SELECT DISTINCT p.name
            FROM players p
            JOIN measurements m ON m.player_id = p.id
            ORDER BY p.name
        """)
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

    # ── Maintenance ───────────────────────────────────────────────────

    def clean_orphan_players(self) -> int:
        """
        Remove players who have no measurements at all.
        Happens when a failed seed attempt creates player records
        without the matching measurement records.
        Returns number of deleted rows.
        """
        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()
        cur.execute("""
            DELETE FROM players
            WHERE id NOT IN (
                SELECT DISTINCT player_id FROM measurements
            )
        """)
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted

    def reset_and_reseed(self):
        """
        Drop all data and reseed from scratch.
        Use this to fix a corrupted/duplicate Supabase state.
        """
        conn = self._connect()
        cur  = conn.cursor()
        # Delete in dependency order
        cur.execute("DELETE FROM measurements")
        cur.execute("DELETE FROM sessions")
        cur.execute("DELETE FROM players")
        # Reset auto-increment sequences (PostgreSQL)
        if self.use_postgres:
            cur.execute("ALTER SEQUENCE measurements_id_seq RESTART WITH 1")
            cur.execute("ALTER SEQUENCE sessions_id_seq RESTART WITH 1")
            cur.execute("ALTER SEQUENCE players_id_seq RESTART WITH 1")
        conn.commit()
        conn.close()
        self._seed_if_empty()

    def rename_player(self, old_name: str, new_name: str):
        """Rename a player across all measurements."""
        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()
        cur.execute(f"UPDATE players SET name={ph} WHERE name={ph}",
                    (new_name, old_name))
        conn.commit()
        conn.close()

    def delete_player(self, name: str):
        """
        Remove a player and all their measurements from the database.
        Use when a player has permanently left the squad.
        """
        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()
        cur.execute(f"SELECT id FROM players WHERE name={ph}", (name,))
        row = cur.fetchone()
        if row:
            cur.execute(f"DELETE FROM measurements WHERE player_id={ph}", (row[0],))
            cur.execute(f"DELETE FROM players WHERE id={ph}", (row[0],))
        conn.commit()
        conn.close()

    def get_player_sessions(self, name: str) -> list:
        """Return list of sessions a player has appeared in."""
        conn = self._connect()
        cur  = conn.cursor()
        ph   = self._ph()
        cur.execute(f"""
            SELECT s.label FROM measurements m
            JOIN sessions s ON s.id = m.session_id
            JOIN players p ON p.id = m.player_id
            WHERE p.name = {ph}
            ORDER BY s.session_date
        """, (name,))
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    def get_all_players_ever(self) -> list:
        """All players who have ever had a measurement — the real squad list."""
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("""
            SELECT DISTINCT p.name
            FROM players p
            JOIN measurements m ON m.player_id = p.id
            ORDER BY p.name
        """)
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
