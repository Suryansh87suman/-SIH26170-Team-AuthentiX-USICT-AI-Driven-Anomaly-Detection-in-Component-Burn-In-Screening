from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import pandas as pd


def _connect(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str | Path):
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS batch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at TEXT NOT NULL,
                batch_name TEXT NOT NULL,
                source_name TEXT,
                parameter TEXT,
                component_count INTEGER,
                normal_count INTEGER,
                warning_count INTEGER,
                critical_count INTEGER,
                early_reject_count INTEGER,
                mean_risk REAL,
                max_risk REAL
            );
            CREATE TABLE IF NOT EXISTS engineer_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at TEXT NOT NULL,
                component_id TEXT NOT NULL,
                lot_id TEXT,
                parameter TEXT NOT NULL,
                model_risk REAL,
                model_level TEXT,
                model_early_reject INTEGER,
                engineer_label TEXT NOT NULL,
                note TEXT
            );
            """
        )


def save_batch_snapshot(db_path: str | Path, batch_name: str, source_name: str, parameter: str, scored: pd.DataFrame):
    init_db(db_path)
    values = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        batch_name,
        source_name,
        parameter,
        int(len(scored)),
        int((scored["risk_level"] == "Normal").sum()),
        int((scored["risk_level"] == "Warning").sum()),
        int((scored["risk_level"] == "Critical").sum()),
        int(scored["early_reject"].sum()),
        float(scored["risk_score"].mean()),
        float(scored["risk_score"].max()),
    )
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO batch_history
            (saved_at,batch_name,source_name,parameter,component_count,normal_count,warning_count,critical_count,early_reject_count,mean_risk,max_risk)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            values,
        )


def save_feedback(db_path: str | Path, row: pd.Series, parameter: str, engineer_label: str, note: str = ""):
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO engineer_feedback
            (saved_at,component_id,lot_id,parameter,model_risk,model_level,model_early_reject,engineer_label,note)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                str(row.get("component_id", "")),
                str(row.get("lot_id", "")),
                parameter,
                float(row.get("risk_score", 0)),
                str(row.get("risk_level", "")),
                int(bool(row.get("early_reject", False))),
                engineer_label,
                note,
            ),
        )


def load_history(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM batch_history ORDER BY id DESC", conn)


def load_feedback(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM engineer_feedback ORDER BY id DESC", conn)
