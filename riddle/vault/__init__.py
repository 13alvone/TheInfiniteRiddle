#!/usr/bin/env python3
"""SQLite Vault helpers for The Infinite Riddle."""
import sqlite3
from pathlib import Path
from typing import Optional

from ..io import sha256_of_file


def ensure_vault(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
          id INTEGER PRIMARY KEY,
          started_utc TEXT NOT NULL,
          theme TEXT NOT NULL,
          seed_commitment TEXT NOT NULL UNIQUE,
          duration_sec INTEGER NOT NULL,
          form_path TEXT NOT NULL,
          bpm_base REAL NOT NULL,
          lufs_target REAL,
          coherence REAL,
          presence REAL,
          hostility REAL,
          obliquity REAL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_harmony (
          run_id INTEGER REFERENCES runs(id),
          section_idx INTEGER,
          key_root TEXT,
          mode TEXT,
          start_sec REAL,
          end_sec REAL,
          PRIMARY KEY (run_id, section_idx)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_rhythm (
          run_id INTEGER REFERENCES runs(id),
          section_idx INTEGER,
          time_sig TEXT,
          euclid TEXT,
          ca_rule INTEGER,
          swing REAL,
          PRIMARY KEY (run_id, section_idx)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
          id INTEGER PRIMARY KEY,
          run_id INTEGER REFERENCES runs(id),
          kind TEXT CHECK(kind IN ('wav','midi','mp4','json','stem','mythic')),
          path TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          duration_sec REAL,
          bpm_est REAL,
          key_hint TEXT,
          mythic_type TEXT
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_theme ON runs(theme);")
        conn.commit()
    finally:
        conn.close()


def vault_insert_run(db_path: Path, run) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO runs(started_utc, theme, seed_commitment, duration_sec, form_path, bpm_base, lufs_target, coherence, presence, hostility, obliquity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run["started_utc"], run["theme"], run["seed_commitment"], run["duration_sec"],
            run["form_path"], run["bpm_base"], run["lufs_target"], run["coherence"],
            run["presence"], run["hostility"], run["obliquity"]
        ))
        run_id = cur.lastrowid
        for idx, sec in enumerate(run["sections"]):
            cur.execute("""
            INSERT INTO run_harmony(run_id, section_idx, key_root, mode, start_sec, end_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, idx, sec["key_root"], sec["mode"], sec["start_sec"], sec["end_sec"]))
            cur.execute("""
            INSERT INTO run_rhythm(run_id, section_idx, time_sig, euclid, ca_rule, swing)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, idx, sec["time_sig"], sec["euclid"], sec["ca_rule"], sec["swing"]))
        conn.commit()
        return run_id
    finally:
        conn.close()


def vault_insert_artifact(db_path: Path, run_id: int, kind: str, path: Path, duration_sec: float,
                          bpm_est: Optional[float], key_hint: Optional[str], mythic_type: Optional[str]) -> None:
    sha256 = sha256_of_file(path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO artifacts(run_id, kind, path, sha256, duration_sec, bpm_est, key_hint, mythic_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, kind, str(path), sha256, duration_sec, bpm_est, key_hint, mythic_type))
        conn.commit()
    finally:
        conn.close()
