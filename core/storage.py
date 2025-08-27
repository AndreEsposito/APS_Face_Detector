from __future__ import annotations
import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from .paths import SQLITE_PATH, ACCESS_LOG_CSV
from .paths import SAMPLES_DIR

def db_init():
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT UNIQUE NOT NULL,
            nivel INTEGER NOT NULL,
            criado_em TEXT NOT NULL,
            samples INTEGER NOT NULL DEFAULT 0
        )
        """)
        conn.commit()

def db_upsert_user(nome: str, nivel: int) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(SQLITE_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, nivel FROM users WHERE nome = ?", (nome,))
        row = c.fetchone()
        if row:
            uid, nivel_old = row
            if nivel_old != nivel:
                c.execute("UPDATE users SET nivel = ? WHERE id = ?", (nivel, uid))
                conn.commit()
            return uid
        c.execute("INSERT INTO users (nome, nivel, criado_em, samples) VALUES (?, ?, ?, 0)",
                  (nome, nivel, now))
        conn.commit()
        return c.lastrowid

def db_get_user_by_id(user_id: int) -> Optional[Tuple[int, str, int, str, int]]:
    with sqlite3.connect(SQLITE_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, nome, nivel, criado_em, samples FROM users WHERE id = ?", (user_id,))
        return c.fetchone()

def db_inc_samples(user_id: int, add: int = 1):
    with sqlite3.connect(SQLITE_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET samples = samples + ? WHERE id = ?", (add, user_id))
        conn.commit()

def log_access(user_pred_id: Optional[int], dist: Optional[float],
               liveness_ok: bool, nivel_concedido: int, obs: str = ""):
    Path(ACCESS_LOG_CSV).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(ACCESS_LOG_CSV).exists()
    with open(ACCESS_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "user_pred_id", "user_pred_nome", "distancia",
                        "liveness_ok", "nivel_concedido", "obs"])
        nome = None
        if user_pred_id is not None:
            row = db_get_user_by_id(user_pred_id)
            nome = row[1] if row else None
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    user_pred_id, nome, f"{dist:.3f}" if dist is not None else "",
                    int(liveness_ok), nivel_concedido, obs])
