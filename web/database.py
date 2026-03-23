"""
SQLite database for LLMBench quiz system.

Stores quiz sessions, individual answers, user accounts, and result summaries.
Uses WAL mode for safe concurrent access from multiple gunicorn workers.

Database location: results/llmbench.db (persisted via Docker volume mount).
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

from flask import g

logger = logging.getLogger("llmbench.db")

DB_PATH = Path(os.environ.get(
    "LLMBENCH_DB_PATH",
    str(Path(__file__).parent.parent / "results" / "llmbench.db"),
))

SCHEMA = """
-- Users: optional registration. Anonymous users get a row too.
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE,
    display_name    TEXT NOT NULL,
    email           TEXT,
    password_hash   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT
);

-- Quiz sessions: replaces the in-memory dict
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id),
    dataset         TEXT NOT NULL,
    max_questions   INTEGER NOT NULL,
    current_index   INTEGER NOT NULL DEFAULT 0,
    question_ids    TEXT NOT NULL,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    is_complete     INTEGER NOT NULL DEFAULT 0
);

-- Individual question responses
CREATE TABLE IF NOT EXISTS answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    question_id     TEXT NOT NULL,
    question_index  INTEGER NOT NULL,
    dataset         TEXT NOT NULL,
    selected_answer INTEGER NOT NULL,
    correct_answer  INTEGER NOT NULL,
    is_correct      INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    hlcc_score      REAL NOT NULL,
    cbm_score       REAL NOT NULL,
    answered_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Completed session summaries
CREATE TABLE IF NOT EXISTS session_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    user_id         INTEGER REFERENCES users(id),
    dataset         TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    accuracy        REAL NOT NULL,
    mean_confidence REAL NOT NULL,
    mean_hlcc       REAL NOT NULL,
    mean_cbm        REAL NOT NULL,
    total_hlcc      REAL NOT NULL,
    total_cbm       REAL NOT NULL,
    ece             REAL NOT NULL,
    brier           REAL NOT NULL,
    calibration_gap REAL NOT NULL,
    finished_at     TEXT NOT NULL,
    share_token     TEXT UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_answers_session ON answers(session_id);
CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
CREATE INDEX IF NOT EXISTS idx_answers_user_question
    ON answers(session_id, question_id);
CREATE INDEX IF NOT EXISTS idx_session_results_user ON session_results(user_id);
CREATE INDEX IF NOT EXISTS idx_session_results_dataset ON session_results(dataset);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- AI model benchmark runs
CREATE TABLE IF NOT EXISTS model_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,
    model_name      TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    accuracy        REAL NOT NULL,
    mean_confidence REAL NOT NULL,
    mean_hlcc       REAL NOT NULL,
    mean_cbm        REAL NOT NULL,
    total_hlcc      REAL NOT NULL,
    total_cbm       REAL NOT NULL,
    ece             REAL NOT NULL DEFAULT 0,
    brier           REAL NOT NULL DEFAULT 0,
    calibration_gap REAL NOT NULL DEFAULT 0,
    method          TEXT,
    temperature     REAL,
    run_timestamp   TEXT NOT NULL,
    uploaded_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-question AI model answers
CREATE TABLE IF NOT EXISTS model_answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES model_runs(run_id),
    question_id     TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    selected_answer INTEGER NOT NULL,
    correct_answer  INTEGER NOT NULL,
    is_correct      INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    hlcc_score      REAL NOT NULL,
    cbm_score       REAL NOT NULL,
    processing_time REAL,
    temperature     REAL
);

CREATE INDEX IF NOT EXISTS idx_model_runs_model ON model_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_runs_dataset ON model_runs(dataset);
CREATE INDEX IF NOT EXISTS idx_model_answers_run ON model_answers(run_id);
CREATE INDEX IF NOT EXISTS idx_model_answers_question ON model_answers(question_id);
CREATE INDEX IF NOT EXISTS idx_model_answers_model ON model_answers(model_name);
CREATE INDEX IF NOT EXISTS idx_model_answers_model_question ON model_answers(model_name, question_id);
"""


def get_db():
    """Get a database connection for the current request (stored in flask.g)."""
    if "db" not in g:
        g.db = _connect()
    return g.db


def close_db(e=None):
    """Close the database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables if they don't exist. Handles concurrent worker startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            conn = _connect()
            conn.executescript(SCHEMA)
            # Migration: add email column if missing
            cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            if "email" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
            conn.commit()
            logger.info("Database initialized at %s", DB_PATH)
            conn.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                import time
                time.sleep(0.5)
                continue
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass


def init_app(app):
    """Register database teardown with a Flask app."""
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
        migrate_json_results()


def _connect():
    """Open a new SQLite connection with WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ============================================================
# SESSION OPERATIONS
# ============================================================

def create_session(session_id, user_id, dataset, max_questions, question_ids):
    """Create a new quiz session."""
    db = get_db()
    db.execute(
        "INSERT INTO sessions (id, user_id, dataset, max_questions, question_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, user_id, dataset, max_questions, json.dumps(question_ids)),
    )
    db.commit()


def get_session(session_id):
    """Get a session by ID. Returns dict or None."""
    db = get_db()
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["question_ids"] = json.loads(result["question_ids"])
    return result


def advance_session(session_id):
    """Increment current_index by 1."""
    db = get_db()
    db.execute(
        "UPDATE sessions SET current_index = current_index + 1 WHERE id = ?",
        (session_id,),
    )
    db.commit()


def complete_session(session_id):
    """Mark session as complete."""
    db = get_db()
    db.execute(
        "UPDATE sessions SET is_complete = 1, finished_at = datetime('now') WHERE id = ?",
        (session_id,),
    )
    db.commit()


# ============================================================
# ANSWER OPERATIONS
# ============================================================

def save_answer(session_id, question_id, question_index, dataset,
                selected_answer, correct_answer, is_correct,
                confidence, hlcc_score, cbm_score):
    """Save a single answer."""
    db = get_db()
    db.execute(
        "INSERT INTO answers "
        "(session_id, question_id, question_index, dataset, selected_answer, "
        "correct_answer, is_correct, confidence, hlcc_score, cbm_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, question_id, question_index, dataset,
         selected_answer, correct_answer, int(is_correct),
         confidence, hlcc_score, cbm_score),
    )
    db.commit()


def get_session_answers(session_id):
    """Get all answers for a session, ordered by question_index."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM answers WHERE session_id = ? ORDER BY question_index",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_running_stats(session_id):
    """Get running accuracy and HLCC for a session."""
    db = get_db()
    row = db.execute(
        "SELECT COUNT(*) as n, "
        "  SUM(is_correct) as total_correct, "
        "  AVG(CAST(is_correct AS REAL)) as accuracy, "
        "  AVG(hlcc_score) as mean_hlcc, "
        "  AVG(cbm_score) as mean_cbm "
        "FROM answers WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else {}


# ============================================================
# USER OPERATIONS
# ============================================================

def create_user(username, display_name, password_hash=None, email=None):
    """Create a user account. Returns user_id."""
    db = get_db()
    cur = db.execute(
        "INSERT INTO users (username, display_name, password_hash, email) VALUES (?, ?, ?, ?)",
        (username, display_name, password_hash, email),
    )
    db.commit()
    return cur.lastrowid


def get_user_by_username(username):
    """Find user by username. Returns dict or None."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Find user by ID. Returns dict or None."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


def create_anonymous_user(display_name):
    """Create an anonymous user (no username/password). Returns user_id."""
    db = get_db()
    cur = db.execute(
        "INSERT INTO users (username, display_name) VALUES (NULL, ?)",
        (display_name,),
    )
    db.commit()
    return cur.lastrowid


def update_last_seen(user_id):
    """Update the last_seen_at timestamp."""
    db = get_db()
    db.execute(
        "UPDATE users SET last_seen_at = datetime('now') WHERE id = ?",
        (user_id,),
    )
    db.commit()


# ============================================================
# RESULTS OPERATIONS
# ============================================================

def save_session_result(session_id, user_id, dataset, total_questions,
                        accuracy, mean_confidence, mean_hlcc, mean_cbm,
                        total_hlcc, total_cbm, ece, brier,
                        calibration_gap, share_token):
    """Save completed session summary."""
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO session_results "
        "(session_id, user_id, dataset, total_questions, accuracy, "
        "mean_confidence, mean_hlcc, mean_cbm, total_hlcc, total_cbm, "
        "ece, brier, calibration_gap, finished_at, share_token) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)",
        (session_id, user_id, dataset, total_questions,
         accuracy, mean_confidence, mean_hlcc, mean_cbm,
         total_hlcc, total_cbm, ece, brier, calibration_gap, share_token),
    )
    db.commit()


def get_result_by_share_token(share_token):
    """Look up shared results. Returns dict or None."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM session_results WHERE share_token = ?", (share_token,)
    ).fetchone()
    return dict(row) if row else None


def get_human_leaderboard(dataset=None):
    """Get human leaderboard entries, optionally filtered by dataset."""
    db = get_db()
    if dataset:
        rows = db.execute(
            "SELECT sr.*, u.display_name FROM session_results sr "
            "LEFT JOIN users u ON sr.user_id = u.id "
            "WHERE sr.dataset = ? ORDER BY sr.mean_hlcc DESC",
            (dataset,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT sr.*, u.display_name FROM session_results sr "
            "LEFT JOIN users u ON sr.user_id = u.id "
            "ORDER BY sr.mean_hlcc DESC",
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_history(user_id):
    """Get all completed sessions for a user."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM session_results WHERE user_id = ? ORDER BY finished_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_user_dataset_progress(user_id):
    """Get per-dataset stats for a user: sessions completed, questions answered."""
    db = get_db()
    rows = db.execute(
        "SELECT dataset, COUNT(*) as sessions, "
        "  SUM(total_questions) as total_answered, "
        "  AVG(accuracy) as avg_accuracy, "
        "  AVG(mean_hlcc) as avg_hlcc, "
        "  AVG(ece) as avg_ece "
        "FROM session_results WHERE user_id = ? GROUP BY dataset",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# MODEL BENCHMARK OPERATIONS
# ============================================================

def save_model_run(run_id, model_name, dataset, total_questions,
                   accuracy, mean_confidence, mean_hlcc, mean_cbm,
                   total_hlcc, total_cbm, ece, brier, calibration_gap,
                   method, temperature, run_timestamp):
    """Save an AI model benchmark run summary."""
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO model_runs "
        "(run_id, model_name, dataset, total_questions, accuracy, "
        "mean_confidence, mean_hlcc, mean_cbm, total_hlcc, total_cbm, "
        "ece, brier, calibration_gap, method, temperature, run_timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, model_name, dataset, total_questions,
         accuracy, mean_confidence, mean_hlcc, mean_cbm,
         total_hlcc, total_cbm, ece, brier, calibration_gap,
         method, temperature, run_timestamp),
    )
    db.commit()


def save_model_answer(run_id, question_id, dataset, model_name,
                      selected_answer, correct_answer, is_correct,
                      confidence, hlcc_score, cbm_score,
                      processing_time=None, temperature=None):
    """Save a single AI model answer."""
    db = get_db()
    db.execute(
        "INSERT INTO model_answers "
        "(run_id, question_id, dataset, model_name, selected_answer, "
        "correct_answer, is_correct, confidence, hlcc_score, cbm_score, "
        "processing_time, temperature) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, question_id, dataset, model_name,
         selected_answer, correct_answer, int(is_correct),
         confidence, hlcc_score, cbm_score,
         processing_time, temperature),
    )
    db.commit()


def save_model_answers_batch(answers):
    """Save multiple AI model answers in a single transaction."""
    db = get_db()
    db.executemany(
        "INSERT INTO model_answers "
        "(run_id, question_id, dataset, model_name, selected_answer, "
        "correct_answer, is_correct, confidence, hlcc_score, cbm_score, "
        "processing_time, temperature) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(a["run_id"], a["question_id"], a["dataset"], a["model_name"],
          a["selected_answer"], a["correct_answer"], int(a["is_correct"]),
          a["confidence"], a["hlcc_score"], a["cbm_score"],
          a.get("processing_time"), a.get("temperature"))
         for a in answers],
    )
    db.commit()


def get_model_runs(dataset=None, model_name=None):
    """Get model benchmark runs, optionally filtered."""
    db = get_db()
    query = "SELECT * FROM model_runs WHERE 1=1"
    params = []
    if dataset:
        query += " AND dataset = ?"
        params.append(dataset)
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    query += " ORDER BY uploaded_at DESC"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_model_answers_for_run(run_id):
    """Get all per-question answers for a model run."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM model_answers WHERE run_id = ? ORDER BY question_id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_question_model_stats(question_id):
    """Get per-model stats for a specific question."""
    db = get_db()
    rows = db.execute(
        "SELECT model_name, "
        "  COUNT(*) as times_answered, "
        "  AVG(CAST(is_correct AS REAL)) as accuracy, "
        "  AVG(confidence) as mean_confidence, "
        "  AVG(hlcc_score) as mean_hlcc "
        "FROM model_answers WHERE question_id = ? "
        "GROUP BY model_name ORDER BY mean_hlcc DESC",
        (question_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_hardest_questions(dataset=None, limit=50):
    """Get questions ranked by difficulty (lowest accuracy across all models)."""
    db = get_db()
    query = (
        "SELECT question_id, dataset, "
        "  COUNT(DISTINCT model_name) as models_tested, "
        "  COUNT(*) as total_attempts, "
        "  AVG(CAST(is_correct AS REAL)) as accuracy, "
        "  AVG(confidence) as mean_confidence, "
        "  AVG(hlcc_score) as mean_hlcc, "
        "  AVG(confidence) - AVG(CAST(is_correct AS REAL)) as overconfidence "
        "FROM model_answers"
    )
    params = []
    if dataset:
        query += " WHERE dataset = ?"
        params.append(dataset)
    query += " GROUP BY question_id, dataset ORDER BY accuracy ASC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_question_dynamics(dataset=None):
    """Get question-level analytics: which questions are hardest, most overconfident, etc."""
    db = get_db()
    # Combine human and model answers for full picture
    query = """
        SELECT question_id, dataset,
          COUNT(*) as total_attempts,
          SUM(CASE WHEN source = 'model' THEN 1 ELSE 0 END) as model_attempts,
          SUM(CASE WHEN source = 'human' THEN 1 ELSE 0 END) as human_attempts,
          AVG(CAST(is_correct AS REAL)) as overall_accuracy,
          AVG(CASE WHEN source = 'model' THEN CAST(is_correct AS REAL) END) as model_accuracy,
          AVG(CASE WHEN source = 'human' THEN CAST(is_correct AS REAL) END) as human_accuracy,
          AVG(confidence) as mean_confidence,
          AVG(hlcc_score) as mean_hlcc
        FROM (
          SELECT question_id, dataset, is_correct, confidence, hlcc_score, 'model' as source
          FROM model_answers
          UNION ALL
          SELECT question_id, dataset, is_correct, confidence, hlcc_score, 'human' as source
          FROM answers
        )
    """
    params = []
    if dataset:
        query += " WHERE dataset = ?"
        params.append(dataset)
    query += " GROUP BY question_id, dataset ORDER BY overall_accuracy ASC"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_question_stats(question_id):
    """Get aggregate stats for a specific question across all users."""
    db = get_db()
    row = db.execute(
        "SELECT question_id, COUNT(*) as times_answered, "
        "  AVG(CAST(is_correct AS REAL)) as accuracy, "
        "  AVG(confidence) as mean_confidence, "
        "  AVG(hlcc_score) as mean_hlcc "
        "FROM answers WHERE question_id = ? GROUP BY question_id",
        (question_id,),
    ).fetchone()
    return dict(row) if row else None


# ============================================================
# MIGRATION: import existing JSON results
# ============================================================

def migrate_json_results():
    """One-time migration of results/human/*.json into the database."""
    human_dir = DB_PATH.parent / "human"
    if not human_dir.exists():
        return

    db = _connect()
    try:
        count = db.execute("SELECT COUNT(*) FROM session_results").fetchone()[0]
        if count > 0:
            return  # already migrated

        json_files = list(human_dir.glob("*.json"))
        if not json_files:
            return

        logger.info("Migrating %d JSON result files...", len(json_files))
        for fp in json_files:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)

                participant = data.get("participant", "Anonymous")
                # Create anonymous user
                cur = db.execute(
                    "INSERT INTO users (display_name) VALUES (?)", (participant,)
                )
                user_id = cur.lastrowid
                session_id = data.get("session_id", fp.stem)
                dataset = data.get("dataset", "unknown")
                metrics = data.get("metrics", {})

                # Insert session
                db.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, user_id, dataset, max_questions, question_ids, is_complete) "
                    "VALUES (?, ?, ?, ?, '[]', 1)",
                    (session_id, user_id, dataset, data.get("total_questions", 0)),
                )

                # Insert individual answers if available
                for i, resp in enumerate(data.get("responses", [])):
                    db.execute(
                        "INSERT INTO answers "
                        "(session_id, question_id, question_index, dataset, "
                        "selected_answer, correct_answer, is_correct, "
                        "confidence, hlcc_score, cbm_score) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, resp.get("question_id", f"q_{i}"), i, dataset,
                         resp.get("selected_answer", 0),
                         resp.get("correct_answer", 0),
                         int(resp.get("is_correct", False)),
                         resp.get("confidence", 0.5),
                         resp.get("hlcc_score", 0),
                         resp.get("cbm_score", 0)),
                    )

                # Insert result summary
                import secrets
                db.execute(
                    "INSERT OR IGNORE INTO session_results "
                    "(session_id, user_id, dataset, total_questions, accuracy, "
                    "mean_confidence, mean_hlcc, mean_cbm, total_hlcc, total_cbm, "
                    "ece, brier, calibration_gap, finished_at, share_token) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, user_id, dataset,
                     data.get("total_questions", 0),
                     metrics.get("accuracy", 0),
                     metrics.get("mean_confidence", 0),
                     data.get("mean_hlcc", 0),
                     data.get("mean_cbm", 0),
                     data.get("total_hlcc", 0),
                     data.get("total_cbm", 0),
                     metrics.get("ece", 0),
                     metrics.get("brier", 0),
                     metrics.get("calibration_gap", 0),
                     data.get("finished", data.get("started", "")),
                     secrets.token_urlsafe(8)),
                )
            except Exception as e:
                logger.warning("Failed to migrate %s: %s", fp.name, e)

        db.commit()
        logger.info("Migration complete.")
    finally:
        db.close()
