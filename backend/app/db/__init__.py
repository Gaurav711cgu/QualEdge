"""
SQLite persistence layer with fail-fast NOWAIT locking and concurrent safety.
"""
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.app.db.session import get_connection

logger = logging.getLogger("Database-Persistence")

# Global Reentrant Lock for thread-level Mutex control (distributed lock manager pattern)
_db_lock = threading.RLock()

@contextmanager
def write_transaction():
    """
    Context manager for database write transactions.
    Employs fail-fast NOWAIT threading lock acquisition to prevent connection pool exhaustion.
    Uses BEGIN IMMEDIATE to prevent SQLite engine-level deadlock states.
    """
    acquired = _db_lock.acquire(blocking=False)
    if not acquired:
        logger.warning("Database write lock collision: Fail-fast NOWAIT lock rejected request.")
        raise sqlite3.OperationalError("Database is locked (fail-fast NOWAIT collision)")
        
    conn = get_connection()
    try:
        # Lock database file at SQLite engine level immediately for writing
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.commit()
    except sqlite3.OperationalError as oe:
        if "locked" in str(oe).lower():
            logger.warning(f"SQLite engine lock conflict detected: {oe}")
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Write transaction failed, rolled back: {e}")
        raise
    finally:
        conn.close()
        _db_lock.release()

def init_db() -> None:
    """Create tables if they don't exist."""
    with write_transaction() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                model_name TEXT NOT NULL,
                family TEXT,
                precision TEXT,
                metric_name TEXT,
                metric_value REAL,
                model_size_mb REAL,
                latency_ms REAL,
                device TEXT,
                runtime TEXT,
                accelerator TEXT,
                cpu_fallback_ops TEXT,
                verified_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aihub_jobs (
                id TEXT PRIMARY KEY,
                model_name TEXT,
                device TEXT,
                runtime TEXT,
                compile_job_id TEXT,
                profile_job_id TEXT,
                status TEXT,
                latency_ms REAL,
                cpu_fallback_ops TEXT,
                created_at TEXT
            )
        """)

def upsert_benchmark(record: Dict[str, Any]) -> None:
    """Insert or replace a benchmark record."""
    target = record.get("target", {})
    with write_transaction() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO benchmarks (
                id, source, model_name, family, precision, metric_name,
                metric_value, model_size_mb, latency_ms, device, runtime,
                accelerator, cpu_fallback_ops, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("id"),
            record.get("source", "demo"),
            record.get("modelName"),
            record.get("family"),
            record.get("precision"),
            record.get("metricName"),
            record.get("metricValue"),
            record.get("modelSizeMb"),
            record.get("latencyMs"),
            target.get("device"),
            target.get("runtime"),
            target.get("accelerator"),
            json.dumps(record.get("cpuFallbackOps", [])),
            record.get("verifiedAt", datetime.utcnow().isoformat()),
        ))

def load_all_benchmarks() -> List[Dict[str, Any]]:
    """Load all benchmark records from SQLite (Reads don't require locks under WAL)."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM benchmarks ORDER BY verified_at DESC").fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "source": row["source"],
                "modelName": row["model_name"],
                "family": row["family"],
                "precision": row["precision"],
                "metricName": row["metric_name"],
                "metricValue": row["metric_value"],
                "modelSizeMb": row["model_size_mb"],
                "latencyMs": row["latency_ms"],
                "target": {
                    "device": row["device"],
                    "runtime": row["runtime"],
                    "accelerator": row["accelerator"],
                },
                "cpuFallbackOps": json.loads(row["cpu_fallback_ops"] or "[]"),
                "verifiedAt": row["verified_at"],
            })
        return results
    finally:
        conn.close()

def upsert_aihub_job(record: Dict[str, Any]) -> None:
    """Insert or replace an AI Hub job record."""
    with write_transaction() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO aihub_jobs (
                id, model_name, device, runtime, compile_job_id, profile_job_id,
                status, latency_ms, cpu_fallback_ops, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("id"),
            record.get("model_name"),
            record.get("device"),
            record.get("runtime"),
            record.get("compile_job_id"),
            record.get("profile_job_id"),
            record.get("status", "completed"),
            record.get("latency_ms"),
            json.dumps(record.get("cpu_fallback_ops", [])),
            record.get("created_at", datetime.utcnow().isoformat()),
        ))
