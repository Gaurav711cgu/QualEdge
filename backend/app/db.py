"""
SQLite persistence layer for QualEdge benchmark results.

Replaces the in-memory lists in compression_service.py that reset on every server
restart. A live demo where the interviewer restarts the server will now retain all
benchmark history.
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "qualedge.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_connection()
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
    conn.commit()
    conn.close()


def upsert_benchmark(record: Dict[str, Any]) -> None:
    """Insert or replace a benchmark record."""
    conn = get_connection()
    target = record.get("target", {})
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
    conn.commit()
    conn.close()


def load_all_benchmarks() -> List[Dict[str, Any]]:
    """Load all benchmark records from SQLite, returning them as dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM benchmarks ORDER BY verified_at DESC").fetchall()
    conn.close()
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


def upsert_aihub_job(record: Dict[str, Any]) -> None:
    """Insert or replace an AI Hub job record."""
    conn = get_connection()
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
    conn.commit()
    conn.close()
