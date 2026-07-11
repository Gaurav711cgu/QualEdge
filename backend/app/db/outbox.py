"""
SQLite-backed Transactional Outbox implementation for robust asynchronous job scheduling.
"""
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from backend.app.db.session import get_connection

logger = logging.getLogger("Transactional-Outbox")

def init_outbox() -> None:
    """Creates the outbox table if not exists."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                error TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize outbox database: {e}")
    finally:
        conn.close()

def enqueue_outbox_event(conn: sqlite3.Connection, event_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """
    Enqueues a new action into the outbox.
    Must be executed within the active database transaction session passed as 'conn'.
    """
    conn.execute("""
        INSERT INTO outbox_events (id, event_type, payload, status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        event_id,
        event_type,
        json.dumps(payload),
        "pending",
        datetime.utcnow().isoformat()
    ))
    logger.info(f"Enqueued Transactional Outbox event {event_id} of type '{event_type}'.")

def get_pending_events() -> List[Dict[str, Any]]:
    """Retrieves all pending events from the outbox."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM outbox_events WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        events = []
        for r in rows:
            events.append({
                "id": r["id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload"]),
                "status": r["status"]
            })
        return events
    finally:
        conn.close()

def update_event_status(event_id: str, status: str, error: Optional[str] = None) -> None:
    """Updates the status and processed time of an event in the outbox."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE outbox_events
            SET status = ?, processed_at = ?, error = ?
            WHERE id = ?
        """, (
            status,
            datetime.utcnow().isoformat() if status in ("completed", "failed") else None,
            error,
            event_id
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update outbox event {event_id}: {e}")
    finally:
        conn.close()

# Singleton background worker
_worker_started = False
_worker_lock = threading.Lock()

def start_outbox_worker(pipeline_callback) -> None:
    """
    Starts a background worker thread that monitors the outbox table
    and processes pending compression suite jobs.
    """
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True

    init_outbox()

    def worker_loop():
        logger.info("Transactional Outbox Worker thread started.")
        while True:
            try:
                events = get_pending_events()
                for event in events:
                    event_id = event["id"]
                    event_type = event["event_type"]
                    payload = event["payload"]

                    logger.info(f"Worker picked up event {event_id} of type '{event_type}'")
                    update_event_status(event_id, "processing")

                    if event_type == "compression_run":
                        model_name = payload["model_name"]
                        ood_calibration = payload["ood_calibration"]
                        try:
                            # Execute optimization pipeline synchronously on worker thread
                            pipeline_callback(event_id, model_name, ood_calibration)
                            update_event_status(event_id, "completed")
                        except Exception as ex:
                            logger.error(f"Outbox pipeline execution failed for {event_id}: {ex}")
                            update_event_status(event_id, "failed", error=str(ex))
                    else:
                        update_event_status(event_id, "failed", error=f"Unknown event type: {event_type}")

            except Exception as e:
                logger.error(f"Error in outbox worker loop: {e}")
            
            time.sleep(1.0) # Poll every second

    t = threading.Thread(target=worker_loop, daemon=True, name="Outbox-Worker")
    t.start()
