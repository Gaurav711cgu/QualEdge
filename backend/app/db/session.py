"""
SQLite Session management for QualEdge database.
"""
import sqlite3
import os
import logging
from pathlib import Path

logger = logging.getLogger("Database-Session")
DB_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "qualedge.db"

def get_connection(timeout: float = 5.0) -> sqlite3.Connection:
    """
    Creates and configures a SQLite connection.
    Enables Write-Ahead Logging (WAL) for concurrent read/write stability.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=timeout, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Performance tuning & concurrency handling
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        return conn
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise
