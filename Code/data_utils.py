"""
data_utils.py

Utility functions for:
- Loading CSV/Excel into a SQLite database
- Inspecting database schema
- Safely executing SQL queries (read-only, auto-LIMIT)

Used by both the core logic and the Streamlit app.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Default SQLite DB file name (created in project root)
DEFAULT_DB_PATH = Path("uploaded_data.db")


def create_db_from_dataframe(
    df: pd.DataFrame,
    table_name: str = "uploaded_table",
    db_path: Path = DEFAULT_DB_PATH,
) -> Tuple[Path, str]:
    """
    Store a Pandas DataFrame into a SQLite database.

    Returns:
        (db_path, table_name)
    """
    if df.empty:
        raise ValueError("Uploaded file appears to be empty.")

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return db_path, table_name


def get_schema(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, List[str]]:
    """
    Inspect the SQLite database and return a dict:
        {table_name: [col1, col2, ...], ...}
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema: Dict[str, List[str]] = {}
        for t in tables:
            cursor.execute(f"PRAGMA table_info('{t}');")
            cols = [row[1] for row in cursor.fetchall()]
            schema[t] = cols

    finally:
        conn.close()

    return schema


FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "ATTACH",
    "DETACH",
]


def is_query_safe(sql: str) -> bool:
    """
    Very simple safety check:
    - Disallow obvious write / DDL operations.
    """
    upper = sql.upper()
    return not any(kw in upper for kw in FORBIDDEN_KEYWORDS)


def ensure_limit_clause(sql: str, default_limit: int = 200) -> str:
    """
    For SELECT queries, ensure there's some LIMIT to avoid huge result sets.
    """
    sql_stripped = sql.strip().rstrip(";")
    upper = sql_stripped.upper()

    if not upper.startswith("SELECT"):
        return sql_stripped + ";"

    if "LIMIT " in upper:
        return sql_stripped + ";"

    return f"{sql_stripped} LIMIT {default_limit};"


def run_safe_sql(
    sql: str,
    db_path: Path = DEFAULT_DB_PATH,
    default_limit: int = 200,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Execute a SQL query against the SQLite db with basic safety checks.

    Returns:
        (result_df, error_message)
        If error_message is not None, result_df may be empty.
    """
    if not is_query_safe(sql):
        return pd.DataFrame(), "Blocked potentially unsafe query."

    sql_limited = ensure_limit_clause(sql, default_limit=default_limit)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_limited, conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"SQL execution error: {e}"
    finally:
        conn.close()
