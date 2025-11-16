# sql.py
import io
import re
import os
import sqlite3
from typing import Iterable, Optional, Union, List, Dict, Tuple

import pandas as pd


# --------------------------- Sanitizers & Normalizers ---------------------------

def sanitize_to_select(text: str) -> str:
    """
    Make LLM/model output executable by SQLite safely:
    - Strip ```sql / ```sqlite fences
    - Remove leading 'sql/sqlite' language tags
    - Keep only from the first SELECT... (case-insensitive)
    - Trim at the first semicolon if present
    """
    if not text:
        return ""
    s = text.strip()

    # Remove fenced blocks like ```sql ... ```
    s = re.sub(r"^```[\w-]*\s*", "", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)

    # Drop comment/empty lines up front
    lines = [ln for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    if not lines:
        return ""

    # Remove a bare first-line language tag like "sql" or "sqlite"
    if re.fullmatch(r"(sql|sqlite)\s*", lines[0].strip(), flags=re.IGNORECASE):
        lines = lines[1:]

    s = "\n".join(lines).strip()

    # Find the first occurrence of SELECT
    m = re.search(r"\bselect\b", s, flags=re.IGNORECASE)
    if not m:
        return ""
    s = s[m.start():].strip()

    # Keep up to first semicolon if present
    semi = s.find(";")
    if semi != -1:
        s = s[:semi].strip()

    return s


# Common synonym patterns → canonical column names
_SYNONYM_MAP: Dict[str, str] = {
    r"\bstudent[_\s]*name\b": "NAME",
    r"\bname\b": "NAME",
    r"\bscore\b": "MARKS",
    r"\bmarks?\b": "MARKS",
    r"\bgrade(s)?\b": "MARKS",
    r"\bclass(name)?\b": "CLASS",
    r"\bsection\b": "SECTION",
}

def normalize_sql_columns(sql: str, valid_cols: Iterable[str]) -> str:
    """
    Replace common synonyms with actual column names (case-insensitive).
    Then upper/normalize exact valid column tokens for good measure.
    """
    s = sql
    for pat, col in _SYNONYM_MAP.items():
        s = re.sub(pat, col, s, flags=re.IGNORECASE)
    for col in valid_cols:
        s = re.sub(rf"\b{re.escape(col)}\b", col, s, flags=re.IGNORECASE)
    return s


# --------------------------- SQLite Utilities ---------------------------

def _sanitize_identifier(name: str) -> str:
    """Make a safe SQLite identifier from an arbitrary name (filename/sheet/column)."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "TBL"
    # Avoid starting with digit
    if re.match(r"^\d", cleaned):
        cleaned = f"T_{cleaned}"
    return cleaned


def _sanitize_columns(cols: Iterable[str]) -> List[str]:
    """Return a list of safe column names, preserving order."""
    seen = set()
    out = []
    for c in cols:
        base = _sanitize_identifier(c)
        name = base
        i = 1
        while name.upper() in seen:
            name = f"{base}_{i}"
            i += 1
        seen.add(name.upper())
        out.append(name)
    return out


class SQLiteDataStore:
    """
    Convenience wrapper around a SQLite database that:
      - Loads CSV/Excel into tables (with safe names)
      - Lists tables and reads schema
      - Runs SELECT queries safely with sanitizer + optional synonym auto-repair
    """

    def __init__(self, db_path: str = "uploaded_data.db"):
        self.db_path = db_path
        # ensure file/dir exists (SQLite will create file as needed)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    # ---- connection ----
    def connect(self):
        return sqlite3.connect(self.db_path)

    # ---- metadata ----
    def list_tables(self) -> List[str]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            return [r[0] for r in cur.fetchall()]

    def get_schema(self, table: str) -> List[Tuple]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            return cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

    # ---- loaders ----
    def load_csv(
        self,
        file: Union[str, io.BytesIO],
        table_name: Optional[str] = None,
        if_exists: str = "replace",
        dtype: Optional[Dict[str, str]] = None,
        **read_csv_kwargs,
    ) -> str:
        """
        Load a CSV into SQLite.

        Args:
          file: path or BytesIO
          table_name: optional name; sanitized if provided; otherwise from filename or 'CSV_TABLE'
          if_exists: 'replace' | 'append' | 'fail'
          dtype: optional pandas dtype mapping
          **read_csv_kwargs: passed to pandas.read_csv

        Returns:
          The final SQLite table name used.
        """
        # Read CSV
        if isinstance(file, (str, os.PathLike)):
            df = pd.read_csv(file, dtype=dtype, **read_csv_kwargs)
            inferred = os.path.splitext(os.path.basename(str(file)))[0]
        else:
            df = pd.read_csv(file, dtype=dtype, **read_csv_kwargs)
            inferred = "CSV_TABLE"

        # Sanitize columns
        df = df.copy()
        df.columns = _sanitize_columns(df.columns)

        # Decide table name
        tbl = _sanitize_identifier(table_name) if table_name else _sanitize_identifier(inferred)

        with self.connect() as conn:
            df.to_sql(tbl, conn, if_exists=if_exists, index=False)
        return tbl

    def load_excel(
        self,
        file: Union[str, io.BytesIO],
        sheet: Optional[Union[str, int]] = None,
        if_exists: str = "replace",
        dtype: Optional[Dict[str, str]] = None,
        **read_excel_kwargs,
    ) -> List[str]:
        """
        Load an Excel file into SQLite.
        - If 'sheet' is None, loads **all sheets** (one table per sheet).
        - If 'sheet' is provided (name or index), loads only that sheet.

        Table naming:
          <workbook_name>__<sheet_name>  (both sanitized)

        Returns:
          List of tables created/updated.
        """
        # Read once with pandas
        if isinstance(file, (str, os.PathLike)):
            xls = pd.ExcelFile(file, engine=read_excel_kwargs.pop("engine", "openpyxl"))
            base = os.path.splitext(os.path.basename(str(file)))[0]
        else:
            # file-like buffer
            xls = pd.ExcelFile(file, engine=read_excel_kwargs.pop("engine", "openpyxl"))
            base = "XLS"

        tables_created: List[str] = []

        # Which sheets?
        if sheet is None:
            sheets = xls.sheet_names
        else:
            sheets = [sheet]

        for sh in sheets:
            df = xls.parse(sh, dtype=dtype, **read_excel_kwargs).copy()
            df.columns = _sanitize_columns(df.columns)
            tbl_name = _sanitize_identifier(f"{base}__{sh}")
            with self.connect() as conn:
                df.to_sql(tbl_name, conn, if_exists=if_exists, index=False)
            tables_created.append(tbl_name)

        return tables_created

    # ---- query runner ----
    def run_select(
        self,
        sql_text: str,
        autorepair_synonyms: bool = True,
    ) -> pd.DataFrame:
        """
        Execute a SELECT query safely.
        - Sanitizes LLM output to the first clean SELECT statement
        - If 'no such column' occurs and autorepair is True, tries synonym normalization
          against the detected table in the query (simple heuristic).

        Returns: pandas DataFrame
        Raises: sqlite3.OperationalError on DB errors if not repairable
        """
        cleaned = sanitize_to_select(sql_text)
        if not cleaned.lower().startswith("select"):
            raise ValueError(f"Refusing to execute non-SELECT SQL after sanitation: {sql_text!r}")

        try:
            with self.connect() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(cleaned)
                rows = cur.fetchall()
                cols = rows[0].keys() if rows else []
                return pd.DataFrame(rows, columns=cols)
        except sqlite3.OperationalError as e:
            if not autorepair_synonyms:
                raise

            msg = str(e).lower()
            if "no such column" in msg:
                # naive: try find a mentioned table and normalize columns
                target_table = None
                for t in self.list_tables():
                    if re.search(rf"\b{re.escape(t)}\b", cleaned, flags=re.IGNORECASE):
                        target_table = t
                        break
                if target_table:
                    schema = self.get_schema(target_table)
                    valid_cols = [c[1] for c in schema]
                    repaired = normalize_sql_columns(cleaned, valid_cols)
                    if repaired != cleaned:
                        with self.connect() as conn2:
                            conn2.row_factory = sqlite3.Row
                            cur2 = conn2.cursor()
                            cur2.execute(repaired)
                            rows = cur2.fetchall()
                            cols = rows[0].keys() if rows else []
                            return pd.DataFrame(rows, columns=cols)

            # Could not repair
            raise


# --------------------------- Convenience, top-level helpers ---------------------------

def open_store(db_path: str = "uploaded_data.db") -> SQLiteDataStore:
    """Shorthand to construct the data store."""
    return SQLiteDataStore(db_path=db_path)


__all__ = [
    "SQLiteDataStore",
    "open_store",
    "sanitize_to_select",
    "normalize_sql_columns",
]
