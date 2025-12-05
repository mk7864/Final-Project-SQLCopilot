"""
mywork.py – Ayush Meshram (Individual Work Extraction)

Contains:
1. SQL Safety Layer
2. Synonym Normalization Logic
3. Prompt Engineering + Gemini Integration
4. Evaluation Dataset Construction & Execution
5. Performance Testing & Latency Evaluation
"""

import os
import re
import time
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
import google.generativeai as genai


# ==========================================================
# 1. SAFETY LAYER: BLOCK HARMFUL SQL + AUTO LIMIT
# ==========================================================

DEFAULT_DB_PATH = Path("uploaded_data.db")

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "CREATE", "ATTACH", "DETACH"
]


def is_query_safe(sql: str) -> bool:
    """Check SQL query for unsafe keywords."""
    upper = sql.upper()
    return not any(kw in upper for kw in FORBIDDEN_KEYWORDS)


def ensure_limit_clause(sql: str, default_limit: int = 200) -> str:
    """Add LIMIT if not present to avoid large outputs."""
    sql_stripped = sql.strip().rstrip(";")
    upper = sql_stripped.upper()

    if not upper.startswith("SELECT"):
        return sql_stripped + ";"
    if "LIMIT " in upper:
        return sql_stripped + ";"
    return f"{sql_stripped} LIMIT {default_limit};"


def run_safe_sql(sql: str, db_path: Path = DEFAULT_DB_PATH) -> Tuple[pd.DataFrame, Optional[str]]:
    """Execute SQL safely with error handling."""
    if not is_query_safe(sql):
        return pd.DataFrame(), "Blocked potentially unsafe query."
    sql_limited = ensure_limit_clause(sql)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_limited, conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"SQL execution error: {e}"
    finally:
        conn.close()


# ==========================================================
# 2. SYNONYM NORMALIZATION LOGIC
# ==========================================================

BASE_SYNONYMS = {
    "user": ["customer", "client", "member"],
    "year": ["yr", "fiscal year"],
    "country": ["nation", "region"],
    "sales": ["revenue", "turnover"],
    "profit": ["margin", "net income"],
    "fossil_share": ["fossil fuel share", "fossil dependence"]
}


def build_synonym_map(schema: Dict[str, List[str]]) -> Dict[str, str]:
    """Create reverse map from synonyms to schema column names."""
    reverse = {}
    all_cols = {c.lower() for cols in schema.values() for c in cols}
    for canonical, syns in BASE_SYNONYMS.items():
        if canonical.lower() in all_cols:
            for s in syns:
                reverse[s.lower()] = canonical
    return reverse


def normalize_question(question: str, schema: Dict[str, List[str]]) -> str:
    """Replace synonyms in NL question with true column names."""
    syn_map = build_synonym_map(schema)
    def repl(m):
        tok = m.group(0).lower()
        return syn_map.get(tok, m.group(0))
    return re.sub(r"\b\w+\b", repl, question.strip())


# ==========================================================
# 3. PROMPT ENGINEERING + GEMINI INTEGRATION
# ==========================================================

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"


def format_schema_for_prompt(schema: Dict[str, List[str]]) -> str:
    """Format schema into text block for LLM."""
    return "\n".join(f"- {t}({', '.join(cols)})" for t, cols in schema.items())


def build_prompt(question: str, schema: Dict[str, List[str]]) -> str:
    """Construct final Gemini prompt with safety rules."""
    schema_text = format_schema_for_prompt(schema)
    return f"""
You are an expert SQL assistant.
Use only the schema below to write safe, read-only SQL queries.

Schema:
{schema_text}

Rules:
- No DDL/DML (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE)
- Return only one SELECT statement with LIMIT if needed.
- Do not add markdown or text explanations.

Question: {question}
SQL:
""".strip()


def call_gemini_for_sql(prompt: str) -> str:
    """Query Gemini and clean output."""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    text = response.text.strip()
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)
    return text.strip().rstrip(";") + ";"


def generate_sql_from_question(question: str, schema: Dict[str, List[str]]) -> str:
    """Full pipeline: normalization → prompt → Gemini → SQL."""
    q = normalize_question(question, schema)
    prompt = build_prompt(q, schema)
    sql = call_gemini_for_sql(prompt)
    if not is_query_safe(sql):
        raise ValueError("Unsafe SQL blocked by safety layer.")
    return sql


# ==========================================================
# 4. EVALUATION DATASET (30 Q/A pairs) + PERFORMANCE TESTING
# ==========================================================

def canonicalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (remove whitespace, case)."""
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate_model(
    eval_csv: Path,
    schema: Dict[str, List[str]],
    db_path: Path = DEFAULT_DB_PATH,
) -> Dict[str, float]:
    """
    Evaluate model on benchmark dataset (30 Q/A pairs).
    Returns exact match accuracy and avg latency.
    """
    df = pd.read_csv(eval_csv)
    exact, total = 0, len(df)
    latencies = []

    for _, row in df.iterrows():
        q, gold = row["question"], row["gold_sql"]
        t0 = time.time()
        try:
            pred = generate_sql_from_question(q, schema)
        except Exception:
            pred = ""
        latencies.append(time.time() - t0)
        if canonicalize_sql(pred) == canonicalize_sql(gold):
            exact += 1

    return {
        "total": total,
        "exact_match": exact / total if total else 0,
        "avg_latency_sec": sum(latencies) / len(latencies)
    }


# ==========================================================
# MAIN EXECUTION 
# ==========================================================

if __name__ == "__main__":
    from data_utils import get_schema
    schema = get_schema(DEFAULT_DB_PATH)
    results = evaluate_model(Path("nl2sql_full_fossil_trend.csv"), schema)
    print("Evaluation Results:", results)
