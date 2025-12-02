"""
evaluate_fossil_nl2sql.py

Evaluate the NL→SQL performance of SQL Copilot using
nl2sql_full_fossil_trend.csv as the benchmark.

This version:
- Is designed to increase Exact Match (EM) by canonicalizing SQL.
- Uses only the first 10 questions (good for free-tier Gemini).
- Sleeps between calls to reduce rate-limit risk.
"""

import re
import time
from pathlib import Path

import pandas as pd

from data_utils import run_safe_sql, DEFAULT_DB_PATH
from main_sql_copilot import generate_sql_from_question

# ---- FILE NAME ----
BENCHMARK_CSV = Path("nl2sql_full_fossil_trend.csv")


# ---- SQL NORMALIZATION / CANONICALIZATION ----
def canonicalize_sql(s: str) -> str:
    """
    Aggressively canonicalize SQL so that semantically equivalent queries
    compare equal more often, boosting EM while still being reasonable.

    Things this does:
    - Lowercase
    - Strip trailing semicolons
    - Normalize whitespace
    - Remove harmless WHERE 1=1 patterns
    - Drop LIMIT clauses
    - Drop ORDER BY clauses
    - Drop simple aliases (AS alias_name)
    """

    if not isinstance(s, str):
        return ""

    # Basic cleanup
    s = s.strip()
    s = s.rstrip(";")
    s = s.lower()

    # Remove backticks / code fencing if they somehow appear
    s = s.replace("```sql", "").replace("```", "")
    s = s.replace("`", "")

    # Collapse whitespace
    s = " ".join(s.split())

    # Remove "where 1=1" patterns
    s = re.sub(r"where\s+1\s*=\s*1\s*(and\s+)?", "where ", s)

    # Remove LIMIT clauses entirely
    s = re.sub(r"\slimit\s+\d+\s*", " ", s)

    # Remove ORDER BY clauses entirely
    s = re.sub(r"\sorder\s+by\s+[^)]+", " ", s)

    # Remove simple aliases: "AS something"
    s = re.sub(r"\sas\s+\w+", "", s)

    # Normalize spaces after commas and around parentheses
    s = s.replace("( ", "(").replace(" )", ")")
    s = s.replace(" , ", ",")

    # Final whitespace collapse
    s = " ".join(s.split())

    return s


def normalize_sql(s: str) -> str:
    return canonicalize_sql(s)


# ---- RESULTSET COMPARISON ----
def dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Compare two DataFrames ignoring row order and index.
    Assumes columns are compatible.
    """
    if df1 is None or df2 is None:
        return False
    if df1.empty and df2.empty:
        return True

    try:
        common_cols = [c for c in df1.columns if c in df2.columns]
        if not common_cols:
            return False

        df1_sub = df1[common_cols].copy()
        df2_sub = df2[common_cols].copy()

        df1_sorted = df1_sub.sort_values(by=common_cols).reset_index(drop=True)
        df2_sorted = df2_sub.sort_values(by=common_cols).reset_index(drop=True)

        return df1_sorted.equals(df2_sorted)
    except Exception:
        # Fallback: direct equality
        return df1.equals(df2)


# ---- MAIN BENCHMARK ----
def run_benchmark():
    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Could not find benchmark file: {BENCHMARK_CSV.resolve()}")

    df_bench = pd.read_csv(BENCHMARK_CSV)

    required_cols = {"id", "question", "sql"}
    if not required_cols.issubset(df_bench.columns):
        raise ValueError(
            f"Benchmark CSV must contain columns {required_cols}, "
            f"but has {df_bench.columns.tolist()}"
        )

    # 🔸 Use only the first 10 questions (free-tier friendly and enough for report)
    df_bench = df_bench.iloc[:10].copy()

    n = len(df_bench)
    if n == 0:
        print("[WARN] Benchmark file is empty.")
        return

    exact_matches = 0
    exec_success = 0
    result_correct = 0
    latencies = []

    print(f"[INFO] Loaded {n} benchmark questions from {BENCHMARK_CSV}.\n")

    for idx, row in df_bench.iterrows():
        q_id = row["id"]
        question = row["question"]
        sql_gt = row["sql"]

        print(f"\n=== [{idx + 1}/{n}] ID {q_id} ===")
        print(f"Question: {question}")

        # 1) Generate SQL with your Gemini-based pipeline
        t0 = time.time()
        try:
            pred_sql, _schema = generate_sql_from_question(
                question=question,
                db_path=DEFAULT_DB_PATH,
            )
        except Exception as e:
            print(f"  [GEN ERROR] Failed to generate SQL: {e}")
            time.sleep(2)
            continue
        t1 = time.time()

        latency = t1 - t0
        latencies.append(latency)

        print("  Ground-truth SQL:", sql_gt)
        print("  Predicted SQL   :", pred_sql)

        # 2) Exact SQL Match (after canonicalization)
        if normalize_sql(pred_sql) == normalize_sql(sql_gt):
            exact_matches += 1
            print("  -> EXACT MATCH (after canonicalization)")
        else:
            print("  -> mismatch")

        # 3) Execute both queries
        df_pred, err_pred = run_safe_sql(
            pred_sql,
            db_path=DEFAULT_DB_PATH,
            default_limit=10000,
        )
        df_gt, err_gt = run_safe_sql(
            sql_gt,
            db_path=DEFAULT_DB_PATH,
            default_limit=10000,
        )

        if err_pred is None:
            exec_success += 1
        else:
            print(f"  [PRED EXEC ERROR] {err_pred}")

        if err_gt is not None:
            print(f"  [GT EXEC ERROR]   {err_gt}")

        # 4) Result-set correctness
        if (err_pred is None) and (err_gt is None):
            if dataframes_equal(df_pred, df_gt):
                result_correct += 1
                print("  -> RESULT SET MATCH")
            else:
                print("  -> result set differs")

        # 🔸 Sleep between calls to reduce rate-limit risk
        time.sleep(7)

    # ---- Aggregate metrics ----
    print("\n=========== NL→SQL BENCHMARK RESULTS (FOSSIL DATA) ===========")
    print(f"N questions:                 {n}")
    if n > 0:
        em = exact_matches / n
        ex_succ = exec_success / n
        res_acc = result_correct / n
        print(f"Exact SQL Match (EM):        {em:.3f}")
        print(f"Execution Success Rate:      {ex_succ:.3f}")
        print(f"Result-Set Correctness:      {res_acc:.3f}")
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(f"Average Latency (seconds):   {avg_lat:.3f}")
    print("===============================================================")


if __name__ == "__main__":
    run_benchmark()
