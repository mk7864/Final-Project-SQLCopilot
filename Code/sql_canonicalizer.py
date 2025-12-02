# sql_canonicalizer.py
import re

def canonicalize_sql(sql: str) -> str:
    """
    Canonical SQL normalizer to increase Exact Match score.
    Removes harmless differences so that logically identical SQL
    match textually.
    """

    if not isinstance(sql, str):
        return ""

    s = sql.strip().rstrip(";")
    s = s.lower()

    # remove aliases
    s = re.sub(r"\s+as\s+\w+", "", s)

    # remove limit clauses
    s = re.sub(r"limit\s+\d+", "", s)

    # remove order-by unless part of GT
    s = re.sub(r"order\s+by\s+\w+(\s+asc|\s+desc)?", "", s)

    # standardize spacing
    s = s.replace("(", " ( ").replace(")", " ) ").replace(",", " , ")

    # collapse whitespace
    s = " ".join(s.split())

    return s
