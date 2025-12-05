"""
main_sql_copilot.py

Core logic for SQL Copilot:
- Text preprocessing & light synonym normalization
- Schema-aware prompt construction
- Call to Gemini 2.5 Flash to generate SQL
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import google.generativeai as genai

from data_utils import DEFAULT_DB_PATH, get_schema

# -------------------------------------------------------------------
# Environment & Gemini setup
# -------------------------------------------------------------------

# Folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load .env from this folder
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# -------------------------------------------------------------------
# Simple rule-based synonym normalization
# -------------------------------------------------------------------

BASE_SYNONYMS = {
    "sales": ["revenue", "income"],
    "customers": ["clients", "users"],
    "country": ["nation"],
    "year": ["yr"],
}


def build_synonym_map(schema: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build reverse lookup mapping from synonym words to canonical column names.
    """
    reverse_map: Dict[str, str] = {}
    all_columns = {col.lower() for cols in schema.values() for col in cols}

    for canonical, syns in BASE_SYNONYMS.items():
        if canonical.lower() in all_columns:
            for s in syns:
                reverse_map[s.lower()] = canonical
    return reverse_map


def normalize_question(question: str, schema: Dict[str, List[str]]) -> str:
    """
    Lowercase & replace known synonyms with actual column names when present.
    """
    q = question.strip()
    syn_map = build_synonym_map(schema)

    def replace_token(match: re.Match) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in syn_map:
            return syn_map[lower]
        return token

    normalized = re.sub(r"\b\w+\b", replace_token, q)
    return normalized


# -------------------------------------------------------------------
# Prompt construction & Gemini call
# -------------------------------------------------------------------

def format_schema_for_prompt(schema: Dict[str, List[str]]) -> str:
    """
    Format DB schema into a text block for the LLM prompt.
    """
    lines = []
    for table, cols in schema.items():
        col_list = ", ".join(cols)
        lines.append(f"- {table}({col_list})")
    return "\n".join(lines)


def build_prompt(
    question: str,
    schema: Dict[str, List[str]],
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Build a schema-aware prompt for Gemini.

    few_shot_examples: list of (natural_language_question, correct_sql)
    """
    schema_text = format_schema_for_prompt(schema)

    examples_text = ""
    if few_shot_examples:
        ex_lines = []
        for nl, sql in few_shot_examples:
            ex_lines.append(f"Q: {nl}\nSQL: {sql.strip()}")
        examples_text = "\n\n".join(ex_lines)

    prompt = f"""
You are an expert SQL assistant that translates English questions into valid SQLite SQL queries.

Database schema:
{schema_text}

Rules:
- Use ONLY the tables and columns listed in the schema above.
- Do NOT invent new columns or tables.
- The database is SQLite, so use SQLite-compatible syntax.
- Prefer simple SELECT queries.
- DO NOT perform any data modification (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, etc.).
- If the user does not specify a limit, return a reasonable number of rows (e.g., 100).
- Output ONLY the SQL query, nothing else. Do not use markdown or backticks.

Here are some examples (if any are provided):

{examples_text}

Now generate a SQL query for the following question:

Q: {question}
SQL:
""".strip()

    return prompt


def call_gemini_for_sql(prompt: str) -> str:
    """
    Call Gemini 2.5 Flash and extract the SQL text.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    text = response.text.strip()
    # Strip ```sql ... ``` if the model adds code fences
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_sql_from_question(
    question: str,
    db_path: Path = DEFAULT_DB_PATH,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, Dict[str, List[str]]]:
    """
    High-level function used by the Streamlit app:
    - Read schema from SQLite
    - Normalize the question
    - Build prompt & call Gemini
    - Return (sql, schema_dict)
    """
    schema = get_schema(db_path=db_path)
    if not schema:
        raise RuntimeError("No tables found in SQLite database. Upload a dataset first.")

    normalized_question = normalize_question(question, schema)
    prompt = build_prompt(
        normalized_question,
        schema,
        few_shot_examples=few_shot_examples,
    )
    sql = call_gemini_for_sql(prompt)
    return sql, schema
