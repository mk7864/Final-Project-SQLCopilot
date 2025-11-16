# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Our helper module you saved as sql.py
from sql import open_store

# Gemini (new client)
from google import genai

# ===================== Config =====================
DB_PATH = "uploaded_data.db"  # same default used in sql.py

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing GEMINI_API_KEY (or GOOGLE_API_KEY).\n"
        "Create a .env next to app.py with:\n"
        "GEMINI_API_KEY=your_key_here"
    )
client = genai.Client(api_key=api_key)

SYSTEM_PROMPT_BASE = """
You convert English questions into valid **SQLite SELECT** queries.

STRICT RULES:
- Use ONLY the provided table and column names exactly as given. Do NOT invent names.
- Return ONLY the SQL text (no commentary, no code fences, no leading 'sql').
- SQLite syntax. Exactly ONE statement. SELECT-only.
"""

# ===================== LLM helper (schema-aware) =====================
def build_schema_prompt(store, table_name: str) -> str:
    """Return a short, strict system prompt scoped to a specific table/columns."""
    schema = store.get_schema(table_name)  # list of tuples (cid, name, type, notnull, dflt, pk)
    cols = [c[1] for c in schema]
    schema_str = ", ".join(cols) if cols else "(no columns)"
    return (
        SYSTEM_PROMPT_BASE
        + f"\nTarget table: {table_name}\nColumns: {schema_str}\n"
          "If asked for 'student name', use NAME. "
          "If asked for 'score/marks', use MARKS. "
          "If asked for 'class', use CLASS. "
          "If asked for 'section', use SECTION."
    )

def get_sql_from_question(question: str, store, table_name: str) -> str:
    sys_prompt = build_schema_prompt(store, table_name)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[sys_prompt.strip(), question.strip()]
    )
    return (getattr(resp, "text", "") or "").strip()

# ===================== Simple plotting (optional) =====================
def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def auto_plot(df: pd.DataFrame):
    """Super simple plot chooser (matplotlib-only)."""
    if df.empty or df.shape[1] == 0:
        st.info("Nothing to plot.")
        return

    cols = list(df.columns)
    # Prefer a (cat, num) bar
    cat = next((c for c in cols if not is_numeric(df[c])), None)
    num = next((c for c in cols if is_numeric(df[c])), None)

    if cat is not None and num is not None:
        fig, ax = plt.subplots()
        # Aggregate if necessary (groupby)
        agg = df.groupby(cat, dropna=False)[num].mean().sort_values(ascending=False).head(30)
        ax.bar(range(len(agg.index)), agg.values)
        ax.set_xticks(range(len(agg.index)))
        ax.set_xticklabels([str(x) for x in agg.index], rotation=45, ha="right")
        ax.set_title(f"{num} by {cat}")
        ax.set_xlabel(cat); ax.set_ylabel(num)
        fig.tight_layout(); st.pyplot(fig); return

    # Else if one numeric column → histogram
    if num is not None:
        fig, ax = plt.subplots()
        clean = pd.to_numeric(df[num], errors="coerce").dropna()
        bins = min(30, max(5, int(len(clean) ** 0.5)))
        ax.hist(clean, bins=bins)
        ax.set_title(f"{num} — Histogram"); ax.set_xlabel(num); ax.set_ylabel("Frequency")
        fig.tight_layout(); st.pyplot(fig); return

    # Else count bar for first column
    c0 = cols[0]
    vc = df[c0].astype(str).value_counts().head(30)
    fig, ax = plt.subplots()
    ax.bar(range(len(vc.index)), vc.values)
    ax.set_xticks(range(len(vc.index)))
    ax.set_xticklabels(vc.index, rotation=45, ha="right")
    ax.set_title(f"{c0} — Value counts")
    ax.set_xlabel(c0); ax.set_ylabel("Count")
    fig.tight_layout(); st.pyplot(fig)

# ===================== UI =====================
st.set_page_config(page_title="Upload → SQLite → Ask in English", page_icon="🗃️", layout="wide")
st.title("🗃️ Upload → SQLite → Ask in English")

store = open_store(DB_PATH)

# ---- Sidebar: upload & load ----
st.sidebar.header("1) Upload your data")
up = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

load_btn = st.sidebar.button("Load file into SQLite")

# Excel options
excel_sheet = None
if up is not None and up.name.lower().endswith((".xlsx", ".xls")):
    st.sidebar.caption("Excel options")
    xls_buf = io.BytesIO(up.getbuffer())
    try:
        xls = pd.ExcelFile(xls_buf)
        all_sheets = ["<All sheets>"] + xls.sheet_names
        pick = st.sidebar.selectbox("Choose sheet", all_sheets, index=0)
        if pick != "<All sheets>":
            excel_sheet = pick
    except Exception as e:
        st.sidebar.error(f"Could not read Excel: {e}")

if load_btn and up is not None:
    try:
        if up.name.lower().endswith(".csv"):
            tbl = store.load_csv(io.BytesIO(up.getbuffer()))
            st.sidebar.success(f"Loaded CSV into table: {tbl}")
        else:
            # Excel
            tables = store.load_excel(io.BytesIO(up.getbuffer()), sheet=excel_sheet)
            if excel_sheet is None:
                st.sidebar.success(f"Loaded all sheets into tables: {', '.join(tables)}")
            else:
                st.sidebar.success(f"Loaded sheet into table: {tables[0]}")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# ---- Tables present ----
try:
    tables = store.list_tables()
except Exception as e:
    tables = []
    st.error(f"DB error: {e}")

if not tables:
    st.info("Upload a CSV/Excel and click **Load file into SQLite** to begin.")
    st.stop()

st.sidebar.header("2) Pick a table")
table_name = st.sidebar.selectbox("Table", tables, index=0)

# ---- Main workspaces ----
mode = st.radio("Workspace", ["Ask in English (Text → SQL)", "Run SQL directly"])

# ===================== Ask in English =====================
if mode.startswith("Ask"):
    st.subheader("Ask a question about your uploaded data")
    st.caption(f"Target table: `{table_name}`")

    q = st.text_input("Example: Who scored the highest marks in Data Science class?")
    ask_btn = st.button("Generate SQL & Run")

    if ask_btn:
        if not q.strip():
            st.warning("Please enter a question.")
            st.stop()
        with st.spinner("Thinking..."):
            try:
                raw_sql = get_sql_from_question(q, store, table_name)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        st.subheader("Generated SQL")
        st.code(raw_sql or "<empty>", language="sql")

        if not raw_sql:
            st.error("The model returned an empty query.")
            st.stop()

        try:
            df = store.run_select(raw_sql)  # sql.py handles sanitization + synonym auto-repair
        except Exception as e:
            st.error(f"DB error: {e}")
            st.stop()

        if df.empty:
            st.info("Query returned zero rows.")
        else:
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            if st.toggle("Plot the result"):
                auto_plot(df)

# ===================== Run SQL directly =====================
else:
    st.subheader("Write a SELECT query and run it")
    example = f"SELECT * FROM {table_name} LIMIT 50;"
    sql_text = st.text_area("SQL (SELECT only)", value=example, height=160)
    run_btn = st.button("Run")

    if run_btn:
        try:
            df = store.run_select(sql_text)  # sanitized inside
        except Exception as e:
            st.error(f"DB error: {e}")
            st.stop()

        if df.empty:
            st.info("Query returned zero rows.")
        else:
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            if st.toggle("Plot the result"):
                auto_plot(df)
