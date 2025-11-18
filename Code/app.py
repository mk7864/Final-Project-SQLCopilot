# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
from typing import Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Our helper module you saved as sql.py
from sql import open_store


# ===================== T5 NL → SQL helpers =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5-small-nl2sql")


@st.cache_resource
def load_t5_model():
    """
    Load the fine-tuned T5-small model once per Streamlit session.
    Falls back gracefully if the model directory is missing.
    """
    if not os.path.isdir(MODEL_DIR):
        st.warning(
            f"Model directory not found at {MODEL_DIR}. "
            "Make sure you ran 2_train_t5.py and the path is correct."
        )
        return None, None, torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_sql(sql: str) -> str:
    """
    Simple post-processing for model output:
    - strip whitespace
    - keep only up to the first ';'
    - de-duplicate columns in the SELECT list, if we can detect them
    """
    if not sql:
        return sql

    sql = sql.strip()

    # keep only the first statement (up to first ';')
    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"

    lower_sql = sql.lower()
    if not lower_sql.startswith("select"):
        return sql

    # Try to split on SELECT ... FROM ...
    if " from " in lower_sql:
        idx = lower_sql.index(" from ")
        # len("select ") == 7, but we keep case as-is from original string
        select_part = sql[7:idx].strip()
        from_part = sql[idx:]  # includes " from ..."

        cols = [c.strip() for c in select_part.split(",") if c.strip()]

        dedup_cols = []
        seen = set()
        for c in cols:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                dedup_cols.append(c)

        select_clean = ", ".join(dedup_cols) if dedup_cols else select_part
        return "SELECT " + select_clean + " " + from_part

    return sql


def generate_sql_from_nl(question: str, table_name: Optional[str] = None) -> str:
    """
    Use the fine-tuned T5-small model to turn natural language into SQL.
    Optionally, include the active table name as a hint.
    """
    tokenizer, model, device = load_t5_model()
    if tokenizer is None or model is None:
        return ""

    base_prefix = "translate English to SQL: "
    question = question.strip()

    if table_name:
        full_input = f"{base_prefix}{question} [TABLE={table_name}]"
    else:
        full_input = base_prefix + question

    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=80,
            num_beams=4,
            early_stopping=True,
        )

    raw_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sql(raw_sql)


# ===================== SQL store & plotting helpers =====================

def init_store():
    """Create or load a SQLite store for uploaded data."""
    if "store" not in st.session_state:
        # DB file will live next to Code/ as uploaded_data.db
        db_path = os.path.join(os.path.dirname(__file__), "uploaded_data.db")
        st.session_state.store = open_store(db_path=db_path)
    return st.session_state.store


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def auto_plot(df: pd.DataFrame):
    """Very simple plot chooser (matplotlib-only)."""
    if df.empty or df.shape[1] == 0:
        st.info("Nothing to plot.")
        return

    cols = list(df.columns)

    # Prefer (categorical, numeric) bar chart
    cat = next((c for c in cols if not is_numeric(df[c])), None)
    num = next((c for c in cols if is_numeric(df[c])), None)

    if cat is not None and num is not None:
        fig, ax = plt.subplots()
        grouped = df.groupby(cat)[num].sum().sort_values(ascending=False).head(30)
        ax.bar(grouped.index.astype(str), grouped.values)
        ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha="right")
        ax.set_xlabel(cat)
        ax.set_ylabel(num)
        ax.set_title(f"{num} by {cat}")
        fig.tight_layout()
        st.pyplot(fig)
        return

    # Else, if we have a numeric column, show histogram
    numeric_cols = [c for c in cols if is_numeric(df[c])]
    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna())
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        st.pyplot(fig)
        return

    # Else: simple value counts for first column
    c0 = cols[0]
    vc = df[c0].astype(str).value_counts().head(30)
    fig, ax = plt.subplots()
    ax.bar(range(len(vc.index)), vc.values)
    ax.set_xticks(range(len(vc.index)))
    ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")
    ax.set_title(f"{c0} — Value counts")
    ax.set_xlabel(c0)
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)


def get_sql_from_question(question: str, store, table_name: Optional[str]) -> str:
    """
    Wrapper that calls the local T5 model to generate SQL.
    'store' is unused here but kept for API compatibility.
    """
    question = question.strip()
    if not question:
        return ""
    sql_text = generate_sql_from_nl(question, table_name=table_name)
    return sql_text


# ===================== Streamlit UI =====================

st.set_page_config(page_title="SQL Copilot — NL2SQL", layout="wide")
st.title("SQL Copilot: Natural Language → SQL on Your Own Data")

st.write(
    "Upload a CSV or Excel file, let the app load it into SQLite, "
    "and then ask questions in natural language. A fine-tuned T5-small "
    "model generates SQL, which we run on your uploaded data."
)

store = init_store()

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    table_created = None
    if uploaded is not None:
        name = uploaded.name
        ext = os.path.splitext(name)[1].lower()

        try:
            if ext == ".csv":
                table_created = store.load_csv(uploaded, table_name=None, if_exists="replace")
            elif ext in (".xlsx", ".xls"):
                table_created = store.load_excel(uploaded, sheet=None, table_prefix=None, if_exists="replace")
            else:
                st.error("Unsupported file type.")
        except Exception as e:
            st.error(f"Failed to load file into SQLite: {e}")

        if table_created:
            st.success(f"Loaded data into table: `{table_created}`")
            st.session_state.last_table = table_created

    st.markdown("---")
    st.header("2. Model info")
    st.caption(
        "- Model: `t5-small` fine-tuned on a custom NL→SQL dataset\n"
        "- Backend: Hugging Face Transformers on CPU/GPU\n"
        "- Training: 30 epochs, final train loss ≈ 0.84"
    )

# --- Main layout ---
tables = store.list_tables()
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.subheader("Database schema")

    if not tables:
        st.info("No tables found yet. Upload a CSV/Excel file in the sidebar.")
        table_name = None
    else:
        default_idx = 0
        if "last_table" in st.session_state and st.session_state.last_table in tables:
            default_idx = tables.index(st.session_state.last_table)

        table_name = st.selectbox(
            "Select a table to explore",
            tables,
            index=default_idx,
        )

        if table_name:
            schema_rows = store.get_schema(table_name)
            if schema_rows:
                schema_df = pd.DataFrame(
                    schema_rows,
                    columns=["cid", "name", "type", "notnull", "dflt_value", "pk"],
                )
                st.dataframe(
                    schema_df[["name", "type", "notnull", "pk"]],
                    use_container_width=True,
                )

            with st.expander("Preview table data (first 100 rows)", expanded=False):
                try:
                    preview_df = store.run_select(f"SELECT * FROM {table_name} LIMIT 100;")
                    st.dataframe(preview_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to preview table: {e}")

with col_right:
    st.subheader("Ask questions in natural language")

    question = st.text_input(
        "Example: *Show total revenue per product*",
        value="",
        placeholder="Type your question about the selected table...",
    )

    run_clicked = st.button("Generate SQL and run", type="primary", disabled=not tables)

    if run_clicked:
        if not tables:
            st.warning("Please upload data and select a table first.")
        elif not question.strip():
            st.warning("Please enter a natural language question.")
        else:
            if not 'table_name' in locals() or not table_name:
                st.warning("Please select a table from the left panel.")
            else:
                with st.spinner("Generating SQL with T5-small..."):
                    sql_text = get_sql_from_question(question, store, table_name)

                if not sql_text:
                    st.error("The model did not return any SQL. Check that the model is saved correctly.")
                else:
                    st.markdown("**Generated SQL:**")
                    st.code(sql_text, language="sql")

                    try:
                        df_result = store.run_select(sql_text)
                    except Exception as e:
                        st.error(f"Database error while executing SQL: {e}")
                    else:
                        if df_result.empty:
                            st.info("Query executed successfully but returned zero rows.")
                        else:
                            st.subheader("Query results")
                            st.dataframe(df_result, use_container_width=True)

                            if st.toggle("Plot result", value=False):
                                auto_plot(df_result)
