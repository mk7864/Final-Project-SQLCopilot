"""
app.py

Streamlit UI for SQL Copilot:

- Upload CSV/Excel → converted into SQLite ('uploaded_data.db')
- Show schema
- Text box for natural language questions
- Display generated SQL
- Execute query safely and show results
- Visualize query results
- NEW: Separate section to explore & visualize the raw uploaded dataset
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_utils import (
    DEFAULT_DB_PATH,
    create_db_from_dataframe,
    get_schema,
    run_safe_sql,
)
from main_sql_copilot import generate_sql_from_question

# Make sure environment (GEMINI_API_KEY) is loaded
load_dotenv()

# -------------------------------------------------------------------
# Streamlit layout
# -------------------------------------------------------------------

st.set_page_config(
    page_title="SQL Copilot",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 SQL Copilot – Conversational SQL for Your Data")

# -------------------------------------------------------------------
# Sidebar: upload dataset
# -------------------------------------------------------------------

st.sidebar.header("1. Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xls", "xlsx"],
)

table_name = "uploaded_table"

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

if uploaded_file is not None:
    try:
        # Read uploaded file into a DataFrame
        file_bytes = uploaded_file.read()
        buffer = io.BytesIO(file_bytes)

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)

        # Save to SQLite
        _, table_name = create_db_from_dataframe(
            df=df,
            table_name=table_name,
            db_path=DEFAULT_DB_PATH,
        )
        st.session_state.db_ready = True

        # Also keep the full DataFrame in session_state for raw visualization
        st.session_state["uploaded_df"] = df

        st.sidebar.success(f"Loaded into SQLite as table: {table_name}")

        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head())

    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.session_state.db_ready = False
        st.session_state["uploaded_df"] = pd.DataFrame()

# -------------------------------------------------------------------
# Sidebar: show schema
# -------------------------------------------------------------------

st.sidebar.header("2. Database Schema")

if st.session_state.db_ready:
    schema = get_schema(db_path=DEFAULT_DB_PATH)
    if not schema:
        st.sidebar.warning("No tables detected in the database.")
    else:
        for table, cols in schema.items():
            st.sidebar.markdown(f"**{table}**")
            st.sidebar.write(", ".join(cols))
else:
    st.sidebar.info("Upload a dataset to initialize the database.")

# -------------------------------------------------------------------
# Section 3: NL-to-SQL interaction
# -------------------------------------------------------------------

st.header("3. Ask Questions in English")

# Use session_state to allow proper clearing
if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_input(
    "Type a question about your data:",
    placeholder="e.g., Show me total sales per year.",
    key="question",
)

col_run, col_clear = st.columns([1, 1])

with col_run:
    run_button = st.button("Generate & Run SQL")
with col_clear:
    clear_button = st.button("Clear")

# Clear button: reset the text input and rerun
if clear_button:
    st.session_state["question"] = ""
    st.rerun()

# -------------------------------------------------------------------
# When user clicks "Generate & Run SQL"
# -------------------------------------------------------------------

if run_button:
    if not st.session_state.db_ready:
        st.error("Please upload a dataset first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking with Gemini 2.5 Flash..."):
            try:
                sql_query, schema = generate_sql_from_question(
                    question,
                    db_path=DEFAULT_DB_PATH,
                    few_shot_examples=None,  # plug in your own examples if you want
                )
            except Exception as e:
                st.error(f"Error generating SQL: {e}")
                sql_query = ""

        if sql_query:
            st.subheader("Generated SQL")
            st.code(sql_query, language="sql")

            # Execute the SQL safely
            result_df, err_msg = run_safe_sql(sql_query, db_path=DEFAULT_DB_PATH)

            if err_msg:
                st.error(err_msg)
            else:
                st.subheader("Query Results")
                if result_df.empty:
                    st.info("Query executed successfully but returned no rows.")
                else:
                    st.dataframe(result_df)

                    # -------------------------------------------------------------------
                    # Visualization of QUERY RESULTS
                    # -------------------------------------------------------------------
                    st.subheader("Visualize Query Results (Optional)")

                    plot_type = st.selectbox(
                        "Choose a chart type for query results:",
                        ["None", "Bar", "Line", "Scatter"],
                        index=0,
                        key="query_plot_type",
                    )

                    if plot_type != "None":
                        numeric_cols = list(result_df.select_dtypes(include="number").columns)
                        all_cols = list(result_df.columns)

                        if not numeric_cols:
                            st.warning("No numeric columns available for plotting.")
                        else:
                            x_col = st.selectbox(
                                "X-axis column (query results):",
                                all_cols,
                                index=0,
                                key="query_x_col",
                            )
                            y_col = st.selectbox(
                                "Y-axis column (numeric):",
                                numeric_cols,
                                index=0,
                                key="query_y_col",
                            )

                            if x_col and y_col:
                                fig, ax = plt.subplots()
                                if plot_type == "Bar":
                                    ax.bar(result_df[x_col].astype(str), result_df[y_col])
                                    ax.set_xlabel(x_col)
                                    ax.set_ylabel(y_col)
                                    ax.set_title(f"{plot_type} chart of {y_col} by {x_col}")
                                    plt.xticks(rotation=45, ha="right")
                                elif plot_type == "Line":
                                    ax.plot(result_df[x_col], result_df[y_col])
                                    ax.set_xlabel(x_col)
                                    ax.set_ylabel(y_col)
                                    ax.set_title(f"{plot_type} chart of {y_col} vs {x_col}")
                                elif plot_type == "Scatter":
                                    ax.scatter(result_df[x_col], result_df[y_col])
                                    ax.set_xlabel(x_col)
                                    ax.set_ylabel(y_col)
                                    ax.set_title(f"{plot_type} plot of {y_col} vs {x_col}")

                                st.pyplot(fig)

# -------------------------------------------------------------------
# Section 4: Explore & Visualize the RAW Uploaded Dataset
# -------------------------------------------------------------------

st.header("4. Explore & Visualize Uploaded Data")

uploaded_df = st.session_state.get("uploaded_df", pd.DataFrame())

if uploaded_df is None or uploaded_df.empty:
    st.info("Upload a dataset in the sidebar to explore it here.")
else:
    st.subheader("Dataset Overview")

    # Basic info
    st.write(f"**Rows:** {uploaded_df.shape[0]} &nbsp;&nbsp; **Columns:** {uploaded_df.shape[1]}")

    with st.expander("Show full column list and types", expanded=False):
        col_info = pd.DataFrame(
            {
                "column": uploaded_df.columns,
                "dtype": [str(t) for t in uploaded_df.dtypes],
            }
        )
        st.dataframe(col_info)

    with st.expander("Preview first rows", expanded=False):
        st.dataframe(uploaded_df.head())

    st.subheader("Create a Chart from Uploaded Data")

    base_plot_type = st.selectbox(
        "Choose a chart type for uploaded data:",
        ["None", "Histogram", "Bar", "Line", "Scatter"],
        index=0,
        key="base_plot_type",
    )

    if base_plot_type != "None":
        numeric_cols = list(uploaded_df.select_dtypes(include="number").columns)
        all_cols = list(uploaded_df.columns)

        if not numeric_cols:
            st.warning("No numeric columns available for plotting.")
        else:
            if base_plot_type == "Histogram":
                hist_col = st.selectbox(
                    "Numeric column for histogram:",
                    numeric_cols,
                    index=0,
                    key="hist_col",
                )
                bins = st.slider("Number of bins:", min_value=5, max_value=50, value=20)

                fig, ax = plt.subplots()
                ax.hist(uploaded_df[hist_col].dropna(), bins=bins)
                ax.set_xlabel(hist_col)
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram of {hist_col}")
                st.pyplot(fig)

            else:
                x_col = st.selectbox(
                    "X-axis column (uploaded data):",
                    all_cols,
                    index=0,
                    key="base_x_col",
                )
                y_col = st.selectbox(
                    "Y-axis column (numeric):",
                    numeric_cols,
                    index=0,
                    key="base_y_col",
                )

                if x_col and y_col:
                    fig, ax = plt.subplots()

                    if base_plot_type == "Bar":
                        ax.bar(uploaded_df[x_col].astype(str), uploaded_df[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{base_plot_type} chart of {y_col} by {x_col}")
                        plt.xticks(rotation=45, ha="right")
                    elif base_plot_type == "Line":
                        ax.plot(uploaded_df[x_col], uploaded_df[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{base_plot_type} chart of {y_col} vs {x_col}")
                    elif base_plot_type == "Scatter":
                        ax.scatter(uploaded_df[x_col], uploaded_df[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{base_plot_type} plot of {y_col} vs {x_col}")

                    st.pyplot(fig)
