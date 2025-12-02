# 🧠 SQL Copilot — Natural Language to SQL (Gemini Integrated)

This project implements an interactive **SQL Copilot** powered by **Google Gemini** and **SQLite**
to help users generate SQL queries **from natural language questions**.

Users can:
✔️ Upload a dataset (CSV/Excel)  
✔️ Auto-load into SQLite  
✔️ Ask English questions → Get SQL queries  
✔️ Execute queries and visualize results instantly  
✔️ Evaluate model-generated SQL using benchmark data  

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| NL → SQL Generation | Uses Gemini model for semantic query generation |
| Automatic DB Loading | Converts uploaded file to SQLite table |
| Interactive UI | Streamlit app for execution and visualization |
| Chart Creation | Bar/line charts based on SQL output data |
| Evaluation Script | Benchmarks SQL Copilot with fossil dataset |

---

## 🛠️ Project Structure
```text
Final-Project-SQLCopilot/
│
├── Code/
│   ├── app.py                      # Streamlit UI
│   ├── main_sql_copilot.py         # NL → SQL logic using Gemini
│   ├── data_utils.py               # DB + schema utilities
│   ├── sql_canonicalizer.py        # SQL normalization for evaluation
│   ├── evaluate_fossil_nl2sql.py   # Benchmarking script
│   ├── requirements.txt            # Python dependencies
│   └── uploaded_data.db            # Generated after uploading dataset
│
└── data/
    └── nl2sql_full.csv             # Benchmark dataset'''

# 👉 Installation
git clone https://github.com/mk7864/Final-Project-SQLCopilot.git
cd Final-Project-SQLCopilot/Code
pip install -r requirements.txt

# 👉 Environment Setup
# Create a .env file inside the Code folder
echo GEMINI_API_KEY=YOUR_KEY_HERE > .env

# 👉 Run Application
cd Code
streamlit run app.py
# Then open browser: http://localhost:8501
# Upload CSV → Ask SQL queries → View visualizations

# 👉 Evaluate NL → SQL Performance
cd Code
python evaluate_fossil_nl2sql.py
# This checks SQL accuracy using canonical SQL comparison
