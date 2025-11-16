# Final Project – SQLCopilot: Natural Language to SQL with T5 and Explainability

**Course:** NLP  
**Semester:** Fall 2025  
**Team:** Shaik Mohammad Mujahid Khalandar & Partner  

---

## 📌 Project Overview

SQLCopilot is an intelligent **Natural Language → SQL** assistant designed to help users query custom datasets **without writing SQL manually**.  
Users can upload **CSV or Excel** files, ask questions in English, and the system automatically converts the query to SQL, runs it on a **SQLite database**, and displays results with optional visualizations.

This project integrates:

- 🧠 **T5 Transformer (Seq2Seq) Model** fine‑tuned for NL→SQL generation  
- 🗄️ **SQLite backend** for executing queries safely  
- 📊 **Streamlit UI** for interaction, visualization, and usability  
- 🔍 **LIME Explainability** to show which input words influenced the output query  

This end‑to‑end pipeline demonstrates **data preparation, model training, evaluation, deployment, and interpretability**, satisfying course requirements.

---

## 📁 Repository Structure

```
Final-Project-SQLCopilot/
│
├─ Code/
│   ├─ app.py                       # Streamlit interface
│   ├─ sql.py                       # SQLite helper module
│   ├─ requirements.txt             # Dependencies
│   │
│   ├─ data/
│   │   └─ nl2sql_pairs.csv         # Training data for model
│   │
│   ├─ models/
│   │   └─ t5_nl2sql/               # Saved fine‑tuned model
│   │
│   ├─ 1_prepare_data.py            # Train/Val preprocessing
│   ├─ 2_train_t5.py                # Model training script
│   ├─ 3_evaluate_t5.py             # Metrics + BLEU + examples
│   └─ 4_explainability_lime.py     # Explainability module
│
├─ Final-Group-Project-Report/
│   └─ Final_SQLCopilot_Report.pdf
│
├─ Final-Group-Presentation/
│   └─ Final_SQLCopilot_Presentation.pdf
│
└─ README.md                        # Main documentation
```

---

## 🚀 Features

| Feature | Status |
|--------|---------|
| Upload CSV / Excel | ✅ |
| Auto‑load into SQLite | ✅ |
| English → SQL | 🚧 fine‑tuning |
| Query execution & results table | ✅ |
| Graph visualization (auto mode) | ✅ |
| Safety rules (SELECT‑only + LIMIT) | ✅ |
| Explainability via LIME | 🚧 integration |
| Model training pipeline | 🚧 ongoing |

---

## 🧩 Model Architecture

- **Base Architecture:** T5‑Small  
- **Task Format:**  
  *Input:* `"translate English to SQL: <question>"`  
  *Output:* `<SQL query>`

- **Training:** Supervised fine‑tuning on NL→SQL pairs  
- **Evaluation Metrics:** BLEU, Exact Match Accuracy  

---

## 🛠️ Installation

### 1️⃣ Clone repository

```bash
git clone https://github.com/<your-username>/Final-Project-SQLCopilot.git
cd Final-Project-SQLCopilot/Code
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 How To Run

### 🔹 Option A: Train model (optional if preloaded)

```bash
python 1_prepare_data.py
python 2_train_t5.py
python 3_evaluate_t5.py
```

### 🔹 Option B: Launch Streamlit App

```bash
streamlit run app.py
```

Then open browser:  
👉 http://localhost:8501/

---

## 📊 Visualization Options

- Automatic best‑fit charts  
- Bar, Histogram, or Category Counts  
- Works even without numeric‑only queries  

---

## 🔍 Explainability (Planned)

LIME will highlight which words affect:

- Query **structure**
- SQL **operators**
- **Column selection**

---

## 📌 Academic Requirements Coverage

| Requirement | Status |
|-------------|---------|
| Custom NLP task | ✔️ |
| Transformer model usage | ✔️ |
| Model training & evaluation | ✔️ |
| Visualization & UI | ✔️ |
| Interpretability | In progress |

---

## 👥 Team Roles

| Member | Responsibilities |
|--------|------------------|
| Shaik Mohammad Mujahid Khalandar | Coding, modeling, Streamlit UI, evaluation |
| Partner | Data expansion, explainability module, presentation |

---

## 📧 Contact

For questions or reproduction help:  
📩 smdkh@gwu.edu  

---

### ⭐ Future Enhancements

- Add semantic SQL validation  
- Add multi‑table JOIN reasoning  
- Deploy via Hugging Face Spaces or Streamlit Cloud  

---

**End of README — All rights reserved © 2025**
