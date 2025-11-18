import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5-small-nl2sql")

print(f"Loading model from: {MODEL_DIR}")

# -------------------------------------------------------------------
# Load model & tokenizer
# -------------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------------------------------------------
# SQL cleaning helper
# -------------------------------------------------------------------
def clean_sql(sql: str) -> str:
    """
    Simple post-processing:
    - strip spaces
    - keep only before first ';'
    - if it has SELECT ... FROM, dedupe columns in SELECT
    """
    if not sql:
        return sql

    sql = sql.strip()

    # keep only part before first ';'
    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"

    lower_sql = sql.lower()
    if not lower_sql.startswith("select"):
        return sql

    # try to split SELECT ... FROM ...
    if " from " in lower_sql:
        idx = lower_sql.index(" from ")
        select_part = sql[len("SELECT "):idx].strip()
        from_part = sql[idx:]  # includes ' from ...'

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

# -------------------------------------------------------------------
# Generation helper
# -------------------------------------------------------------------
def generate_sql(question: str) -> str:
    """Generate cleaned SQL from natural language question."""
    prefix = "translate English to SQL: "
    input_text = prefix + question

    inputs = tokenizer(
        input_text,
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

# -------------------------------------------------------------------
# Simple CLI loop
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔹 NL → SQL test console (type 'quit' to exit)\n")
    while True:
        q = input("Enter a natural language question: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break
        if not q:
            continue

        sql = generate_sql(q)
        print(f"👉 Generated SQL:\n{sql}\n")
