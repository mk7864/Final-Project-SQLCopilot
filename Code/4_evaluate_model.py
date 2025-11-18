import os
import pandas as pd
import numpy as np
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# -----------------------------
# Paths & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "val.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5-small-nl2sql")

OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "val_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Helper functions
# -----------------------------
def normalize_sql(s: str) -> str:
    """Simple normalisation: lowercase, collapse whitespace."""
    return " ".join(str(s).strip().lower().split())


def generate_sql(model, tokenizer, question: str, max_input_len: int = 64, max_output_len: int = 128) -> str:
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,
            early_stopping=True,
        )

    sql = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql.strip()


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(f"Validation file not found at {VAL_PATH}")

    print(f"Loading val set from: {VAL_PATH}")
    df_val = pd.read_csv(VAL_PATH)

    if "question" not in df_val.columns or "sql" not in df_val.columns:
        raise ValueError("val.csv must have at least 'question' and 'sql' columns.")

    questions = df_val["question"].astype(str).tolist()
    gold_sqls = df_val["sql"].astype(str).tolist()

    print(f"Loaded {len(df_val)} validation examples.")

    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    preds = []
    exact_matches = []

    # For BLEU
    refs_tokenized = []   # list of list-of-references (each reference is list of tokens)
    hyps_tokenized = []   # list of hypothesis token lists

    # For ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = []

    for q, gold in zip(questions, gold_sqls):
        pred = generate_sql(model, tokenizer, q)

        preds.append(pred)

        gold_norm = normalize_sql(gold)
        pred_norm = normalize_sql(pred)

        exact = 1 if gold_norm == pred_norm else 0
        exact_matches.append(exact)

        # BLEU tokenisation
        refs_tokenized.append([gold_norm.split()])
        hyps_tokenized.append(pred_norm.split())

        # ROUGE-L
        r = rouge_scorer_obj.score(gold_norm, pred_norm)
        rougeL_scores.append(r["rougeL"].fmeasure)

    # -----------------------------
    # Metrics
    # -----------------------------
    # Exact Match
    em_score = float(np.mean(exact_matches))

    # BLEU
    smoothing = SmoothingFunction().method1
    bleu_score_val = corpus_bleu(
        refs_tokenized,
        hyps_tokenized,
        smoothing_function=smoothing,
    )

    # ROUGE-L (average F1)
    rougeL_mean = float(np.mean(rougeL_scores))

    print("\n=== Evaluation Results on val.csv ===")
    print(f"Exact Match: {em_score:.4f}")
    print(f"BLEU:        {bleu_score_val:.4f}")
    print(f"ROUGE-L F1:  {rougeL_mean:.4f}")

    # Save per-example predictions
    df_out = df_val.copy()
    df_out["predicted_sql"] = preds
    df_out["exact_match"] = exact_matches
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved detailed predictions to:\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
