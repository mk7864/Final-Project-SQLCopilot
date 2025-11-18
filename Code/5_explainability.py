import os
import torch
from typing import List, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5-small-nl2sql")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "all", "show", "list",
    "give", "me", "with", "in", "on", "by", "from", "who", "which", "what",
    "that", "where", "when", "is", "are", "have", "has"
}


def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model


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
# X1 – Keyword overlap importance
# -----------------------------
def keyword_overlap_importance(question: str, sql: str) -> List[Tuple[str, float]]:
    sql_lower = sql.lower()
    tokens = question.split()
    scores = []

    for tok in tokens:
        t = tok.lower()
        if t in STOPWORDS:
            score = 0.1
        elif t in sql_lower:
            score = 1.0
        else:
            score = 0.3
        scores.append((tok, score))

    return scores


# -----------------------------
# X2 – Perturbation-based importance
# -----------------------------
def _seq_loss(model, tokenizer, question: str, target_sql: str) -> float:
    """Compute cross-entropy loss of target_sql given question."""
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to(DEVICE)

    labels = tokenizer(
        target_sql,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).input_ids.to(DEVICE)

    with torch.no_grad():
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        ).loss

    return float(loss.item())


def perturbation_importance(question: str, model, tokenizer) -> List[Tuple[str, float]]:
    """
    X2: LIME-style occlusion.
    For each token, remove it and see how much the sequence loss changes.
    Higher delta = more important token.
    """
    tokens = question.split()
    if len(tokens) <= 1:
        return [(tokens[0], 1.0)]

    # First get model's own predicted SQL
    base_sql = generate_sql(model, tokenizer, question)
    base_loss = _seq_loss(model, tokenizer, question, base_sql)

    importances = []
    for i in range(len(tokens)):
        perturbed_tokens = tokens[:i] + tokens[i + 1:]
        perturbed_q = " ".join(perturbed_tokens) if perturbed_tokens else question

        loss_i = _seq_loss(model, tokenizer, perturbed_q, base_sql)
        delta = max(0.0, loss_i - base_loss)  # higher delta -> more important
        importances.append((tokens[i], delta))

    # Normalise scores 0–1 for nicer visualisation
    max_val = max(score for _, score in importances) or 1.0
    norm_importances = [(tok, score / max_val) for tok, score in importances]

    return norm_importances


def explain_question(question: str):
    """Simple console helper to test explanations."""
    tokenizer, model = load_model()
    sql = generate_sql(model, tokenizer, question)
    print(f"\nQuestion: {question}")
    print(f"Predicted SQL: {sql}")

    print("\n[X1] Keyword-overlap importance:")
    for tok, score in keyword_overlap_importance(question, sql):
        print(f"  {tok:15s} -> {score:.2f}")

    print("\n[X2] Perturbation-based importance:")
    for tok, score in perturbation_importance(question, model, tokenizer):
        print(f"  {tok:15s} -> {score:.2f}")


if __name__ == "__main__":
    print("🔍 NL→SQL Explainability console (type 'quit' to exit)\n")
    tokenizer, model = load_model()

    while True:
        q = input("Enter a natural language question: ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        sql = generate_sql(model, tokenizer, q)

        print(f"\nPredicted SQL:\n{sql}\n")

        print("[X1] Keyword-overlap importance:")
        for tok, score in keyword_overlap_importance(q, sql):
            print(f"  {tok:15s} -> {score:.2f}")

        print("\n[X2] Perturbation-based importance:")
        for tok, score in perturbation_importance(q, model, tokenizer):
            print(f"  {tok:15s} -> {score:.2f}")

        print("\n" + "-" * 60 + "\n")
