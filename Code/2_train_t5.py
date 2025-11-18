import os
os.environ["USE_TF"] = "0"   # force transformers to use PyTorch, not TF

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "val.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5-small-nl2sql")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Our CSV columns are: id, question, sql
train_df = train_df.rename(columns={"question": "nl"})
val_df = val_df.rename(columns={"question": "nl"})

train_df = train_df[["nl", "sql"]]
val_df = val_df[["nl", "sql"]]

# -------------------------------------------------------------------
# 2. Tokenizer & Model
# -------------------------------------------------------------------
print("Loading tokenizer and model (t5-small)...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# -------------------------------------------------------------------
# 3. Preprocessing function
# -------------------------------------------------------------------
def preprocess_function(batch):
    inputs = ["translate English to SQL: " + q for q in batch["nl"]]
    model_inputs = tokenizer(
        inputs,
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["sql"],
            max_length=64,
            padding="max_length",
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# -------------------------------------------------------------------
# 4. Convert to HF datasets and tokenize
# -------------------------------------------------------------------
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["nl", "sql"],
)
val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["nl", "sql"],
)

# -------------------------------------------------------------------
# 5. Training arguments
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=30,              # more epochs for more data
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=5,
)

# -------------------------------------------------------------------
# 6. Trainer
# -------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
trainer.train()

# -------------------------------------------------------------------
# 7. Save model & tokenizer
# -------------------------------------------------------------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("\n✅ Training complete!")
print(f"Model saved to: {MODEL_DIR}")
