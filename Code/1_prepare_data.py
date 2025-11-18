import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "nl2sql_full.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

TRAIN_OUT = os.path.join(PROCESSED_DIR, "train.csv")
VAL_OUT = os.path.join(PROCESSED_DIR, "val.csv")

# -------------------------------------------------------------------
# 1. Load full NL→SQL dataset
# -------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

# Expect columns: id, question, sql
df = df[["id", "question", "sql"]].copy()

# Clean text (light)
df["question"] = df["question"].astype(str).str.strip()
df["sql"] = df["sql"].astype(str).str.strip()

# Drop any empty rows (just in case)
df = df[(df["question"] != "") & (df["sql"] != "")]
df = df.drop_duplicates(subset=["question", "sql"]).reset_index(drop=True)

print(f"Total samples after cleaning: {len(df)}")

# -------------------------------------------------------------------
# 2. Train / validation split
# -------------------------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# -------------------------------------------------------------------
# 3. Save processed CSVs
# -------------------------------------------------------------------
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print("\nData processed successfully!")
print(f"Train saved to: {TRAIN_OUT}")
print(f"Val saved to:   {VAL_OUT}")
