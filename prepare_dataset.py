"""
prepare_dataset.py
Downloads b-mc2/sql-create-context from HuggingFace,
auto-labels each row by intent (from SQL answer),
saves to dataset/labeled_dataset.json

Run: python prepare_dataset.py
"""

import re
import json
import os
from collections import Counter
from datasets import load_dataset

SAVE_PATH = "dataset/labeled_dataset.json"
MAX_PER_CLASS = 500  # cap per intent class (balanced)

INTENT_LABELS = ["COUNT", "AVERAGE", "TOP_N", "BOTTOM_N", "FILTER_LT", "FILTER_GT", "SELECT"]


def label_from_sql(sql: str) -> str:
    """Infer intent label by parsing the SQL answer."""
    s = sql.upper().strip()

    if re.search(r'\bCOUNT\s*\(', s):
        return "COUNT"
    if re.search(r'\b(AVG|AVERAGE)\s*\(', s):
        return "AVERAGE"
    if re.search(r'ORDER\s+BY\s+\S+\s+DESC\s+LIMIT', s):
        return "TOP_N"
    if re.search(r'ORDER\s+BY\s+\S+\s+ASC\s+LIMIT', s):
        return "BOTTOM_N"
    if re.search(r'ORDER\s+BY\s+.*\bLIMIT\b', s):
        # no direction specified — default TOP_N
        return "TOP_N"
    if re.search(r'WHERE\s+.*\s*<\s*[\d\'"]', s):
        return "FILTER_LT"
    if re.search(r'WHERE\s+.*\s*>\s*[\d\'"]', s):
        return "FILTER_GT"
    return "SELECT"


def main():
    print("Loading dataset from HuggingFace (b-mc2/sql-create-context)...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    print(f"Total rows: {len(ds)}")

    # Label each row
    labeled = []
    for row in ds:
        question = row["question"].strip()
        sql      = row["answer"].strip()
        intent   = label_from_sql(sql)
        labeled.append({"text": question, "label": intent, "sql": sql})

    # Count distribution
    dist = Counter(item["label"] for item in labeled)
    print("\nRaw distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k:12s}: {v}")

    # Balance: cap each class at MAX_PER_CLASS
    balanced = []
    counts   = Counter()
    for item in labeled:
        if counts[item["label"]] < MAX_PER_CLASS:
            balanced.append(item)
            counts[item["label"]] += 1

    print(f"\nBalanced dataset: {len(balanced)} examples ({MAX_PER_CLASS} per class max)")
    for k, v in sorted(counts.items()):
        print(f"  {k:12s}: {v}")

    os.makedirs("dataset", exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(balanced, f, indent=2)

    print(f"\nSaved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
