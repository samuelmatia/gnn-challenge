#!/usr/bin/env python3
"""
validate_submission.py

Validate submission format for predictions.csv against test.csv.
"""

import sys
import pandas as pd
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python competition/validate_submission.py <predictions.csv> [test.csv]")
        return 1

    pred_path = Path(sys.argv[1])
    default_test_nodes = Path("data/public/test_nodes.csv")
    default_test = Path("data/test.csv")
    test_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (default_test_nodes if default_test_nodes.exists() else default_test)

    if not pred_path.exists():
        print(f"❌ Submission file not found: {pred_path}")
        return 1
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return 1

    preds = pd.read_csv(pred_path)
    test_df = pd.read_csv(test_path)

    if "id" not in preds.columns or "y_pred" not in preds.columns:
        print("❌ predictions.csv must contain columns: id, y_pred")
        return 1

    if len(preds) != len(test_df):
        print(f"❌ Row count mismatch. Expected {len(test_df)}, got {len(preds)}")
        return 1

    if "id" in test_df.columns:
        test_ids = test_df["id"].astype(str).tolist()
    elif "node_id" in test_df.columns:
        test_ids = test_df["node_id"].astype(str).tolist()
    else:
        print("❌ Test file must contain either 'id' or 'node_id' column")
        return 1
    pred_ids = preds["id"].astype(str).tolist()

    if set(pred_ids) != set(test_ids):
        print("❌ IDs in predictions.csv do not match test set IDs")
        return 1

    print("✅ Submission format is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
