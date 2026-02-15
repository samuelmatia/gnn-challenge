#!/usr/bin/env python3
"""
update_leaderboard.py

Scores submissions and updates leaderboard/leaderboard.csv (source of truth)
and regenerates leaderboard.md + docs/leaderboard.csv for GitHub Pages.

Public outputs intentionally contain team, rank, and final metrics.
"""

import csv
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SUBMISSIONS_DIR = Path(os.environ.get("SUBMISSIONS_DIR", "submissions/inbox"))
DATA_DIR = Path("data")
LEADERBOARD_CSV = Path("leaderboard/leaderboard.csv")
LEADERBOARD_MD = Path("leaderboard.md")
DOCS_LEADERBOARD_CSV = Path("docs/leaderboard.csv")

PUBLIC_COLUMNS = ["rank", "team", "f1_score", "accuracy", "precision", "recall"]
ORGANIZER_SUBMISSIONS = [
    Path("submissions/advanced_gnn_preds.csv"),
    Path("submissions/baseline_mlp_preds.csv"),
]


def load_private_labels() -> pd.Series:
    test_labels_path = DATA_DIR / "test_labels.csv"
    if not test_labels_path.exists():
        raise FileNotFoundError("data/test_labels.csv not found. Hidden labels are required for scoring.")

    test_labels = pd.read_csv(test_labels_path, index_col=0)
    if test_labels.shape[1] == 0:
        raise ValueError("test_labels.csv must have at least one column with labels.")

    return test_labels.iloc[:, 0].astype(int)


def _read_public_submission(pred_path: Path) -> pd.DataFrame:
    submission = pd.read_csv(pred_path)
    if "id" not in submission.columns or "y_pred" not in submission.columns:
        raise ValueError("Required columns: id,y_pred")
    return submission[["id", "y_pred"]].rename(columns={"id": "node_id"})


def _read_organizer_submission(pred_path: Path) -> pd.DataFrame:
    submission = pd.read_csv(pred_path)
    if "node_id" not in submission.columns or "target" not in submission.columns:
        raise ValueError("Required columns: node_id,target")
    return submission[["node_id", "target"]].rename(columns={"target": "y_pred"})


def _to_binary_preds(y_pred: pd.Series) -> pd.Series:
    if y_pred.dtype.kind in {"f", "c"}:
        return (y_pred.astype(float) >= 0.5).astype(int)
    return y_pred.astype(int)


def score_submission(submission: pd.DataFrame, test_true: pd.Series) -> dict:
    preds = _to_binary_preds(submission["y_pred"])
    if len(preds) != len(test_true):
        raise ValueError(f"Length mismatch. Expected {len(test_true)}, got {len(preds)}")

    return {
        "rank": 0,
        "team": "",
        "f1_score": f1_score(test_true, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(test_true, preds),
        "precision": precision_score(test_true, preds, zero_division=0),
        "recall": recall_score(test_true, preds, zero_division=0),
    }


def render_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# GNN Challenge Leaderboard",
        "",
        "## Current Leaderboard",
        "",
        "| Rank | Team | F1-Score | Accuracy | Precision | Recall |",
        "|------|------|----------|----------|-----------|--------|",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {rank} | {team} | {f1:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} |".format(
                rank=int(row["rank"]),
                team=row["team"],
                f1=row["f1_score"],
                acc=row["accuracy"],
                prec=row["precision"],
                rec=row["recall"],
            )
        )

    lines.extend([
        "",
        "## Notes",
        "- This leaderboard is auto-generated from `leaderboard/leaderboard.csv`.",
        "- Public leaderboard contains team (GitHub repo), rank, and final metrics only.",
    ])

    return "\n".join(lines)


def main() -> None:
    test_true = load_private_labels()
    rows = []

    for pred_path in SUBMISSIONS_DIR.glob("*/*/predictions.csv"):
        try:
            submission = _read_public_submission(pred_path)
            row = score_submission(submission, test_true)
            team = pred_path.parent.parent.name
            meta_path = pred_path.parent / "metadata.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    team = str(meta.get("team", team))
                except Exception:
                    pass
            row["team"] = team
            rows.append(row)
        except Exception as exc:
            print(f"Skipping {pred_path}: {exc}")

    for pred_path in ORGANIZER_SUBMISSIONS:
        if not pred_path.exists():
            continue
        try:
            submission = _read_organizer_submission(pred_path)
            row = score_submission(submission, test_true)
            row["team"] = pred_path.stem
            rows.append(row)
        except Exception as exc:
            print(f"Skipping {pred_path}: {exc}")

    if not rows:
        print("No valid submissions found.")
        return

    leaderboard = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    leaderboard = leaderboard.sort_values(by=["f1_score", "accuracy"], ascending=False, ignore_index=True)
    leaderboard["rank"] = range(1, len(leaderboard) + 1)

    LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    DOCS_LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(LEADERBOARD_CSV, index=False, columns=PUBLIC_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    leaderboard.to_csv(DOCS_LEADERBOARD_CSV, index=False, columns=PUBLIC_COLUMNS, quoting=csv.QUOTE_MINIMAL)

    LEADERBOARD_MD.write_text(render_markdown(leaderboard), encoding="utf-8")
    print(f"Leaderboard updated with {len(rows)} scored run(s).")


if __name__ == "__main__":
    main()
