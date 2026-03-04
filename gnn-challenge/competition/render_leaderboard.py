"""
render_leaderboard.py

Render leaderboard.md from leaderboard/leaderboard.csv without external dependencies.
"""

import csv
from pathlib import Path

LEADERBOARD_CSV = Path("leaderboard/leaderboard.csv")
LEADERBOARD_MD = Path("leaderboard.md")

if not LEADERBOARD_CSV.exists():
    raise FileNotFoundError("leaderboard/leaderboard.csv not found")


def fmt_float(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


with LEADERBOARD_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

lines = [
    "# 🏆 GNN Challenge Leaderboard",
    "",
    "## Current Leaderboard",
    "",
    "| Rank | Team | Run | Model | Type | F1-Score | Accuracy | Precision | Recall | Date | Submitter |",
    "|------|------|-----|-------|------|----------|----------|-----------|--------|------|-----------|",
]

for row in rows:
    submitter = row.get("submitter", "")
    submitter_url = row.get("submitter_url", "")
    if submitter and submitter_url:
        submitter_md = f"[{submitter}]({submitter_url})"
    else:
        submitter_md = submitter

    lines.append(
        "| {rank} | {team} | {run_id} | {model} | {model_type} | {f1_score} | {accuracy} | {precision} | {recall} | {submission_date} | {submitter} |".format(
            rank=row.get("rank", ""),
            team=row.get("team", ""),
            run_id=row.get("run_id", ""),
            model=row.get("model", ""),
            model_type=row.get("model_type", ""),
            f1_score=fmt_float(row.get("f1_score", "")),
            accuracy=fmt_float(row.get("accuracy", "")),
            precision=fmt_float(row.get("precision", "")),
            recall=fmt_float(row.get("recall", "")),
            submission_date=row.get("submission_date", ""),
            submitter=submitter_md,
        )
    )

lines.extend([
    "",
    "## Notes",
    "- This leaderboard is auto-generated from `leaderboard/leaderboard.csv`.",
    "- Submissions must follow the `submissions/inbox/<team>/<run_id>/predictions.csv` format.",
])

LEADERBOARD_MD.write_text("\n".join(lines), encoding="utf-8")
