"""
render_leaderboard.py

Render leaderboard.md from leaderboard/leaderboard.csv.
Public output contains team, rank, and final metrics.
"""

from pathlib import Path

import pandas as pd

LEADERBOARD_CSV = Path("leaderboard/leaderboard.csv")
LEADERBOARD_MD = Path("leaderboard.md")

if not LEADERBOARD_CSV.exists():
    raise FileNotFoundError("leaderboard/leaderboard.csv not found")

leaderboard = pd.read_csv(LEADERBOARD_CSV)

lines = [
    "# GNN Challenge Leaderboard",
    "",
    "## Current Leaderboard",
    "",
    "| Rank | Team | F1-Score | Accuracy | Precision | Recall |",
    "|------|------|----------|----------|-----------|--------|",
]

for _, row in leaderboard.iterrows():
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

LEADERBOARD_MD.write_text("\n".join(lines), encoding="utf-8")
