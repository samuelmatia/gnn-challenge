# Contributing Submissions

This repository accepts prediction artifacts only.
Participant training code is not executed here.


## Policy

- One submission attempt per participant is allowed.
- Enforcement is automatic in CI.

## ⚙️ Compute Budget

- Full training should not exceed **3 hours on CPU** per competition.
- If your pipeline is too slow, downsize graph complexity (for example fewer nodes/edges, smaller sampled neighborhoods, or lighter model depth).

## Quick Start

1. Fork the repository.
2. Clone your fork.
3. Create a branch.
4. Train your model externally using `data/public/` files.
5. Create submission files under `submissions/inbox/<team>/<run_id>/`.
6. Open a pull request to `main`.

## Required Submission Files

Folder layout:

```text
submissions/inbox/<team>/<run_id>/
```

Files:

- `predictions.csv.enc` (encrypted `predictions.csv` payload)
- `metadata.json`

Example:

```text
submissions/inbox/my_team/run_001/predictions.csv.enc
submissions/inbox/my_team/run_001/metadata.json
```

### Encryption (Required)

Create plaintext predictions locally first (do not commit plaintext):

```bash
python your_model.py  # writes predictions.csv with columns id,y_pred
```

Import organizer public key and encrypt:

```bash
gpg --import .github/keys/submission_public.asc
gpg --list-keys
CI_FPR=$(gpg --show-keys --with-colons .github/keys/submission_public.asc | awk -F: '/^fpr:/ {print $10; exit}')
gpg --output submissions/inbox/<team>/<run_id>/predictions.csv.enc \
  --encrypt --recipient "$CI_FPR" \
  predictions.csv
```

Security requirements:

- Do not commit plaintext `predictions.csv` to your PR.
- CI rejects plaintext prediction files.
- Only `predictions.csv.enc` is accepted for participant submissions.

Commit only encrypted artifact + metadata:

```bash
git add submissions/inbox/<team>/<run_id>/predictions.csv.enc
git add submissions/inbox/<team>/<run_id>/metadata.json
git status
git commit -m '<team name> commit'
git push origin <branch name>
```

Example metadata:

```json
{
  "team": "my_team",
  "run_id": "run_001",
  "model_name": "My GNN v1",
  "model_type": "human",
  "submitter": "my-github-username"
}
```

`submitter` must be your exact GitHub username and must match the PR author.

## 🧩 Working With Graph Data (`A` and `X`) in Your Codebase

Use:

- `data/public/adjacency_matrix.csv` as adjacency matrix `A`
- `data/public/train.csv` and `data/public/test.csv` as feature source for `X`
- `data/public/test_nodes.csv` as ID reference for predictions
- `data/public/graph_artifacts.pt` for prebuilt graph objects (recommended quick-start)

Minimal PyG-style conversion:

```python
import numpy as np
import pandas as pd
import torch

A = pd.read_csv("data/public/adjacency_matrix.csv", index_col=0).values
train = pd.read_csv("data/public/train.csv")
test = pd.read_csv("data/public/test.csv")

feat_cols = [c for c in train.columns if c not in ["node_id", "sample_id", "disease_labels"]]
all_nodes = pd.concat([train[["node_id"] + feat_cols], test[["node_id"] + feat_cols]], ignore_index=True)
all_nodes = all_nodes.drop_duplicates(subset=["node_id"], keep="last")

X = torch.tensor(all_nodes[feat_cols].values, dtype=torch.float32)
src, dst = np.where(A > 0)
edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
```

Important:

- Keep node ordering consistent between `A`, `X`, and node IDs.
- IDs inside your encrypted payload must match `data/public/test_nodes.csv`.

### Using Prebuilt Graph Artifacts (`graph_artifacts.pt`)

If you do not want to rebuild graphs manually, you can load ready-to-use artifacts:

```python
import torch

artifact = torch.load("data/public/graph_artifacts.pt")
train_graph = artifact["train_graph"]
test_graph = artifact["test_graph"]

# Optional helper tensors if present:
# train_idx = artifact.get("train_idx")
# test_idx = artifact.get("test_idx")
```

This is the fastest path for graph-based submissions because node features and connectivity are already aligned.

## Minimal Tabular Baseline

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data/public/train.csv")
test = pd.read_csv("data/public/test.csv")

X = train.drop(columns=["node_id", "sample_id", "disease_labels"], errors="ignore")
y = train["disease_labels"]
X_test = test.drop(columns=["node_id", "sample_id"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)

proba = clf.predict_proba(X_test_scaled)[:, 1]
pd.DataFrame({"id": test["node_id"], "y_pred": proba}).to_csv("predictions.csv", index=False)
```

Then encrypt `predictions.csv` to `predictions.csv.enc` before opening your PR.

## Validation Rules

- Decrypted `predictions.csv` must include `id` and `y_pred`.
- `y_pred` may be probability (`0-1`) or hard label (`0/1`).
- Row IDs must match `data/public/test_nodes.csv`.

## Evaluation and Leaderboard

- Primary metric: F1 Score
- Also reported: Accuracy, Precision, Recall
- Public leaderboard: `https://mubarraqqq.github.io/gnn-challenge/leaderboard.html`


## Maintainer Regeneration

To regenerate leaderboard artifacts consistently:

```bash
python update_leaderboard.py && python competition/render_leaderboard.py
```
