# Contributing Submissions to GNN Challenge

This [repository](https://github.com/Mubarraqqq/gnn-challenge) accepts **prediction files only**. No participant code is executed.

## Quick Start (External Participants)

1) **Fork** the repo on GitHub.  
2) **Clone** your fork:
```bash
git clone https://github.com/<their-username>/gnn-challenge.git
cd gnn-challenge
```
3) **Create a branch**:
```bash
git checkout -b new_submission
```


---
## Minimal ML Baseline (Quick Start) for Graph Based models

If you just want a working baseline, here’s a tiny logistic‑regression example:

```python

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import to_hetero

# load artifact
artifact = torch.load("data/public/graph_artifacts.pt")
train_graph = artifact["train_graph"]
test_graph = artifact["test_graph"]

# simple GraphSAGE backbone
class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# build hetero model
in_dim = train_graph["node"].x.size(1)
model = to_hetero(SAGE(in_dim, 64, 2), train_graph.metadata(), aggr="mean")

# train step example
out = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
y = train_graph["node"].y
train_mask = y >= 0  # only labeled train nodes
loss = F.cross_entropy(out[train_mask], y[train_mask])
loss.backward()


```

---

---
## Minimal ML Baseline (Quick Start) for tabular Based models

If you just want a working baseline, here’s a tiny logistic‑regression example:

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

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

proba = model.predict_proba(X_test_scaled)[:, 1]
preds = pd.DataFrame({"id": test["node_id"], "y_pred": proba})
preds.to_csv("predictions.csv", index=False)

```

---

4) **Train your model** using `data/public/train.csv` and generate predictions for `data/public/test.csv`.
   - Create `predictions.csv` with columns: `id`, `y_pred`.
   - IDs **must** match `data/public/test_nodes.csv`.

5) **Create submission folder**:
```
submissions/inbox/<team>/<run_id>/predictions.csv
submissions/inbox/<team>/<run_id>/metadata.json
```

Example `metadata.json`:
```json
{
  "team": "my_team",
  "run_id": "run_001",
  "model_name": "My GNN v1",
  "model_type": "human"
}
```

6) **Commit + push**:
```bash
git add submissions/inbox/<team>/<run_id>/predictions.csv
git add submissions/inbox/<team>/<run_id>/metadata.json
git commit -m "Add submission: My GNN v1"
git push origin new_submission
```

7) **Open a PR** to the main repo. CI validates + scores. On merge, the leaderboard updates.

---

## Submission Format

Required:
- `predictions.csv` with columns `id`, `y_pred`
- `metadata.json`

Notes:
- `y_pred` can be probability (0–1) or hard label (0/1)
- IDs must match `data/public/test_nodes.csv`


## Leaderboard

Your submission can be viewed [here](https://mubarraqqq.github.io/gnn-challenge/leaderboard.html) after PR has been merged by the [organizer](www.github.com/mubarraqqq)

---

## Evaluation Metrics

Primary metric: **Macro F1**  
Also reported: Accuracy, Precision, Recall.

---

## FAQ

**Do I need to run any code in this repo?**  
No. You only submit prediction files.

**Who merges PRs?**  
Maintainers. The leaderboard updates after merge.

**Can I submit multiple runs?**  
No. Only one submission attempt per participant is allowed and enforced in CI.
