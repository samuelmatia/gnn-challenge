# 🧬 GLIMPS-GNN
**Graph-based Liquid-biopsy Inductive Modeling for PreeclampSia**

## GNN Challenge: cfRNA → Placenta Inductive Prediction

<div align="center">
    <img src="images/IMG3.png" width='640' /> 
</div>

<br>
<br>


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?logo=github)](LICENSE)
[![License: CC-BY-SA-4.0](https://img.shields.io/badge/License-CC--BY--SA--4.0-green.svg?logo=github)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-blue?logo=googlechrome)](https://mubarraqqq.github.io/gnn-challenge/leaderboard.html)

This repository hosts a prediction-only challenge focused on maternal-fetal health modeling using graph learning.
Participant code is run outside this repository. Submissions are scored in CI against hidden labels.

**[🏆 Click me to join competition](https://github.com/Mubarraqqq/gnn-challenge/blob/main/CONTRIBUTING.md)** 
## Scientific Focus

- Inductive graph learning across cfRNA and placental transcriptomics to detect maternal-fetal health issues.
- Learn transferable representations that generalize to unseen samples and domains rather than treating each dataset independently.

## Alignment with [BASIRA Lab's](https://basira-lab.com) Mission

- Prioritizes robust generalization across heterogeneous datasets.
- Uses compute-efficient, non-data-hungry graph learning methods that can run on standard hardware.

## Inspiration from GNN Literature

- Draws from studies on inductive learning, message passing, and representation transfer.
- Model design follows [DGL Lectures 1.1-4.6](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T), covering:
  - Graph construction from tabular data
  - Node feature encoding
  - Neighborhood aggregation (GraphSAGE-style inductive updates)
  - Mini-batch training via neighborhood sampling
  - Inductive inference on unseen nodes

## Overview

- Task: Binary classification (`0=Control`, `1=Preeclampsia`)
- Setting: Inductive transfer from cfRNA (train) to placenta (test)
- Primary metric: F1 Score
- Additional metrics: Accuracy, Precision, Recall
- Public leaderboard: Auto-updated after merged submissions

<div align="center">
    <img src="images/IMG4.jpeg" width='750' /> 
</div>

<br>
<br>

## Dataset Source and Description

### Source

- Public datasets from Gene Expression Omnibus (GEO, NIH)
- Maternal plasma cfRNA: `GSE192902`
- Placental RNA-seq: `GSE234729`

### Data Splits

- Training set: cfRNA samples
- Test set: placenta samples (unseen during training)
- Labels: binary disease status

### Purpose and Integration Goal

- Identify and validate cfRNA biomarkers for early prediction of preeclampsia, often before clinical symptoms appear.
- Support research in maternal-fetal health and early detection of preeclampsia.
- Integrate gene expression and clinical metadata to capture subtle risk patterns while handling noisy and imbalanced data for robust and equitable predictions.

### 🧩 Mandatory Graph Specification

This competition explicitly provides both required graph components:

- Adjacency matrix `A`: `data/public/adjacency_matrix.csv`
- Node feature matrix `X`: derived from `data/public/train.csv` and `data/public/test.csv`

Related graph files:

- `data/public/graph_edges.csv`
- `data/public/node_types.csv`
- `data/public/graph_artifacts.pt`

Interpretation:

- `A[i, j] = 1` indicates an edge between nodes `i` and `j`, else `0`
- `X` is node-by-feature and includes harmonized expression features and released covariates
- Node alignment is by `node_id`; use `data/public/test_nodes.csv` (and node files) as the ordering reference so rows in `X` correspond to the same nodes indexed in `A`.

### 🌍 Dataset Difficulty and Realism

The benchmark includes meaningful modeling difficulty:

- 🧪 Noisy and partially missing metadata
- ⚖️ Label imbalance pressure
- 🧬 High-dimensional features relative to sample size (sparsity pressure)
- 🔄 Cross-domain distribution shift (cfRNA -> placenta)
- 🕸️ Inductive generalization to unseen test nodes

### ⏱️ Computational Affordability

- Full training should not exceed **3 hours on CPU** per competition.
- If needed, downsize graph complexity (for example by reducing node count, edge density, or neighborhood sampling size) while preserving task integrity.

## Dataset Construction and Preprocessing

[`build_dataset.ipynb`](./organizer_scripts/build_dataset.ipynb) and [Kaggle](https://www.kaggle.com/code/freeeman/maternal-2014425c3f4)

Objective: Ensure structural compatibility for graph construction and inductive learning by handling expression data, parsing and cleaning metadata, and expression-metadata fusion.

## Advanced GNN Implementation

[`advanced_GNN_model.py`](./starter_code/advanced_GNN_model.py)

Objective: Implement an advanced inductive GNN for cfRNA -> placenta prediction, ensuring generalizable node representations and inductive learning.

Key Components:

- Graph Construction: Build hetero-graphs using similarity and ancestry edges.
- Node Feature Encoding: Integrate gene expression and metadata into node-level features.
- Neighborhood Aggregation: GraphSAGE-style layers with BatchNorm and ReLU for neighbor information propagation.
- Mini-Batch Training: Use neighborhood sampling for efficient training on large graphs.
- Inductive Inference: Generate predictions for unseen placenta nodes without label leakage.

## Starter Assets

- `starter_code/advanced_GNN_model.py`
- `starter_code/baseline.py`
- `starter_code/build_adjacency_matrix.py`
- `starter_code/build_graph_artifacts.py`

## Submission Policy

Submission instructions are in `CONTRIBUTING.md`.

Key policy:

- Only one submission attempt per participant (enforced in CI)
- This public repository is **not privacy-preserving**: merged submission artifacts and leaderboard metadata are publicly visible.

## Leaderboard

- Public page: `https://mubarraqqq.github.io/gnn-challenge/leaderboard.html`
- Source CSV: `leaderboard/leaderboard.csv`
- Rendered markdown: `leaderboard.md`
- Tie handling: equal scores share rank

## Maintainer Regeneration Command

Use this command to regenerate all leaderboard outputs from the canonical pipeline:

```bash
python update_leaderboard.py && python competition/render_leaderboard.py
```

## Citation

```bibtex
@dataset{gnn_challenge_2026,
  title={GNN Challenge: cfRNA -> Placenta Inductive GNN for Maternal-Fetal Health Prediction},
  author={Mubaraq Onipede},
  year={2026},
  url={https://github.com/Mubarraqqq/gnn-challenge}
}
```

## License

See `LICENSE`.
