
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

# -----------------------------
# 1. Load data
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "../data")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))   # cfRNA
test_df  = pd.read_csv(os.path.join(DATA, "test.csv"))    # placenta
test_labels_path = os.path.join(DATA, "test_labels.csv")
test_labels = pd.read_csv(test_labels_path, index_col=0) if os.path.exists(test_labels_path) else None
edges_df = pd.read_csv(os.path.join(DATA, "graph_edges.csv"))
node_df  = pd.read_csv(os.path.join(DATA, "node_types.csv"))

# -----------------------------
# DROP ROWS WITHOUT TARGETS
# -----------------------------
missing_targets = train_df['disease_labels'].isna().sum()
print(f"{missing_targets} missing values in target column")

if missing_targets > 0:
    print("Dropping rows without target labels from training set...")
    train_df = train_df.dropna(subset=['disease_labels'])
    print(f"Training set now has {len(train_df)} samples with valid labels")

# Identify target column (train uses 'disease_labels' or 'target')
target_col = 'disease_labels' if 'disease_labels' in train_df.columns else 'target'
has_test_labels = test_labels is not None
# ========================================
# Display Target Distribution
# ========================================
print("\n" + "="*70)
print("  📊 TARGET FEATURE DISTRIBUTION")
print("="*70)

print("\n🔹 TRAINING DATA (cfRNA):")
train_counts = train_df['disease_labels'].value_counts().sort_index()
print(f"   Total samples: {len(train_df)}")
for target_val, count in train_counts.items():
    pct = (count / len(train_df)) * 100
    label = "control" if target_val == 0 else "preeclampsia"
    print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")

print("\n🔹 TESTING DATA (Placenta):")
print(f"   Total samples: {len(test_df)}")
if has_test_labels:
    test_counts = pd.Series(test_labels.iloc[:, 0]).value_counts().sort_index()
    for target_val, count in test_counts.items():
        pct = (count / len(test_df)) * 100
        label = "control" if target_val == 0 else "preeclampsia"
        print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")
else:
    print("   ⚠️  No labels (inductive task - labels hidden for evaluation)")

print("="*70 + "\n")

# -----------------------------
# 2. Node indexing
# -----------------------------
node_ids = node_df["node_id"].tolist()
node_map = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# -----------------------------
# 3. Graph construction
# -----------------------------
def build_graph(allowed_edge_types):
    data = HeteroData()
    data["node"].num_nodes = NUM_NODES

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data

USE_ANCESTRY_IN_TEST = False
train_graph = build_graph(["similarity"])
test_graph  = build_graph(["similarity", "ancestry"] if USE_ANCESTRY_IN_TEST else ["similarity"])

# -----------------------------
# 4. Node features
# -----------------------------
# Only use columns that exist in both train and test datasets
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
shared_cols = train_cols.intersection(test_cols)
feat_cols = [c for c in shared_cols if c not in ["node_id", target_col, "sample_id"]]
feat_cols = sorted(feat_cols)  # For consistent ordering

X = torch.zeros((NUM_NODES, len(feat_cols)))

train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
test_idx  = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)

X[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float)
X[test_idx]  = torch.tensor(test_df[feat_cols].values, dtype=torch.float)

train_graph["node"].x = X
test_graph["node"].x  = X

# -----------------------------
# 5. Labels (train only)
# -----------------------------
y = -1 * np.ones(NUM_NODES, dtype=int)  # default for all nodes
y[train_idx] = train_df[target_col].values.astype(int)
y = torch.tensor(y, dtype=torch.long)
print(f"✅ Labels assigned. Train nodes: {len(train_idx)}, Total nodes: {NUM_NODES}")
train_graph["node"].y = y
test_graph["node"].y = y

# -----------------------------
# 5b. Train/validation split
# -----------------------------
train_labels = train_df[target_col].values.astype(int)
train_idx_np = train_idx.cpu().numpy()
train_idx_split, val_idx_split = train_test_split(
    train_idx_np,
    test_size=0.2,
    random_state=42,
    stratify=train_labels
)
train_idx_split = torch.tensor(train_idx_split, dtype=torch.long)
val_idx_split = torch.tensor(val_idx_split, dtype=torch.long)

# -----------------------------
# 6. GraphSAGE model
# -----------------------------
class SAGEBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = SAGEConv(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c)
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return F.relu(x)

class GNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.l1 = SAGEBlock(in_c, hid_c)
        self.l2 = SAGEBlock(hid_c, hid_c)
        self.cls = SAGEConv(hid_c, out_c)
    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x, edge_index)
        return self.cls(x, edge_index)

num_classes = len(train_df[target_col].unique())
base_model = GNN(X.size(1), hid_c=64, out_c=num_classes)
model = to_hetero(base_model, train_graph.metadata(), aggr="mean")


# -----------------------------
# 7. Training setup with stronger class weights
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
y = y.to(device)

# Correct class weighting (inverse frequency)
num_samples = len(train_df)
class_counts = train_df[target_col].value_counts()

# Compute inverse-frequency weights
weights = torch.tensor([
    num_samples / (2 * class_counts.get(0, 1)),  # control
    num_samples / (2 * class_counts.get(1, 1))   # preeclampsia
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
print(f"Using class weights: {weights.cpu().numpy()}")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# -----------------------------
# 8. Training (Neighborhood Mini-Batch)
# -----------------------------
# Define number of neighbors to sample at each GraphSAGE layer (2 layers)
num_neighbors = {etype: [5, 5] for etype in train_graph.edge_types}  # adjust 10→other number if needed

train_loader = NeighborLoader(
    train_graph,
    input_nodes=("node", train_idx_split),  # only train split nodes
    num_neighbors=num_neighbors,      # neighbors per layer per edge type
    batch_size=16,                    # adjust based on GPU memory
    shuffle=True
)

print("Starting neighborhood mini-batch training...")
best_val_loss = float("inf")
best_state = None
patience = 50000
patience_left = patience
for epoch in range(1, 100001):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass on sampled subgraph
        out = model(batch.x_dict, batch.edge_index_dict)["node"]
        
        # Root nodes are the batch nodes we compute loss on
        root_nodes = torch.arange(batch.batch_size_dict["node"], device=device)
        batch_labels = batch["node"].y[root_nodes]

        loss = criterion(out[root_nodes], batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation on full graph (val nodes only)
    model.eval()
    with torch.no_grad():
        val_logits = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
        val_loss = criterion(val_logits[val_idx_split.to(device)], y[val_idx_split.to(device)]).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = patience
    else:
        patience_left -= 1

    if epoch % 1000 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
    if patience_left == 0:
        print(f"Early stopping at epoch {epoch:02d} (best val loss: {best_val_loss:.4f})")
        break

if best_state is not None:
    model.load_state_dict(best_state)




'''# -----------------------------
# 8. Training (FULL-BATCH, inductive-safe)
# -----------------------------
print("Starting training...")
for epoch in range(1, 31):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")'''

# -----------------------------
# 9. Inductive testing (placenta)
# -----------------------------
print("\nGenerating inductive predictions for placenta nodes...")
model.eval()
with torch.no_grad():
    logits = model(test_graph.x_dict, test_graph.edge_index_dict)["node"]
    preds = logits[test_idx].argmax(dim=1).cpu().numpy()
    proba = torch.softmax(logits[test_idx], dim=1).cpu().numpy()

# Evaluate on training set (to verify model performance)
print("\n" + "="*70)
print("  📊 TRAINING SET EVALUATION METRICS")
print("="*70)
with torch.no_grad():
    train_logits = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
    train_preds = train_logits[train_idx].argmax(dim=1).cpu().numpy()
    train_proba = torch.softmax(train_logits[train_idx], dim=1).cpu().numpy()
    train_true = y[train_idx].cpu().numpy()

train_acc = accuracy_score(train_true, train_preds)
train_prec = precision_score(train_true, train_preds, zero_division=0)
train_rec = recall_score(train_true, train_preds, zero_division=0)
train_f1 = f1_score(train_true, train_preds, zero_division=0)
train_cm = confusion_matrix(train_true, train_preds)

print(f"\n  Accuracy:     {train_acc:.4f}")
print(f"  Precision:    {train_prec:.4f}")
print(f"  Recall:       {train_rec:.4f}")
print(f"  F1-Score:     {train_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"     TN={train_cm[0,0]:3d}  FP={train_cm[0,1]:3d}")
print(f"     FN={train_cm[1,0]:3d}  TP={train_cm[1,1]:3d}")

# Test set prediction statistics
print("\n" + "="*70)
print("  🔮 TEST SET PREDICTIONS & EVALUATION (INDUCTIVE - PLACENTA)")
print("="*70)

pred_counts = np.bincount(preds, minlength=2)
print(f"\n📌 Predicted Labels for {len(preds)} Placenta Nodes:")
print(f"   Class 0 (control):       {pred_counts[0]:3d} nodes ({pred_counts[0]/len(preds)*100:.1f}%)")
print(f"   Class 1 (preeclampsia):  {pred_counts[1]:3d} nodes ({pred_counts[1]/len(preds)*100:.1f}%)")

print(f"\n📊 Prediction Confidence Analysis:")
max_conf = proba.max(axis=1)
print(f"   Mean max confidence: {max_conf.mean():.4f}")
print(f"   Min confidence:      {max_conf.min():.4f}")
print(f"   Max confidence:      {max_conf.max():.4f}")
print(f"   Std deviation:       {max_conf.std():.4f}")

# Count high confidence predictions
high_conf_mask = max_conf >= 0.9
print(f"\n   High confidence (≥0.90): {high_conf_mask.sum()} predictions ({high_conf_mask.sum()/len(preds)*100:.1f}%)")
med_conf_mask = (max_conf >= 0.7) & (max_conf < 0.9)
print(f"   Medium confidence (0.70-0.89): {med_conf_mask.sum()} predictions ({med_conf_mask.sum()/len(preds)*100:.1f}%)")
low_conf_mask = max_conf < 0.7
print(f"   Low confidence (<0.70): {low_conf_mask.sum()} predictions ({low_conf_mask.sum()/len(preds)*100:.1f}%)")

# Test set evaluation against true labels
if test_labels is not None:
    print("\n" + "="*70)
    print("  📊 TEST SET EVALUATION METRICS (Against True Labels)")
    print("="*70)

    # Get true labels for test set (aligned with test_df order)
    test_true = test_labels.iloc[:, 0].values.astype(int)

    test_acc = accuracy_score(test_true, preds)
    test_prec = precision_score(test_true, preds, zero_division=0)
    test_rec = recall_score(test_true, preds, zero_division=0)
    test_f1 = f1_score(test_true, preds, zero_division=0)
    test_cm = confusion_matrix(test_true, preds)

    print(f"\n  Accuracy:     {test_acc:.4f}")
    print(f"  Precision:    {test_prec:.4f}")
    print(f"  Recall:       {test_rec:.4f}")
    print(f"  F1-Score:     {test_f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"     TN={test_cm[0,0]:3d}  FP={test_cm[0,1]:3d}")
    print(f"     FN={test_cm[1,0]:3d}  TP={test_cm[1,1]:3d}")

    # Test set prediction breakdown by class
    print(f"\n📊 Prediction Breakdown by True Label:")
    print(f"   True Class 0 samples: {(test_true == 0).sum()} nodes")
    print(f"   True Class 1 samples: {(test_true == 1).sum()} nodes")

    # Breakdown by correct/incorrect predictions
    correct_mask = preds == test_true
    incorrect_mask = preds != test_true

    print(f"\n   Correct predictions: {correct_mask.sum()} ({correct_mask.sum()/len(preds)*100:.1f}%)")
    print(f"   Incorrect predictions: {incorrect_mask.sum()} ({incorrect_mask.sum()/len(preds)*100:.1f}%)")

    if correct_mask.sum() > 0:
        print(f"   Mean confidence (correct): {max_conf[correct_mask].mean():.4f}")
    if incorrect_mask.sum() > 0:
        print(f"   Mean confidence (incorrect): {max_conf[incorrect_mask].mean():.4f}")

    print("\n" + "="*70)
else:
    print("\n" + "="*70)
    print("  📊 TEST SET EVALUATION METRICS")
    print("="*70)
    print("  ⚠️  test_labels.csv not found. Skipping test-set evaluation.")
    print("\n" + "="*70)

# -----------------------------
# 10. Save predictions
# -----------------------------
os.makedirs("submissions", exist_ok=True)

# Hard predictions
submission_hard = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds
})
submission_hard.to_csv("submissions/advanced_gnn_preds.csv", index=False)

# Soft predictions with confidence
submission_soft = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds,
    "confidence_control": proba[:, 0],
    "confidence_preeclampsia": proba[:, 1]
})
submission_soft.to_csv("submissions/advanced_gnn_preds_with_confidence.csv", index=False)

print("✅ Predictions saved successfully!")
print(f"   Hard: submissions/advanced_gnn_preds.csv")
print(f"   Soft: submissions/advanced_gnn_preds_with_confidence.csv")
print(f"   Total predictions: {len(preds)}")
