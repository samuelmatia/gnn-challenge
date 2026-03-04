"""
baseline.py

Simple MLP baseline for cfRNA → placenta preeclampsia prediction

✔ Trains on gene expression features from cfRNA (train.csv)
✔ Predicts on placenta samples (test.csv)
✔ Evaluates against true test labels (test_labels.csv)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ========================================
# 1. Load data
# ========================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "../data")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA, "test.csv"))
test_labels = pd.read_csv(os.path.join(DATA, "test_labels.csv"), index_col=0)

print("="*70)
print("  🔧 BASELINE MLP MODEL")
print("="*70)

# ========================================
# 2. Extract features and labels
# ========================================
target_col = 'disease_labels'
feat_cols = [c for c in train_df.columns if c not in ["node_id", target_col, "sample_id"]]

X_train = train_df[feat_cols].values.astype(np.float32)
y_train = train_df[target_col].values.astype(np.int64)

X_test = test_df[feat_cols].values.astype(np.float32)
y_test = test_labels.iloc[:, 0].values.astype(np.int64)

print(f"\n📊 Data Summary:")
print(f"   Train features: {X_train.shape} (samples={X_train.shape[0]}, genes={X_train.shape[1]})")
print(f"   Test features:  {X_test.shape}")
print(f"   Train labels:   {len(y_train)} samples")
print(f"   Test labels:    {len(y_test)} samples")

# Display target distribution
print(f"\n🔹 Target Distribution (Training):")
train_counts = pd.Series(y_train).value_counts().sort_index()
for target_val, count in train_counts.items():
    label = "control" if target_val == 0 else "preeclampsia"
    pct = (count / len(y_train)) * 100
    print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")

print(f"\n🔹 Target Distribution (Testing):")
test_counts = pd.Series(y_test).value_counts().sort_index()
for target_val, count in test_counts.items():
    label = "control" if target_val == 0 else "preeclampsia"
    pct = (count / len(y_test)) * 100
    print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")

# ========================================
# 3. Create PyTorch datasets
# ========================================
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========================================
# 4. Define MLP model
# ========================================
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# ========================================
# 5. Initialize model and training setup
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n💻 Using device: {device}")

input_dim = X_train.shape[1]
model = MLPBaseline(input_dim=input_dim, hidden_dim=256, num_classes=2, dropout=0.5)
model = model.to(device)

# Compute class weights
class_counts = pd.Series(y_train).value_counts()
weights = torch.tensor([
    class_counts.get(1, 0) / len(y_train),
    class_counts.get(0, 0) / len(y_train)
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ========================================
# 6. Training loop
# ========================================
print(f"\n{'='*70}")
print("  🚀 TRAINING")
print(f"{'='*70}")

num_epochs = 50
best_val_acc = 0
patience = 10
patience_counter = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(y_batch)
    
    train_loss /= len(y_train)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f}")

print("✅ Training complete!")

# ========================================
# 7. Evaluation
# ========================================
print(f"\n{'='*70}")
print("  📊 EVALUATION METRICS")
print(f"{'='*70}")

model.eval()
all_preds = []
all_proba = []

with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        proba = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())

train_preds = np.array(all_preds)
train_proba = np.array(all_proba)

train_acc = accuracy_score(y_train, train_preds)
train_prec = precision_score(y_train, train_preds, zero_division=0)
train_rec = recall_score(y_train, train_preds, zero_division=0)
train_f1 = f1_score(y_train, train_preds, zero_division=0)
train_cm = confusion_matrix(y_train, train_preds)

print(f"\n🔹 TRAINING SET METRICS:")
print(f"   Accuracy:     {train_acc:.4f}")
print(f"   Precision:    {train_prec:.4f}")
print(f"   Recall:       {train_rec:.4f}")
print(f"   F1-Score:     {train_f1:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      TN={train_cm[0,0]:3d}  FP={train_cm[0,1]:3d}")
print(f"      FN={train_cm[1,0]:3d}  TP={train_cm[1,1]:3d}")

# Test set evaluation
all_test_preds = []
all_test_proba = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        proba = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_proba.extend(proba.cpu().numpy())

test_preds = np.array(all_test_preds)
test_proba = np.array(all_test_proba)

test_acc = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_rec = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)
test_cm = confusion_matrix(y_test, test_preds)

print(f"\n🔹 TEST SET METRICS:")
print(f"   Accuracy:     {test_acc:.4f}")
print(f"   Precision:    {test_prec:.4f}")
print(f"   Recall:       {test_rec:.4f}")
print(f"   F1-Score:     {test_f1:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      TN={test_cm[0,0]:3d}  FP={test_cm[0,1]:3d}")
print(f"      FN={test_cm[1,0]:3d}  TP={test_cm[1,1]:3d}")

# Prediction distribution
pred_counts = np.bincount(test_preds, minlength=2)
print(f"\n🔹 PREDICTION DISTRIBUTION (Test Set):")
print(f"   Class 0 (control):      {pred_counts[0]:3d} nodes ({pred_counts[0]/len(test_preds)*100:.1f}%)")
print(f"   Class 1 (preeclampsia): {pred_counts[1]:3d} nodes ({pred_counts[1]/len(test_preds)*100:.1f}%)")

# Confidence analysis
max_conf = test_proba.max(axis=1)
print(f"\n🔹 CONFIDENCE STATISTICS (Test Set):")
print(f"   Mean:        {max_conf.mean():.4f}")
print(f"   Min:         {max_conf.min():.4f}")
print(f"   Max:         {max_conf.max():.4f}")
print(f"   Std:         {max_conf.std():.4f}")

print("\n" + "="*70)

# ========================================
# 8. Save predictions
# ========================================
os.makedirs("submissions", exist_ok=True)

submission_hard = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": test_preds
})
submission_hard.to_csv("submissions/baseline_mlp_preds.csv", index=False)

submission_soft = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": test_preds,
    "confidence_control": test_proba[:, 0],
    "confidence_preeclampsia": test_proba[:, 1]
})
submission_soft.to_csv("submissions/baseline_mlp_preds_with_confidence.csv", index=False)

print("✅ Predictions saved to submissions/")
print(f"   - baseline_mlp_preds.csv")
print(f"   - baseline_mlp_preds_with_confidence.csv")