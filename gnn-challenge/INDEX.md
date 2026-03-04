# 🎯 GNN Challenge - Complete Implementation Guide

Welcome to the automated submission + leaderboard system for the GNN Challenge.

## 📚 Documentation Index

### 🚀 Getting Started
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Quick reference after setup
- **[CHECKLIST.md](CHECKLIST.md)** - Verify all components are working
- **[test_submission_infrastructure.py](test_submission_infrastructure.py)** - Run tests

### 👥 For Participants
- **[CONTRIBUTING.md](CONTRIBUTING.md)** ⭐ **START HERE**
  - Submission format
  - Example code
  - PR process

- **[submissions/inbox/README.md](submissions/inbox/README.md)**
  - Folder structure
  - Required files

### 🏆 Leaderboard
- **[leaderboard.md](leaderboard.md)** - Auto-generated static view
- **[leaderboard/leaderboard.csv](leaderboard/leaderboard.csv)** - Source of truth
- **[docs/leaderboard.html](docs/leaderboard.html)** - Interactive leaderboard (GitHub Pages)

### 🔧 For Organizers
- **[SUBMISSION_SETUP.md](SUBMISSION_SETUP.md)** - Workflow + architecture details

## 🎯 Quick Links

| Need | Document | Purpose |
|------|----------|---------|
| **Submit your model** | [CONTRIBUTING.md](CONTRIBUTING.md) | Submission guide |
| **Check standings** | [leaderboard.md](leaderboard.md) | Static leaderboard |
| **Interactive board** | `docs/leaderboard.html` | GitHub Pages view |
| **File format help** | [submissions/inbox/README.md](submissions/inbox/README.md) | Submission format |
| **System details** | [SUBMISSION_SETUP.md](SUBMISSION_SETUP.md) | Architecture |
| **Verify setup** | [CHECKLIST.md](CHECKLIST.md) | Validation steps |

## 🚀 System Overview

```
Participant PR
   ↓
CI validates submission format
   ↓
CI scores predictions (hidden labels)
   ↓
PR comment with scores
   ↓
On merge: leaderboard/leaderboard.csv updated
   ↓
leaderboard.md + docs/leaderboard.csv regenerated
```

## 🎓 How to Participate (Short)

1. Train model on `data/train.csv`
2. Generate predictions for `data/test.csv`
3. Save:
   - `submissions/inbox/<team>/<run_id>/predictions.csv` (`id`, `y_pred`)
   - `submissions/inbox/<team>/<run_id>/metadata.json`
4. Open PR → CI scores → leaderboard updates on merge

See [CONTRIBUTING.md](CONTRIBUTING.md) for full instructions.
