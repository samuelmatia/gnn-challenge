# Submission & Leaderboard Setup

**This project uses automated scoring only.** Submissions are evaluated in CI against a hidden test set and the leaderboard updates on merge.

## Automated Scoring with GitHub Actions

### How It Works

1. **Participant** submits a PR with:
   - `submissions/inbox/<team>/<run_id>/predictions.csv`
   - `submissions/inbox/<team>/<run_id>/metadata.json`
2. **GitHub Actions** automatically:
   - Validates the submission format
   - Scores predictions using hidden test labels
   - Posts results as a PR comment
   - Updates the leaderboard on merge

### Files Used

- `.github/workflows/score-submission.yml` — CI workflow
- `competition/validate_submission.py` — submission format validation
- `scoring_script.py` — scoring utility
- `update_leaderboard.py` — updates leaderboard CSV + markdown
- `leaderboard/leaderboard.csv` — source of truth
- `docs/leaderboard.html` — interactive leaderboard (GitHub Pages)

### Workflow Trigger

```yaml
on:
  pull_request_target:
    paths:
      - 'submissions/inbox/**'
```

### Leaderboard Architecture

- **Source of truth**: `leaderboard/leaderboard.csv`
- **Static view**: `leaderboard.md` (auto-generated)
- **Interactive view**: `docs/leaderboard.html` (GitHub Pages)

`update_leaderboard.py`:
1. Scores new submissions
2. Appends rows to `leaderboard/leaderboard.csv`
3. Recomputes ranks
4. Regenerates `leaderboard.md`
5. Copies CSV to `docs/leaderboard.csv`

### Enabling GitHub Pages (Interactive Leaderboard)

1. Settings → Pages
2. Source: `main` branch, `/docs` folder
3. Save

Your leaderboard will be available at:
```
https://<org>.github.io/<repo>/leaderboard.html
```

---

## ✅ Checklist

- [x] Create `leaderboard/leaderboard.csv`
- [x] Add `docs/leaderboard.html`, `docs/leaderboard.css`, `docs/leaderboard.js`
- [x] Set up `.github/workflows/score-submission.yml`
- [x] Create `competition/validate_submission.py`
- [ ] Enable GitHub Actions
- [ ] Enable GitHub Pages
- [ ] Test workflow with a sample PR
