# Grouped sequential classifier: how to run

## 1. Offline study (no training, seconds)
Uses the saved probabilities in `results/hyser_refnorm`.

```bash
python scripts/sequential_offline.py          # -> reports/sequential_offline/*.csv, groups.json
python scripts/make_sequential_figures.py     # -> reports/sequential_offline/figures/
```

## 2. Trained version (needs the Hyser data again)
Each subject goes through: download (dynamic raw only), filter, z-score with session 1 statistics,
window, train the flat networks only if no `best.keras` exists (pilots S02, S06, S19, S20),
fit the group head, the within-group specialists and the labelled last-layer baseline,
then delete the raw files and windows. Keep the raw data outside OneDrive.

```bash
conda activate hyser_env                     # or your usual environment
brew install awscli                          # once, if `aws` is missing
python -m pytest -q tests/test_sequential.py # 16 tests, a few seconds

python scripts/run_hyser_sequential.py --subjects 01 --dry-run   # check the commands
python scripts/run_hyser_sequential.py --subjects 01             # one subject (minutes, quiet)
python scripts/run_hyser_sequential.py --subjects 01,03-05,07-18 # the 16 subjects with checkpoints
python scripts/run_hyser_sequential.py --subjects 02,06,19,20    # the 4 pilots: flat training first (slow, loud)
python scripts/evaluate_sequential_trained.py --latex-dir <path to ProgressReport>/generated
```

Re-running skips subjects that are finished. A failed subject is reported and the run continues.

## What is trained
* Frozen backbone: each flat network up to its last layer (features of size 128, 64 or 64).
* Group head: one linear softmax layer over the groups (default grouping D1-5, five groups learned from
  other subjects' session 1 confusions; H5 anatomical groups as comparison).
* Specialists: one linear softmax layer per group with more than one gesture.
* Labelled baseline: the last layer refitted with 1 or 2 labelled session 2 trials per gesture.
* Label-free calibration: per-channel statistics from 10 unlabelled session 2 trials spread over the session.

All heads together are a few thousand parameters, so the deployed cost is that of the flat network plus
one small dense layer per group.

## Outputs
* `results/hyser_seq/<subj>/seq_predictions.npz`, `seq_info.json`
* `reports/sequential_trained/summary.csv`, `ladder.csv`, `per_subject.csv`, `figures/fig_ladder.png`
* with `--latex-dir`: `tab_ladder.tex`, `tab_seq_trained.tex`, `numbers_seq_trained.tex` for the report
