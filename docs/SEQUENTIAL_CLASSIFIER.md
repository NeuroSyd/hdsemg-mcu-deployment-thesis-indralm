# Grouped sequential classifier: how to run

Everything runs in the project virtual environment:

```bash
source ~/hyser_env/bin/activate
cd "<repo folder>"          # hdsemg-mcu-deployment-thesis-indralm
```

## Easiest: one script for everything
Double-click `scripts/run_sequential_all.command` in Finder, or in Terminal:

```bash
zsh scripts/run_sequential_all.command
```

It activates `~/hyser_env`, runs the unit tests, runs subject 01 as a check (and stops if that fails),
then the 15 other subjects with saved networks, then the 4 pilots (S02, S06, S19, S20, whose flat
networks are trained first: the slow, loud part), and finally writes the report tables into
`Report/Thesis A/ProgressReport_ThesisA/generated` and recompiles the report if `latexmk` exists.
The Mac is kept awake with `caffeinate`. Everything is logged to `reports/sequential_trained/run.log`.
Stopping and restarting is safe: finished subjects are skipped.

## Step by step (same thing, by hand)
```bash
python -m pytest -q tests/test_sequential.py                       # 16 tests, seconds
python scripts/run_hyser_sequential.py --subjects 01 --dry-run       # print the commands only
python scripts/run_hyser_sequential.py --subjects 01                 # one subject, about 10 to 15 min
python scripts/run_hyser_sequential.py --subjects 03-05,07-18        # subjects with saved networks
python scripts/run_hyser_sequential.py --subjects 02,06,19,20        # pilots: trains the flat networks first
python scripts/evaluate_sequential_trained.py --latex-dir "<report folder>/generated"
```

Each subject is downloaded (dynamic raw only, both sessions), filtered, z-scored with session 1
statistics, windowed, scored, and then the raw files and windows are deleted. Keep the raw data
outside OneDrive (default `~/hyser_data/hyser`).

## Offline study (no data needed, seconds)
```bash
python scripts/sequential_offline.py
python scripts/make_sequential_figures.py
```

## What is trained
* Frozen backbone: each flat network up to its last layer (features of size 128, 64 or 64).
* Group head: one linear softmax layer over the groups (default D1-5: five groups learned from the
  other subjects' session 1 confusions; H5 anatomical groups as comparison).
* Specialists: one linear softmax layer per group with more than one gesture.
* Labelled baseline: the last layer refitted with 1 or 2 labelled session 2 trials per gesture.
* Label-free update: per-channel statistics from 10 unlabelled session 2 trials spread over the session.
All heads together are a few thousand parameters. No network weight is changed.

## Outputs
* `results/hyser_seq/<subject>/seq_predictions.npz`, `seq_info.json`
* `reports/sequential_trained/summary.csv`, `ladder.csv`, `per_subject.csv`, `figures/fig_ladder.png`, `run.log`
* with `--latex-dir`: `tab_ladder.tex`, `tab_seq_trained.tex`, `numbers_seq_trained.tex`, `fig_ladder.png`;
  Section 3.4.6 of the report shows them automatically after recompiling.
