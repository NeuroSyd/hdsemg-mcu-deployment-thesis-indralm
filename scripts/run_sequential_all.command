#!/bin/zsh -l
# Trained grouped sequential classifier + labelled baseline on all 20 Hyser subjects, then the report tables.
# Double-click in Finder, or run in Terminal:  zsh scripts/run_sequential_all.command
# Safe to stop and restart: finished subjects are skipped. The Mac is kept awake while it runs.
REPO="/Users/IndraMalik/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney Indra/Thesis Indra Lakshmana Malik 2026:2027/GitHub/hdsemg-mcu-deployment-thesis-indralm"
REPORT="/Users/IndraMalik/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney Indra/Thesis Indra Lakshmana Malik 2026:2027/Report/Thesis A/ProgressReport_ThesisA"
cd "$REPO" || { echo "Repo not found at $REPO"; exit 1; }
source ~/hyser_env/bin/activate || { echo "Could not activate ~/hyser_env"; exit 1; }
mkdir -p reports/sequential_trained
LOG=reports/sequential_trained/run.log
echo "=== started $(date)" | tee -a $LOG
python -c "import sys, tensorflow as tf, sklearn; print('python', sys.version.split()[0], 'tensorflow', tf.__version__, 'sklearn', sklearn.__version__)" 2>&1 | tee -a $LOG

echo "--- unit tests" | tee -a $LOG
python -m pytest -q tests/test_sequential.py 2>&1 | tee -a $LOG
[ ${pipestatus[1]} -eq 0 ] || { echo "Tests failed, stopping. Send the log to Claude."; exit 1; }

echo "--- subject 01 first (check)" | tee -a $LOG
caffeinate -i python scripts/run_hyser_sequential.py --subjects 01 --work-root "$HOME/hyser_data/seq_work" 2>&1 | tee -a $LOG
[ -f results/hyser_seq/01/seq_predictions.npz ] || { echo "Subject 01 failed, stopping. Send $LOG to Claude."; exit 1; }

echo "--- the 15 other subjects with saved networks (quiet)" | tee -a $LOG
caffeinate -i python scripts/run_hyser_sequential.py --subjects 03-05,07-18 --work-root "$HOME/hyser_data/seq_work" 2>&1 | tee -a $LOG

echo "--- pilots S02, S06, S19, S20: flat networks are trained first (slow, loud)" | tee -a $LOG
caffeinate -i python scripts/run_hyser_sequential.py --subjects 02,06,19,20 --work-root "$HOME/hyser_data/seq_work" 2>&1 | tee -a $LOG

echo "--- results and report tables" | tee -a $LOG
if [ -d "$REPORT" ]; then
  python scripts/evaluate_sequential_trained.py --latex-dir "$REPORT/generated" 2>&1 | tee -a $LOG
  if command -v latexmk >/dev/null; then (cd "$REPORT" && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 && echo "Report recompiled: $REPORT/main.pdf" | tee -a $LOG); fi
else
  python scripts/evaluate_sequential_trained.py 2>&1 | tee -a $LOG
  echo "Report folder not found; tables are in reports/sequential_trained/" | tee -a $LOG
fi
echo "=== finished $(date)" | tee -a $LOG
echo; echo "Done. You can close this window."
