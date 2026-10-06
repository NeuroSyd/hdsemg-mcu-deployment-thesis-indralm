# hdsemg-mcu-deployment-thesis-indralm

Final-year thesis: HD-sEMG hand gesture/grasp classification with deep learning, compressed and deployed on microcontroller hardware (Hyser + CEMHSEY datasets).

## Overview

End-to-end pipeline from raw HD-sEMG extraction through to on-device deployment, benchmarking three deep architectures (1D CNN, GRU, TCN) against a staged compression trajectory and real microcontroller hardware constraints.

- **Datasets:** Hyser (PhysioNet, 256 ch) and CEMHSEY (Zenodo, 320 ch, GRASP + GESTURE sub-tasks)
- **Architectures:** 1D CNN, GRU, TCN, all sharing one raw-signal input tensor, no hand-crafted features (LDA baseline uses classical features separately)
- **Compression:** post-training quantisation → quantisation-aware training → pruning, with accuracy/size/latency/power tracked at every stage
- **Deployment target:** Arm Cortex-M, TensorFlow Lite for Microcontrollers + CMSIS-NN (fallback: X-CUBE-AI)

## Pipeline stages

| Stage | Scope | Output |
|---|---|---|
| 0. Extraction | Direct pull from PhysioNet (Hyser) and Zenodo (CEMHSEY), no mirrors | Raw WFDB / .mat files + manifest |
| 1. Preprocessing | Shared 10–500 Hz Butterworth bandpass + 50 Hz notch; per-subject-per-session z-score normalisation | Filtered, normalised signal matrices |
| 2. Windowing & features | 200 ms windows, 50% overlap (matches CEMHSEY's own LDA benchmark protocol) | Windowed tensors (deep models) + classical feature vectors (LDA baseline) |
| 3. Training | Subject-dependent split primary; per-window and per-trial (majority vote) accuracy reported | Trained checkpoints per architecture per dataset |
| 4. Compression | PTQ → QAT → pruning, applied independently to all three architectures; the winner is chosen after compression on accuracy-vs-efficiency | Compressed checkpoints + accuracy retention curve per architecture |
| 5. MCU deployment | Accuracy, latency, model size, power measured jointly on one board | Full on-device benchmark |
| 6. Robustness | Cross-session/cross-day evaluation + simulated channel dropout | Degradation curves vs. day gap and vs. channel loss |

## Running Stages 0 to 3 on one subject (proof run)

Scope is set in the configs (`subjects`, `sessions`, `task_types` in `configs/preprocessing.yaml`; `data:` in `configs/training.yaml`), so widening to more subjects is a config change, not a code change.

```bash
python src/extraction/extract_hyser.py --out data/raw/hyser --subjects 01 --workers 12   # Stage 0 (parallel, resumable, SHA256-checked)
python src/preprocessing/pipeline.py --config configs/preprocessing.yaml                  # Stage 1
python src/windowing/pipeline.py     --config configs/windowing.yaml                      # Stage 2a (attaches gesture labels)
python src/training/train_lda.py     --config configs/training.yaml                       # Stage 3 LDA baseline
python src/training/train.py         --config configs/training.yaml --architecture cnn1d  # Stage 3 (also: tcn, gru)
python src/evaluation/progress_figures.py --subject 01                                    # figures + summary tables -> reports/
pytest -q tests                                                                           # sanity tests for Stages 1 to 3
```

Notes: normalisation uses per-channel statistics over the whole subject-session (saved as `norm_stats.npz` for reuse at deployment); the train/val/test split is trial-level and gesture-stratified, so overlapping windows from one trial never straddle splits. Keep `data/` out of cloud-synced folders where possible: OneDrive "Files On-Demand" can leave raw files unreadable (`Resource deadlock avoided`) until they are set to "Always keep on this device".

## Key design decisions

- Channel counts and layouts stay dataset-specific until Stage 6; no forced cross-dataset channel reduction.
- All three deep architectures receive an identical input tensor, so any accuracy/latency/size difference is attributable to architecture choice alone.
- Compression order is PTQ → QAT → pruning (not pruning first), so sparsity is set on weights that already reflect their deployed quantised distribution.
- Every MCU parameter sweep reports accuracy, latency, size, and power together — no existing HD-sEMG MCU deployment work does this jointly.

## Supervisor

Omid Kavehei, School of Biomedical Engineering, University of Sydney

