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
| 4. Compression | PTQ → QAT → pruning, applied to the best accuracy/efficiency architecture from Stage 3 | Compressed checkpoints + accuracy retention curve |
| 5. MCU deployment | Accuracy, latency, model size, power measured jointly on one board | Full on-device benchmark |
| 6. Robustness | Cross-session/cross-day evaluation + simulated channel dropout | Degradation curves vs. day gap and vs. channel loss |

## Key design decisions

- Channel counts and layouts stay dataset-specific until Stage 6; no forced cross-dataset channel reduction.
- All three deep architectures receive an identical input tensor, so any accuracy/latency/size difference is attributable to architecture choice alone.
- Compression order is PTQ → QAT → pruning (not pruning first), so sparsity is set on weights that already reflect their deployed quantised distribution.
- Every MCU parameter sweep reports accuracy, latency, size, and power together — no existing HD-sEMG MCU deployment work does this jointly.

## Status

Currently at Week 6 of Thesis A (Sem 2 2026). Framework design complete; Stage 0–1 (extraction/preprocessing) implementation in progress. See weekly progress reports for full detail.

## Supervisor

Omid Kavehei, School of Biomedical Engineering, University of Sydney

