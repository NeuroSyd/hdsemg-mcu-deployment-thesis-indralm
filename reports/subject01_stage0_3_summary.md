# Subject 01: Stage 0 to 3 proof run

- Stage 0: 399 dynamic trials (session counts: S1=202, S2=197), all SHA256-verified.
- Stage 2: 3192 windows of 410 samples (256 ch), step 205.
- Stage 3 split (session 1 only): {'train': 134, 'val': 34, 'test': 34} trials, trial-level, gesture-stratified, seed 42; 272 test windows.

## Primary split (seed 42)

| model               |   per_window_acc_% |   per_trial_acc_% |   xsession_window_% |   xsession_trial_% |   parameters |   MFLOPs/inference |   latency_ms_desktop |   epochs |   train_s |
|:--------------------|-------------------:|------------------:|--------------------:|-------------------:|-------------:|-------------------:|---------------------:|---------:|----------:|
| LDA (baseline)      |               79   |              85.3 |                33.5 |               37.1 |          nan |              nan   |               nan    |      nan |     nan   |
| 1D CNN              |               76.8 |              88.2 |                61.7 |               71.6 |        97666 |               46.3 |                 1.6  |       53 |      80.2 |
| TCN                 |               74.6 |              82.4 |                61.7 |               75.6 |       181602 |              146.1 |                 3.24 |       27 |     241.3 |
| GRU, time pooled x8 |               61.4 |              67.6 |                53.7 |               68.5 |       187682 |               19.2 |                 3.8  |       25 |      50.9 |
| GRU, raw 410 steps  |               33.8 |              55.9 |                26.8 |               41.6 |       187682 |              151.4 |                30.16 |       21 |     320.7 |

## Repeated random trial splits (mean ± SD)

| model               |   splits | per_window_%   | per_trial_%   | xsession_window_%   | xsession_trial_%   |
|:--------------------|---------:|:---------------|:--------------|:--------------------|:-------------------|
| LDA (baseline)      |        4 | 84.0 ± 4.5     | 91.2 ± 5.4    | 32.6 ± 0.6          | 35.3 ± 1.7         |
| 1D CNN              |        4 | 79.4 ± 1.9     | 87.5 ± 2.8    | 65.9 ± 3.6          | 79.2 ± 5.7         |
| TCN                 |        4 | 75.9 ± 3.3     | 84.6 ± 1.5    | 58.7 ± 5.0          | 72.1 ± 8.1         |
| GRU, time pooled x8 |        4 | 67.3 ± 4.9     | 81.6 ± 9.4    | 55.2 ± 1.0          | 68.9 ± 1.5         |
