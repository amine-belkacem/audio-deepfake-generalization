| System | eval, as is | eval, silence removed (same model) | eval, silence removed (retrained) | In-the-Wild, as is | In-the-Wild, silence removed (same model) | In-the-Wild, silence removed (retrained) |
|---|---|---|---|---|---|---|
| LFCC + GMM | 10.48 [10.16, 10.78] | 20.57 [20.10, 21.03] | 21.28 [20.83, 21.71] | 36.72 [36.14, 37.29] | 41.11 [40.58, 41.70] | not run |
| LFCC + LCNN | 4.86 ± 0.42 | 21.53 ± 1.09 | 17.40 ± 1.55 | 72.70 ± 1.64 | 76.78 ± 2.01 | 82.46 ± 4.11 |
| wav2vec 2.0 + probe (layer 3) | 1.21 [1.08, 1.32] | 8.20 [7.90, 8.50] | 7.68 [7.42, 7.95] | 22.44 [21.94, 22.85] | 21.71 [21.24, 22.14] | 22.71 [22.20, 23.13] |
| wav2vec 2.0 + MLP | 1.64 ± 0.20 | 11.59 ± 0.63 | 8.72 ± 0.78 | 26.18 ± 0.24 | 27.68 ± 0.42 | 28.58 ± 0.58 |
