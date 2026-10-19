# Milestone 2: which wav2vec 2.0 layer hears the fake?

## Setup

- **Model**: wav2vec 2.0 BASE (`torchaudio.pipelines.WAV2VEC2_BASE`, pre-trained on
  LibriSpeech, no fine-tuning), frozen.
- **Features**: for each of the 12 transformer layers, mean and standard deviation over
  time of the 768-dim frame vectors, so 1536 numbers per layer and per utterance.
- **Probe**: one logistic regression per layer (standardised inputs, balanced class
  weights). C is picked in {0.001, 0.01, 0.1, 1} on the **dev** set, which only contains
  the training attacks A01–A06. Eval is never used to choose anything.
- **Extraction**: Kaggle, Tesla T4, one file at a time (no padding, see
  `src/adg/ssl_features.py` for why). Measured 22–24 ms per file, 46 min for the
  121,461 files of train + dev + eval. For comparison, the same model on my laptop CPU
  took 592 ± 20 ms for a 3.5 s file (10 runs), so about 20 h for the dataset.
- Intervals are 95 % bootstrap (1000 resamples), seed 0.

## Result

![EER per layer](../results/w2v2_probes/eer_per_layer.png)

| Layer | eval EER (%) | 95 % CI | min t-DCF | dev EER (%) |
|---|---|---|---|---|
| 1 | 1.76 | [1.62, 1.92] | 0.054 | 0.19 |
| 2 | 0.99 | [0.86, 1.11] | 0.031 | 0.16 |
| **3** (picked on dev) | **1.21** | [1.08, 1.32] | 0.037 | **0.15** |
| 4 | 0.98 | [0.87, 1.07] | 0.027 | 0.20 |
| 5 | 1.85 | [1.67, 1.99] | 0.053 | 0.27 |
| 6 | 2.63 | [2.44, 2.80] | 0.073 | 0.47 |
| 7 | 3.57 | [3.36, 3.75] | 0.102 | 0.79 |
| 8 | 3.69 | [3.48, 3.91] | 0.102 | 0.82 |
| 9 | 3.71 | [3.51, 3.94] | 0.105 | 0.42 |
| 10 | 2.79 | [2.60, 2.97] | 0.078 | 0.24 |
| 11 | 3.15 | [2.93, 3.40] | 0.095 | 0.78 |
| 12 | 3.87 | [3.62, 4.15] | 0.118 | 1.06 |

What I take from it:

- **The useful information sits in the early layers, 2 to 4**, not in the last one. Layer
  12 is about four times worse than layer 4, and their intervals are far apart
  ([0.87, 1.07] against [3.62, 4.15]). That answers hypothesis H2 from my spec. My
  reading, not verified: the last layers of a model pre-trained to predict masked speech
  units become more abstract and phonetic, and lose the low-level acoustic detail where
  vocoder artefacts live.
- **Dev picked layer 3, eval prefers layers 2 and 4.** The three are within a few tenths
  of a point and their intervals overlap. I report layer 3 because that is what the
  protocol picked; choosing layer 4 after looking at eval would be cheating by 0.2 points.
- **Dev is a poor guide to eval.** Dev EERs are 0.15–1.06 %, eval 1–4 %, because dev only
  contains attacks seen in training. The gap between the two is the generalisation cost.

## Against the LFCC + GMM baseline (milestone 1)

| | pooled EER (%) | known A16, A19 | unseen (11 attacks) | min t-DCF |
|---|---|---|---|---|
| LFCC + GMM (B02 settings) | 10.48 [10.16, 10.78] | 16.09 | 9.02 | 0.279 |
| wav2vec 2.0 layer 3 + linear probe | **1.21** [1.08, 1.32] | **0.90** | **1.25** | **0.037** |

A linear model on a frozen layer divides the EER by about 8.7 on the full eval set and
by 7.2 on the unseen attacks. The intervals do not overlap, so hypothesis H1 holds.

Per attack, the hardest one stays the same for both systems: A10 (22.8 % for LFCC + GMM,
3.7 % for the probe). A19, the "known" attack that LFCC + GMM failed on (20.0 %), drops
to 1.1 %. Full table: `results/w2v2_probes/metrics_best_layer.csv`.

## Why I do not fully trust this yet

1.2 % EER from a logistic regression on a time average is almost too good. The features
are pooled over the whole file, **silence included**. ASVspoof 2019 is known to leak the
label through the length of leading and trailing silence (Müller et al., 2021, "Speech
is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?",
arXiv:2106.12914), and a mean and a standard deviation over time can easily encode "how much of this
file is silence". Milestone 4 tests exactly that: same probes after trimming silence,
and a model that only sees silence duration. Until then, these numbers are an upper
bound on what the model really detects.

## Reproducing

```bash
# on Kaggle: kaggle/extract_wav2vec2.ipynb  (GPU, Internet on)
# then, with features/w2v2_base/ downloaded:
python scripts/run_probes.py --feat-dir features/w2v2_base
```
