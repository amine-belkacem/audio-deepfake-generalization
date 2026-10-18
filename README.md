# audio-deepfake-generalization

![tests](https://github.com/amine-belkacem/audio-deepfake-generalization/actions/workflows/tests.yml/badge.svg)

Real or synthetic voice? I compare classical features (LFCC) with a frozen
self-supervised model (wav2vec 2.0) on ASVspoof 2019 LA, and look at what
happens on generators the detectors never saw during training.

Three questions drive the project:

1. **Unseen attacks.** 11 of the 13 eval attacks are generators absent from training.
   How much does each approach lose on them?
2. **Shortcuts.** ASVspoof 2019 is known to leak the label through leading and trailing
   silence. Are my detectors listening to the voice, or to the silence?
3. **Telephone channel.** Voice fraud happens over the phone. What is left of the
   detection after 8 kHz resampling, G.711, Opus and packet loss?

## Status

Work in progress. Milestone 0 is done: protocol parsing, EER, min t-DCF and
bootstrap confidence intervals, all unit-tested in CI. No results yet: every number
will be added here only once it has been measured, with the script that produced it.

| Milestone | Content | State |
|---|---|---|
| M0 | Foundations: metrics, protocols, tests, CI | done |
| M1 | LFCC + GMM baseline, reproduce the official EER | next |
| M2 | wav2vec 2.0 layer-wise probing | |
| M3 | Full comparison, 3 seeds, In-the-Wild | |
| M4 | Silence shortcut and frequency-band analysis | |
| M5 | Telephone-channel robustness | |
| M6 | Raspberry Pi 5 inference (optional) | |

## Running the tests

```bash
pip install -r requirements-ci.txt
pytest -v
```

## Data

See [docs/DATA.md](docs/DATA.md) for sources, licences and attribution. No audio is
stored in this repository.

## License

Code: MIT. Datasets keep their own licences (see docs/DATA.md).
