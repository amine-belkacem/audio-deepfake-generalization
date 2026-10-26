# Data sources and licences

No audio is stored in this repository. The scripts in `scripts/` download each
dataset from its official source into `data/` (ignored by git).

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| ASVspoof 2019 LA | train / dev / eval, main experiments | [Edinburgh DataShare, doi:10.7488/ds/2555](https://doi.org/10.7488/ds/2555) (`LA.zip`, 7.12 GB) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) (checked in `LICENSE_text.txt` of the release) |
| ASVspoof 2021 LA (eval) | real telephony / VoIP channels (milestone 5), **not used in the end** | audio: [Zenodo record 4837263](https://zenodo.org/record/4837263) (7.8 GB); labels and channel metadata: `LA-keys-full.tar.gz` on asvspoof.org | ODC-By 1.0 (checked on the Zenodo page). asvspoof.org answered HTTP 503 throughout milestone 5, so the labels could not be fetched and this set was skipped |
| In-the-Wild | out-of-domain test (milestone 3) | [Hugging Face `mueller91/In-The-Wild`](https://huggingface.co/datasets/mueller91/In-The-Wild) (`release_in_the_wild.zip`, 8.16 GB); also listed on the [Fraunhofer AISEC page](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | The two sources disagree: CC-BY-SA 4.0 on the Hugging Face card, Apache 2.0 on the Fraunhofer page when I checked it. Both allow what I do here (use the data, publish scores and metrics with attribution, never redistribute the audio) |

## Kaggle runs and the community copy of ASVspoof 2019

From Kaggle, Edinburgh DataShare answered HTTP 403 to every request (milestone 3, October 2026), while it
answered normally from my laptop. The milestone 3 notebook therefore reads the community copy
[awsaf49/asvpoof-2019-dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset), which ODC-By
allows. Before anything runs, `scripts/verify_asvspoof_copy.py` checks it against my official download: the
official counts, speaker disjointness, the presence of all 121,461 audio files, and SHA-256 digests of the
protocol files, the ASV score file and 60 random audio files (`docs/asvspoof2019_la_sha256.json`). It is a
sample check, not a full byte comparison.

## Attribution

ASVspoof 2019:

> Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N.,
> Kinnunen, T., Lee, K. A., Vestman, V., Nautsch, A. (2019). ASVspoof 2019: The 3rd
> Automatic Speaker Verification Spoofing and Countermeasures Challenge database.
> University of Edinburgh, CSTR. https://doi.org/10.7488/ds/2555

In-the-Wild:

> Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., Böttinger, K. (2022).
> Does Audio Deepfake Detection Generalize? Interspeech 2022. arXiv:2203.16263

## Expected counts (ASVspoof 2019 LA, CM protocols)

These are checked automatically by `adg.protocols.check_counts`.

| Partition | Bona fide | Spoof | Attacks | Total |
|---|---|---|---|---|
| train | 2,580 | 22,800 (6 × 3,800) | A01–A06 | 25,380 |
| dev | 2,548 | 22,296 (6 × 3,716) | A01–A06 | 24,844 |
| eval | 7,355 | 63,882 (13 × 4,914) | A07–A19 | 71,237 |

On the eval set, A16 and A19 reuse the algorithms of A04 and A06, so I treat them as
"known" attacks. The other 11 are generators the model never saw in training.
