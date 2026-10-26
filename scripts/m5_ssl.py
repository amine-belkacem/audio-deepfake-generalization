"""Milestone 5, wav2vec 2.0 side: probe (layer 3) and MLP through simulated channels.

Two training conditions, both on audio without the leading / trailing silence:
    trim40 : train_trim40 / dev_trim40, as in milestone 4
    aug    : train_trim40 + train_trim40-randchan / dev_trim40 + dev_trim40-randchan
             (every file once clean, once through one random channel)
Each model is scored on the 5,900-file eval subset, clean (evalsub_trim40) and
through each of the ten channels (evalsub_trim40-<channel>).

The clean subset is taken from the full eval_trim40 features (milestone 4), so
clean and degraded scores come from the same files.

Outputs: results/m5/w2v2_probe_<cond>/scores_<part>.csv,
         results/m5/w2v2_mlp_<cond>/scores_<part>_seed<s>.csv

Usage:
    python scripts/m5_ssl.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adg.models import WeightedLayerMLP  # noqa: E402
from adg.probes import fit_layer_probe, probe_scores  # noqa: E402
from adg.training import PooledDataset, TrainConfig, chunked_mean_std, score, set_seed, train_model  # noqa: E402
from adg.transforms import CHANNELS  # noqa: E402
from m4_ssl import PROBE_LAYER, SUBSET_CSV, FeatureStore, write  # noqa: E402


def load_any(store: FeatureStore, part: str):
    """Like store.load, but 'evalsub_<x>' falls back to the rows of 'eval_<x>' that are in the subset."""
    try:
        return store.load(part)
    except FileNotFoundError:
        if not part.startswith("evalsub_"):
            raise
    feats, labels = store.load(part.replace("evalsub_", "eval_", 1))
    keep = set(pd.read_csv(SUBSET_CSV)["utt_id"])
    rows = np.flatnonzero(labels["utt_id"].isin(keep).to_numpy())
    return feats[rows], labels.iloc[rows].reset_index(drop=True)


def load_concat(store: FeatureStore, parts: list[str]):
    loaded = [load_any(store, p) for p in parts]
    feats = np.concatenate([np.asarray(f) for f, _ in loaded]) if len(loaded) > 1 else loaded[0][0]
    labels = pd.concat([lab for _, lab in loaded], ignore_index=True)
    return feats, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feat-dirs", type=Path, nargs="+",
                    default=[Path("features/w2v2_base"), Path("features/w2v2_m4"), Path("features/w2v2_m5")])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--out", type=Path, default=Path("results/m5"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    store = FeatureStore(args.feat_dirs, args.la_root, args.itw_root)

    tests = ["evalsub_trim40"] + [f"evalsub_trim40-{ch}" for ch in CHANNELS]
    conditions = {
        "trim40": (["train_trim40"], ["dev_trim40"]),
        "aug": (["train_trim40", "train_trim40-randchan"], ["dev_trim40", "dev_trim40-randchan"]),
    }
    k = PROBE_LAYER - 1
    for cond, (train_parts, dev_parts) in conditions.items():
        xt, yt = load_concat(store, train_parts)
        xd, yd = load_concat(store, dev_parts)
        print(f"[{cond}] train {len(yt)} files, dev {len(yd)} files", flush=True)
        probe = fit_layer_probe(k, xt[:, k], yt["label"], xd[:, k], yd["label"])
        print(f"  probe layer {PROBE_LAYER}: C = {probe.c:g}, dev EER {100 * probe.dev_eer:.3f} %", flush=True)
        for part in tests:
            x, lab = load_any(store, part)
            write(args.out / f"w2v2_probe_{cond}" / f"scores_{part}.csv", lab["utt_id"], probe_scores(probe.model, x[:, k]))

        mean, std = chunked_mean_std(xt)
        for seed in args.seeds:
            set_seed(seed)
            model = WeightedLayerMLP()
            res = train_model(model, PooledDataset(xt, yt["label"].to_numpy(), mean, std),
                              PooledDataset(xd, yd["label"].to_numpy(), mean, std),
                              TrainConfig(epochs=30, lr=1e-3, batch_size=128, patience=5, seed=seed), verbose=False)
            print(f"  MLP seed {seed}: best epoch {res.best_epoch}, dev EER {100 * res.best_dev_eer:.3f} %", flush=True)
            for part in tests:
                x, lab = load_any(store, part)
                write(args.out / f"w2v2_mlp_{cond}" / f"scores_{part}_seed{seed}.csv", lab["utt_id"],
                      score(model, PooledDataset(x, lab["label"].to_numpy(), mean, std)))
        del xt, xd
    return 0


if __name__ == "__main__":
    sys.exit(main())
