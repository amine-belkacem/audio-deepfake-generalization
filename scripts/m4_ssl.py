"""Milestone 4, wav2vec 2.0 side: probe (layer 3) and MLP without silence and with bands removed.

Two training conditions, each scored on several feature sets:
    normal : trained on train / selected on dev, as in milestones 2-3
             -> scored on eval, eval_trim40, itw, itw_trim40, evalsub, evalsub_band*
    trim40 : trained on train_trim40 / selected on dev_trim40
             -> scored on eval_trim40, itw_trim40

Feature files are looked up in several folders (features/w2v2_base from
milestones 2-3, features/w2v2_m4 from the milestone 4 Kaggle run). "evalsub"
(the band-occlusion subset without any band removed) is taken from the full
eval features, so the reference and the masked versions share the same files.

Outputs: results/m4/w2v2_probe_<cond>/scores_<part>.csv and
         results/m4/w2v2_mlp_<cond>/scores_<part>_seed<s>.csv

Usage:
    python scripts/m4_ssl.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.datasets import load_labels_from_index  # noqa: E402
from adg.models import WeightedLayerMLP  # noqa: E402
from adg.probes import fit_layer_probe, probe_scores  # noqa: E402
from adg.training import PooledDataset, TrainConfig, chunked_mean_std, score, set_seed, train_model  # noqa: E402

PROBE_LAYER = 3  # picked on dev in milestone 2
BANDS = [f"band{lo}-{lo + 1000}" for lo in range(0, 8000, 1000)]
SUBSET_CSV = Path("results/m4/band_subset_eval.csv")


class FeatureStore:
    """Finds <part>.npy in a list of folders and returns (features, labels) in feature-row order."""

    def __init__(self, dirs: list[Path], la_root: Path, itw_root: Path):
        self.dirs, self.la_root, self.itw_root = dirs, la_root, itw_root

    def _find(self, part: str) -> Path:
        for d in self.dirs:
            if (d / f"{part}.npy").exists():
                return d
        raise FileNotFoundError(f"{part}.npy not found in {[str(d) for d in self.dirs]}")

    def load(self, part: str):
        if part == "evalsub":
            feats, labels = self.load("eval")
            keep = set(pd.read_csv(SUBSET_CSV)["utt_id"])
            rows = np.flatnonzero(labels["utt_id"].isin(keep).to_numpy())
            return feats[rows], labels.iloc[rows].reset_index(drop=True)
        d = self._find(part)
        done = d / f"{part}.done.npy"
        if done.exists() and not np.load(done).all():
            raise ValueError(f"{part}: extraction not finished")
        ids = pd.read_csv(d / f"{part}_utt_ids.csv")["utt_id"].tolist()
        labels = load_labels_from_index(part, ids, self.la_root, self.itw_root)
        return np.load(d / f"{part}.npy", mmap_mode="r"), labels


def write(out: Path, ids, scores):
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"utt_id": ids, "score": scores}).to_csv(out, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feat-dirs", type=Path, nargs="+", default=[Path("features/w2v2_base"), Path("features/w2v2_m4")])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--out", type=Path, default=Path("results/m4"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    store = FeatureStore(args.feat_dirs, args.la_root, args.itw_root)

    conditions = {
        "normal": ("train", "dev", ["eval", "eval_trim40", "itw", "itw_trim40", "evalsub"]
                   + [f"evalsub_{b}" for b in BANDS]),
        "trim40": ("train_trim40", "dev_trim40", ["eval_trim40", "itw_trim40"]),
    }
    k = PROBE_LAYER - 1
    for cond, (train_part, dev_part, test_parts) in conditions.items():
        xt, yt = store.load(train_part)
        xd, yd = store.load(dev_part)
        print(f"[{cond}] probe on layer {PROBE_LAYER}", flush=True)
        probe = fit_layer_probe(k, xt[:, k], yt["label"], xd[:, k], yd["label"])
        print(f"  C = {probe.c:g}, dev EER {100 * probe.dev_eer:.3f} %", flush=True)
        for part in test_parts:
            x, lab = store.load(part)
            write(args.out / f"w2v2_probe_{cond}" / f"scores_{part}.csv", lab["utt_id"], probe_scores(probe.model, x[:, k]))

        mean, std = chunked_mean_std(xt)
        for seed in args.seeds:
            print(f"[{cond}] MLP seed {seed}", flush=True)
            set_seed(seed)
            model = WeightedLayerMLP()
            res = train_model(model, PooledDataset(xt, yt["label"].to_numpy(), mean, std),
                              PooledDataset(xd, yd["label"].to_numpy(), mean, std),
                              TrainConfig(epochs=30, lr=1e-3, batch_size=128, patience=5, seed=seed), verbose=False)
            print(f"  best epoch {res.best_epoch}, dev EER {100 * res.best_dev_eer:.3f} %", flush=True)
            for part in test_parts:
                x, lab = store.load(part)
                write(args.out / f"w2v2_mlp_{cond}" / f"scores_{part}_seed{seed}.csv", lab["utt_id"],
                      score(model, PooledDataset(x, lab["label"].to_numpy(), mean, std)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
