"""Milestone 3: wav2vec 2.0 (frozen) + MLP on a learned weighting of the 12 layers.

Runs on CPU in minutes, since the wav2vec 2.0 features are pre-computed.
For each seed: train, keep the best dev epoch, score eval and In-the-Wild
(if features/w2v2_base/itw.npy exists), and save the learned layer weights.

Usage:
    python scripts/train_mlp.py --feat-dir features/w2v2_base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.datasets import load_labels_from_index  # noqa: E402
from adg.models import WeightedLayerMLP  # noqa: E402
from adg.probes import load_split  # noqa: E402
from adg.training import PooledDataset, TrainConfig, chunked_mean_std, score, set_seed, train_model  # noqa: E402


def pooled_dataset(args, part, mean, std):
    ids = pd.read_csv(args.feat_dir / f"{part}_utt_ids.csv")["utt_id"].tolist()
    labels = load_labels_from_index(part, ids, args.la_root, args.itw_root)
    feats = load_split(args.feat_dir, part, labels)
    return PooledDataset(feats, labels["label"].to_numpy(), mean, std), ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feat-dir", type=Path, default=Path("features/w2v2_base"))
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--out", type=Path, default=Path("results/w2v2_mlp"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    train_feats = np.load(args.feat_dir / "train.npy", mmap_mode="r")
    mean, std = chunked_mean_std(train_feats)  # per (layer, dim), train only
    eval_parts = ["eval"] + (["itw"] if (args.feat_dir / "itw.npy").exists() else [])
    print(f"evaluating on: {eval_parts}", flush=True)

    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        cfg = TrainConfig(epochs=args.epochs, lr=1e-3, batch_size=128, patience=5, seed=seed)
        train_ds, _ = pooled_dataset(args, "train", mean, std)
        dev_ds, _ = pooled_dataset(args, "dev", mean, std)
        set_seed(seed)
        model = WeightedLayerMLP()
        t0 = time.perf_counter()
        result = train_model(model, train_ds, dev_ds, cfg)
        info = {"seed": seed, "config": asdict(cfg), "best_epoch": result.best_epoch,
                "best_dev_eer": result.best_dev_eer, "train_seconds": time.perf_counter() - t0,
                "layer_weights": model.layer_weights().detach().tolist(), "history": result.history}
        for part in eval_parts:
            ds, ids = pooled_dataset(args, part, mean, std)
            pd.DataFrame({"utt_id": ids, "score": score(model, ds)}).to_csv(
                args.out / f"scores_{part}_seed{seed}.csv", index=False)
        (args.out / f"history_seed{seed}.json").write_text(json.dumps(info, indent=2))
        print(f"seed {seed}: best epoch {result.best_epoch}, dev EER {100 * result.best_dev_eer:.3f} %", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
