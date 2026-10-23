"""Milestone 3: LFCC + LCNN, one run per seed.

For each seed: train on train (random 4 s crops), keep the epoch with the best
dev EER, then score eval and, when available, In-the-Wild (first 4 s of each file).

Outputs in --out (default results/lfcc_lcnn):
    scores_<partition>_seed<s>.csv, history_seed<s>.json, lcnn_seed<s>.pt (not committed)

Usage (Kaggle GPU):
    python scripts/train_lcnn.py --feat-dir /tmp/features/lfcc_asvspoof2019_b02 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.datasets import load_labels_from_index  # noqa: E402
from adg.features import open_partition  # noqa: E402
from adg.models import LCNN  # noqa: E402
from adg.training import FrameCropDataset, TrainConfig, chunked_mean_std, score, set_seed, train_model  # noqa: E402

CROP_FRAMES = 400  # 4 s at a 10 ms hop


def crop_dataset(feat_dir: Path, part: str, args, mean, std, train: bool, seed: int = 0):
    frames, index = open_partition(feat_dir / part)
    labels = load_labels_from_index(part, index["utt_id"], args.la_root, args.itw_root)
    if args.max_utts is not None:
        # ASVspoof protocols list bona fide first, so take a strided subset to keep both classes
        keep = np.linspace(0, len(index) - 1, min(args.max_utts, len(index))).astype(int)
        index, labels = index.iloc[keep].reset_index(drop=True), labels.iloc[keep].reset_index(drop=True)
    ds = FrameCropDataset(frames, index["start"].to_numpy(), index["n_frames"].to_numpy(),
                          labels["label"].to_numpy(), mean, std, n_frames=CROP_FRAMES, train=train, seed=seed)
    return ds, index["utt_id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feat-dir", type=Path, default=Path("features/lfcc_asvspoof2019_b02"))
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--out", type=Path, default=Path("results/lfcc_lcnn"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-utts", type=int, default=None, help="smoke test: first N utterances of each split")
    ap.add_argument("--train-part", default="train", help="e.g. train_trim40 (milestone 4)")
    ap.add_argument("--dev-part", default="dev", help="e.g. dev_trim40 (milestone 4)")
    ap.add_argument("--eval-parts", nargs="+", default=None,
                    help="feature folders to score; default: eval and itw when they exist")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_frames, _ = open_partition(args.feat_dir / args.train_part)
    mean, std = chunked_mean_std(train_frames, chunk=500_000)  # per-dimension, train only
    if args.eval_parts is None:
        eval_parts = [p for p in ("eval", "itw") if (args.feat_dir / p / "index.csv").exists()]
    else:
        eval_parts = args.eval_parts
        missing = [p for p in eval_parts if not (args.feat_dir / p / "index.csv").exists()]
        if missing:
            raise SystemExit(f"no features for {missing} in {args.feat_dir}")
    print(f"evaluating on: {eval_parts}", flush=True)

    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=64, patience=5, seed=seed)
        train_ds, _ = crop_dataset(args.feat_dir, args.train_part, args, mean, std, train=True, seed=seed)
        dev_ds, _ = crop_dataset(args.feat_dir, args.dev_part, args, mean, std, train=False)
        set_seed(seed)  # weight initialisation happens here
        model = LCNN()
        t0 = time.perf_counter()
        result = train_model(model, train_ds, dev_ds, cfg, device=args.device)
        info = {"seed": seed, "config": asdict(cfg), "best_epoch": result.best_epoch,
                "best_dev_eer": result.best_dev_eer, "train_seconds": time.perf_counter() - t0,
                "history": result.history, "device": args.device, "crop_frames": CROP_FRAMES,
                "train_part": args.train_part, "dev_part": args.dev_part}
        torch.save(model.state_dict(), args.out / f"lcnn_seed{seed}.pt")
        for part in eval_parts:
            ds, ids = crop_dataset(args.feat_dir, part, args, mean, std, train=False)
            t1 = time.perf_counter()
            s = score(model, ds, device=args.device)
            info[f"score_{part}_seconds"] = time.perf_counter() - t1
            pd.DataFrame({"utt_id": ids, "score": s}).to_csv(args.out / f"scores_{part}_seed{seed}.csv", index=False)
        (args.out / f"history_seed{seed}.json").write_text(json.dumps(info, indent=2))
        print(f"seed {seed}: best epoch {result.best_epoch}, dev EER {100 * result.best_dev_eer:.3f} %", flush=True)
    np.savez(args.out / "lfcc_norm.npz", mean=mean, std=std)
    return 0


if __name__ == "__main__":
    sys.exit(main())
