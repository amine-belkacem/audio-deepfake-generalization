"""Milestone 1: LFCC + GMM baseline on ASVspoof 2019 LA.

Stages (each one is skipped if its output already exists, so the script can be
re-run after an interruption):
    1. LFCC extraction for train and eval        -> features/lfcc_<preset>/<partition>/
    2. one 512-component GMM per class on train  -> results/lfcc_gmm_<preset>/gmm_{bona,spoof}.npz
    3. eval scores                               -> results/lfcc_gmm_<preset>/scores_eval.csv
    4. metrics                                   -> results/lfcc_gmm_<preset>/metrics_eval.csv

Usage:
    python scripts/run_lfcc_gmm.py --preset asvspoof2019_b02 --em-iter 20
    python scripts/run_lfcc_gmm.py --preset first_run --em-iter 10   # first attempt, kept for the record
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.evaluation import evaluate, join_scores, load_asv_rates  # noqa: E402
from adg.features import PRESETS, LfccExtractor, default_workers, extract_partition, open_partition  # noqa: E402
from adg.gmm import DiagGMM, fit_gmm, utterance_llr  # noqa: E402
from adg.protocols import check_counts, load_partition  # noqa: E402

AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "eval": "ASVspoof2019_LA_eval"}


def class_frames(frames: np.ndarray, index: pd.DataFrame, utt_ids: set[str]) -> np.ndarray:
    """Copy the frames of the selected utterances into one float16 array."""
    sel = index[index["utt_id"].isin(utt_ids)]
    out = np.empty((int(sel["n_frames"].sum()), frames.shape[1]), dtype=np.float16)
    pos = 0
    for start, n in zip(sel["start"].to_numpy(), sel["n_frames"].to_numpy()):
        out[pos:pos + n] = frames[start:start + n]
        pos += n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--preset", choices=sorted(PRESETS), default="asvspoof2019_b02")
    ap.add_argument("--components", type=int, default=512)
    ap.add_argument("--em-iter", type=int, default=20, help="20 in the official baseline")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.feat_dir = Path("features") / f"lfcc_{args.preset}"
    args.out = Path("results") / f"lfcc_gmm_{args.preset}"

    args.out.mkdir(parents=True, exist_ok=True)
    proto_dir = args.la_root / "ASVspoof2019_LA_cm_protocols"
    protocols = {p: load_partition(proto_dir, p) for p in ("train", "eval")}
    for p, df in protocols.items():
        check_counts(df, p)

    timings = {}
    extractor = LfccExtractor(PRESETS[args.preset])

    # 1. features
    for p, df in protocols.items():
        if (args.feat_dir / p / "index.csv").exists():
            continue
        t0 = time.perf_counter()
        extract_partition(df["utt_id"].tolist(), args.la_root / AUDIO_DIRS[p] / "flac",
                          args.feat_dir / p, extractor, workers=args.workers)
        timings[f"lfcc_{p}_s"] = time.perf_counter() - t0

    # 2. GMMs
    train_frames, train_index = open_partition(args.feat_dir / "train")
    train = protocols["train"]
    gmms = {}
    for cls, label in (("bona", 1), ("spoof", 0)):
        path = args.out / f"gmm_{cls}.npz"
        if path.exists():
            gmms[cls] = DiagGMM.load(path)
            continue
        x = class_frames(train_frames, train_index, set(train.loc[train["label"] == label, "utt_id"]))
        print(f"GMM {cls}: {x.shape[0]:,} frames", flush=True)
        t0 = time.perf_counter()
        gmms[cls] = fit_gmm(x, args.components, args.em_iter, seed=args.seed)
        timings[f"gmm_{cls}_s"] = time.perf_counter() - t0
        gmms[cls].save(path)
        del x

    # 3. scores
    score_path = args.out / "scores_eval.csv"
    if not score_path.exists():
        frames, index = open_partition(args.feat_dir / "eval")
        t0 = time.perf_counter()
        llr = utterance_llr(frames, index["start"].to_numpy(), index["n_frames"].to_numpy(),
                            gmms["bona"], gmms["spoof"])
        timings["scoring_eval_s"] = time.perf_counter() - t0
        pd.DataFrame({"utt_id": index["utt_id"], "score": llr}).to_csv(score_path, index=False)

    # 4. metrics
    merged = join_scores(pd.read_csv(score_path), protocols["eval"])
    asv = load_asv_rates(args.la_root)
    table = evaluate(merged, asv, n_boot=args.n_boot, seed=args.seed)
    table.to_csv(args.out / "metrics_eval.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    run_info_path = args.out / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    run_info.update({
        "preset": args.preset, "lfcc_config": asdict(PRESETS[args.preset]), "workers": args.workers,
        "seed": args.seed, "components": args.components, "em_iter": args.em_iter,
        "asv_rates": asdict(asv),
        "machine": platform.processor() or platform.machine(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), **timings,
    })
    run_info_path.write_text(json.dumps(run_info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
