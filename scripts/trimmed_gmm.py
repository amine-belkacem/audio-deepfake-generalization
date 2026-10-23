"""Milestone 4: the LFCC + GMM baseline with the leading / trailing silence removed.

Silence is dropped from the already extracted LFCC frames (frames whose centre
falls inside the speech bounds measured by scripts/silence_stats.py), so no
audio is re-read. Two conditions:
    trained as before, tested without silence     -> scores_eval_trimtest.csv
    trained without silence, tested without it    -> scores_eval_trimboth.csv (new GMMs, ~45 min)

Usage:
    python scripts/trimmed_gmm.py --threshold-db 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.evaluation import evaluate, join_scores, load_asv_rates  # noqa: E402
from adg.features import PRESETS, open_partition  # noqa: E402
from adg.gmm import DiagGMM, fit_gmm  # noqa: E402
from adg.protocols import load_partition  # noqa: E402
from adg.silence import SpeechBounds, frame_range  # noqa: E402

HOP = PRESETS["asvspoof2019_b02"].hop_length


def speech_frames(index: pd.DataFrame, silence: pd.DataFrame, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute (start, stop) frame indices of the speech part of every utterance, in index order."""
    s = silence.set_index("utt_id").loc[index["utt_id"]]
    lo, hi = np.empty(len(index), np.int64), np.empty(len(index), np.int64)
    for i, (start, n, a, b, ns) in enumerate(zip(index["start"], index["n_frames"], s[f"start_{t}db"],
                                                  s[f"end_{t}db"], s["n_samples"])):
        f0, f1 = frame_range(SpeechBounds(int(a), int(b), int(ns)), int(n), HOP)
        lo[i], hi[i] = start + f0, start + f1
    return lo, hi


def trimmed_llr(frames, lo, hi, bona: DiagGMM, spoof: DiagGMM) -> np.ndarray:
    """Mean frame LLR over [lo, hi) of every utterance. Frame scores are computed once for the
    whole partition, then averaged per range with a cumulative sum."""
    llr = bona.frame_log_likelihood(frames) - spoof.frame_log_likelihood(frames)
    c = np.concatenate([[0.0], np.cumsum(llr)])
    return (c[hi] - c[lo]) / (hi - lo)


def gather(frames, lo, hi) -> np.ndarray:
    out = np.empty((int((hi - lo).sum()), frames.shape[1]), dtype=np.float16)
    pos = 0
    for a, b in zip(lo, hi):
        out[pos:pos + b - a] = frames[a:b]
        pos += b - a
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--feat-dir", type=Path, default=Path("features/lfcc_asvspoof2019_b02"))
    ap.add_argument("--gmm-dir", type=Path, default=Path("results/lfcc_gmm_asvspoof2019_b02"))
    ap.add_argument("--out", type=Path, default=Path("results/m4/lfcc_gmm"))
    ap.add_argument("--threshold-db", type=int, default=40)
    ap.add_argument("--skip-retrain", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t = args.threshold_db

    protos = {p: load_partition(args.la_root / "ASVspoof2019_LA_cm_protocols", p) for p in ("train", "eval")}
    asv = load_asv_rates(args.la_root)
    ev_frames, ev_index = open_partition(args.feat_dir / "eval")
    ev_lo, ev_hi = speech_frames(ev_index, pd.read_csv(f"results/m4/silence_eval.csv"), t)
    print(f"eval: {ev_index['n_frames'].sum():,} frames, {int((ev_hi - ev_lo).sum()):,} kept without silence", flush=True)

    bona, spoof = DiagGMM.load(args.gmm_dir / "gmm_bona.npz"), DiagGMM.load(args.gmm_dir / "gmm_spoof.npz")
    conditions = {"trimtest": (bona, spoof)}

    if not args.skip_retrain:
        tr_frames, tr_index = open_partition(args.feat_dir / "train")
        tr_lo, tr_hi = speech_frames(tr_index, pd.read_csv("results/m4/silence_train.csv"), t)
        labels = protos["train"].set_index("utt_id").loc[tr_index["utt_id"], "label"].to_numpy()
        models = {}
        for cls, lab in (("bona", 1), ("spoof", 0)):
            path = args.out / f"gmm_{cls}_trimmed_{t}db.npz"
            if path.exists():
                models[cls] = DiagGMM.load(path)
                continue
            x = gather(tr_frames, tr_lo[labels == lab], tr_hi[labels == lab])
            print(f"GMM {cls} without silence: {x.shape[0]:,} frames", flush=True)
            t0 = time.perf_counter()
            models[cls] = fit_gmm(x, 512, 20, seed=0)
            print(f"  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            models[cls].save(path)
            del x
        conditions["trimboth"] = (models["bona"], models["spoof"])

    rows = []
    for name, (b, s) in conditions.items():
        scores = pd.DataFrame({"utt_id": ev_index["utt_id"], "score": trimmed_llr(ev_frames, ev_lo, ev_hi, b, s)})
        scores.to_csv(args.out / f"scores_eval_{name}_{t}db.csv", index=False)
        table = evaluate(join_scores(scores, protos["eval"]), asv, n_boot=1000, seed=0)
        table.insert(0, "condition_run", name)
        rows.append(table)
        pooled = table.loc[table["condition"] == "pooled"].iloc[0]
        print(f"{name}: eval EER {pooled['eer_pct']:.2f} % [{pooled['eer_low_pct']:.2f}, {pooled['eer_high_pct']:.2f}]",
              flush=True)
    pd.concat(rows).to_csv(args.out / f"metrics_{t}db.csv", index=False, float_format="%.4f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
