"""Score feature folders with already trained LCNNs (weights + normalisation from train_lcnn.py).

Usage:
    python scripts/score_lcnn.py --model-dir results/m4/lfcc_lcnn_trim40 \
        --feat-dir features/lfcc_asvspoof2019_b02 --parts evalsub_trim40-g711 --out results/m5/lfcc_lcnn_trim40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adg.models import LCNN  # noqa: E402
from adg.training import score  # noqa: E402
from train_lcnn import crop_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--asv21-root", type=Path, default=Path("data/asvspoof2021"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-utts", type=int, default=None, help="smoke test: strided subset of each part")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    norm = np.load(args.model_dir / "lfcc_norm.npz")
    mean, std = norm["mean"], norm["std"]
    for seed in args.seeds:
        model = LCNN()
        model.load_state_dict(torch.load(args.model_dir / f"lcnn_seed{seed}.pt", map_location="cpu"))
        model.to(args.device)
        for part in args.parts:
            ds, ids = crop_dataset(args.feat_dir, part, args, mean, std, train=False)
            pd.DataFrame({"utt_id": ids, "score": score(model, ds, device=args.device)}).to_csv(
                args.out / f"scores_{part}_seed{seed}.csv", index=False)
        print(f"seed {seed}: {len(args.parts)} parts scored", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
