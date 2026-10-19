"""Score any LFCC partition with the GMMs fitted in milestone 1 (no retraining).

Usage:
    python scripts/score_gmm.py --gmm-dir results/lfcc_gmm_asvspoof2019_b02 \
        --feat-dir features/lfcc_asvspoof2019_b02/itw --out results/lfcc_gmm_asvspoof2019_b02/scores_itw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.features import open_partition  # noqa: E402
from adg.gmm import DiagGMM, utterance_llr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gmm-dir", type=Path, required=True)
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bona, spoof = DiagGMM.load(args.gmm_dir / "gmm_bona.npz"), DiagGMM.load(args.gmm_dir / "gmm_spoof.npz")
    frames, index = open_partition(args.feat_dir)
    llr = utterance_llr(frames, index["start"].to_numpy(), index["n_frames"].to_numpy(), bona, spoof)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"utt_id": index["utt_id"], "score": llr}).to_csv(args.out, index=False)
    print(f"{len(llr)} utterances scored -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
