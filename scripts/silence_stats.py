"""Milestone 4: leading / trailing silence of every ASVspoof 2019 LA file, at several thresholds.

Output: results/m4/silence_<partition>.csv with, per utterance, the file length
and the leading / trailing silence in seconds for each threshold (30, 40, 50 dB
below the loudest frame). Several thresholds so that no conclusion hangs on one
arbitrary setting.

Usage:
    python scripts/silence_stats.py --la-root data/LA
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.features import default_workers  # noqa: E402
from adg.protocols import check_counts, load_partition  # noqa: E402
from adg.silence import speech_bounds  # noqa: E402

AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}
THRESHOLDS_DB = (30, 40, 50)


def measure(path: str) -> dict:
    x, sr = sf.read(path, dtype="float32")
    row = {"n_samples": x.size, "duration_s": x.size / sr}
    for t in THRESHOLDS_DB:
        b = speech_bounds(x, sr, top_db=t)
        row[f"lead_s_{t}db"] = b.lead_samples / sr
        row[f"trail_s_{t}db"] = b.trail_samples / sr
        row[f"start_{t}db"], row[f"end_{t}db"] = b.start, b.end
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--out", type=Path, default=Path("results/m4"))
    ap.add_argument("--partitions", nargs="+", default=list(AUDIO_DIRS), choices=list(AUDIO_DIRS))
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for part in args.partitions:
        df = load_partition(args.la_root / "ASVspoof2019_LA_cm_protocols", part)
        check_counts(df, part)
        paths = [str(args.la_root / AUDIO_DIRS[part] / "flac" / f"{u}.flac") for u in df["utt_id"]]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(measure, paths, chunksize=256))
        stats = pd.concat([df[["utt_id", "attack", "key", "label"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        stats.to_csv(args.out / f"silence_{part}.csv", index=False, float_format="%.4f")
        print(f"{part}: {len(stats)} files in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
