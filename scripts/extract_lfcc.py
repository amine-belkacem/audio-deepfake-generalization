"""LFCC frames for any partition, In-the-Wild included.

Usage:
    python scripts/extract_lfcc.py --la-root data/LA --partitions train dev eval
    python scripts/extract_lfcc.py --itw-root data/in_the_wild --partitions itw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.features import PRESETS, LfccExtractor, default_workers, extract_files  # noqa: E402
from adg.in_the_wild import read_in_the_wild  # noqa: E402
from adg.protocols import check_counts, load_partition  # noqa: E402

AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--partitions", nargs="+", required=True, choices=[*AUDIO_DIRS, "itw"])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="asvspoof2019_b02")
    ap.add_argument("--feat-root", type=Path, default=Path("features"))
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    extractor = LfccExtractor(PRESETS[args.preset])
    for part in args.partitions:
        out = args.feat_root / f"lfcc_{args.preset}" / part
        if (out / "index.csv").exists():
            print(f"{part}: already extracted in {out}, skipped")
            continue
        if part == "itw":
            df = read_in_the_wild(args.itw_root)
            paths, resample = df["path"].tolist(), True
        else:
            df = load_partition(args.la_root / "ASVspoof2019_LA_cm_protocols", part)
            check_counts(df, part)
            paths = [args.la_root / AUDIO_DIRS[part] / "flac" / f"{u}.flac" for u in df["utt_id"]]
            resample = False
        t0 = time.perf_counter()
        extract_files(paths, df["utt_id"].tolist(), out, extractor, workers=args.workers, resample=resample)
        print(f"{part}: {len(df)} files in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
