"""LFCC frames for any partition, In-the-Wild included, optionally transformed or subsampled.

Output folder: features/lfcc_<preset>/<partition>[sub][_<transform>], e.g.
    eval                    the full eval set, untouched
    eval_trim40             full eval set, silence trimmed at 40 dB
    evalsub_band1000-2000   the stratified eval subset of milestone 4, 1-2 kHz removed

Usage:
    python scripts/extract_lfcc.py --la-root data/LA --partitions train dev eval
    python scripts/extract_lfcc.py --itw-root data/in_the_wild --partitions itw
    python scripts/extract_lfcc.py --partitions eval --transform trim40
    python scripts/extract_lfcc.py --partitions eval --subset results/m4/band_subset_eval.csv --transform band:1000-2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.features import PRESETS, LfccExtractor, default_workers, extract_files  # noqa: E402
from adg import asvspoof2021  # noqa: E402
from adg.in_the_wild import read_in_the_wild  # noqa: E402
from adg.protocols import check_counts, load_partition  # noqa: E402
from adg.transforms import folder_name, validate  # noqa: E402

AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}


def output_name(part: str, subset: Path | None, transform: str) -> str:
    name = part + ("sub" if subset is not None else "")
    return name if transform == "none" else f"{name}_{folder_name(transform)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--partitions", nargs="+", required=True, choices=[*AUDIO_DIRS, "itw", "asv21"])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="asvspoof2019_b02")
    ap.add_argument("--feat-root", type=Path, default=Path("features"))
    ap.add_argument("--transform", default="none", type=validate)
    ap.add_argument("--subset", type=Path, default=None, help="CSV with a utt_id column (rows kept in protocol order)")
    ap.add_argument("--asv21-root", type=Path, default=Path("data/asvspoof2021"))
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    extractor = LfccExtractor(PRESETS[args.preset])
    keep = set(pd.read_csv(args.subset)["utt_id"]) if args.subset else None
    for part in args.partitions:
        out = args.feat_root / f"lfcc_{args.preset}" / output_name(part, args.subset, args.transform)
        if (out / "index.csv").exists():
            print(f"{out.name}: already extracted in {out}, skipped")
            continue
        if part == "itw":
            df = read_in_the_wild(args.itw_root)
            resample = True
        elif part == "asv21":
            df = asvspoof2021.read_metadata(args.asv21_root / asvspoof2021.METADATA)
            df["path"] = [str(args.asv21_root / asvspoof2021.AUDIO / f"{u}.flac") for u in df["utt_id"]]
            resample = False
        else:
            df = load_partition(args.la_root / "ASVspoof2019_LA_cm_protocols", part)
            check_counts(df, part)
            df["path"] = [str(args.la_root / AUDIO_DIRS[part] / "flac" / f"{u}.flac") for u in df["utt_id"]]
            resample = False
        if keep is not None:
            df = df[df["utt_id"].isin(keep)]
            if df.empty:
                raise SystemExit(f"{args.subset}: no id of the {part} partition")
        t0 = time.perf_counter()
        extract_files(df["path"].tolist(), df["utt_id"].tolist(), out, extractor, workers=args.workers,
                      resample=resample, transform=args.transform)
        print(f"{out.name}: {len(df)} files in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
