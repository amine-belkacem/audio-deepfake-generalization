"""Milestone 2: pooled wav2vec 2.0 BASE features for every layer, all partitions.

Meant to run on a Kaggle GPU (see kaggle/extract_wav2vec2.ipynb), but works on
CPU too, just ~20 h instead of ~1 h on my laptop.

Output, per partition, in --out:
    <partition>.npy        float16, (n_utts, 12, 1536): mean and std over time per layer
    <partition>.done.npy   resume mask
    <partition>_utt_ids.csv  row order (same as the official protocol)
    run_info.json          device, versions, time per file

Usage:
    python scripts/extract_wav2vec2.py --la-root data/LA --out features/w2v2_base --device cuda
    python scripts/extract_wav2vec2.py --itw-root data/in_the_wild --partitions itw --out features/w2v2_base
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.in_the_wild import read_in_the_wild  # noqa: E402
from adg.protocols import check_counts, load_partition  # noqa: E402
from adg.ssl_features import Wav2Vec2LayerExtractor, extract_to_memmap  # noqa: E402

AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}
N_LAYERS, FEAT_DIM = 12, 2 * 768


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--partitions", nargs="+", default=["train", "dev", "eval"], choices=[*AUDIO_DIRS, "itw"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N files per partition")
    ap.add_argument("--random-weights", action="store_true", help="smoke test: untrained model, no download")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = torchaudio.models.wav2vec2_base() if args.random_weights else None
    extractor = Wav2Vec2LayerExtractor(model=model, device=args.device)
    info_path = args.out / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({"model": "wav2vec2_base, RANDOM weights (smoke test)" if args.random_weights
                 else "torchaudio.pipelines.WAV2VEC2_BASE", "limit": args.limit, "device": args.device,
                 "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                 "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                 "python": platform.python_version()})

    proto_dir = args.la_root / "ASVspoof2019_LA_cm_protocols"
    for part in args.partitions:
        if part == "itw":
            df = read_in_the_wild(args.itw_root)
        else:
            df = load_partition(proto_dir, part)
            check_counts(df, part)
        if args.limit is not None:
            df = df.head(args.limit)
        df[["utt_id"]].to_csv(args.out / f"{part}_utt_ids.csv", index=False)
        if part == "itw":
            paths = [Path(p) for p in df["path"]]
        else:
            paths = [args.la_root / AUDIO_DIRS[part] / "flac" / f"{u}.flac" for u in df["utt_id"]]

        t0 = time.perf_counter()
        extract_to_memmap(paths, args.out / f"{part}.npy", extractor, N_LAYERS, FEAT_DIM,
                          resample=(part == "itw"))
        elapsed = time.perf_counter() - t0
        # Only meaningful for an uninterrupted run; a resumed run times the remainder.
        info[f"{part}_seconds"] = elapsed
        info[f"{part}_ms_per_file"] = 1000 * elapsed / len(paths)
        info_path.write_text(json.dumps(info, indent=2))
        print(f"{part}: {len(paths)} files in {elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
