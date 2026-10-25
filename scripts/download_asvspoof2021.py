"""Download ASVspoof 2021 LA eval (audio + keys), check the published MD5, build the milestone 5 subset.

    audio: Zenodo record 4837263, ASVspoof2021_LA_eval.tar.gz, 7.8 GB, MD5 2abee34d8b0b91159555fc4f016e4562
    keys:  https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz, MD5 f052cc2ed276745afa3b5198665d3b26
    licence: ODC-By 1.0

Outputs: <out>/ASVspoof2021_LA_eval/flac/*.flac, <out>/keys/LA/CM/trial_metadata.txt,
<out>/asv21_labels.csv (labels + channel, no audio) and the subset list
results/m5/asv21_subset.csv (eval trials, up to 400 bona fide + 400 spoof per
codec/transmission condition, seed 0).

Exit code 2 when the keys cannot be fetched (asvspoof.org was returning HTTP 503
when I wrote this): the audio alone is useless without labels, and the caller
can skip the 2021 part instead of failing.

Usage:
    python scripts/download_asvspoof2021.py --out /tmp/asv21
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_asvspoof2019 import download  # noqa: E402
from adg import asvspoof2021  # noqa: E402
from adg.datasets import ASV21_LABELS_FILE, export_asv21_labels  # noqa: E402

AUDIO_URL = "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
AUDIO_MD5 = "2abee34d8b0b91159555fc4f016e4562"
KEYS_URL = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
KEYS_MD5 = "f052cc2ed276745afa3b5198665d3b26"
SUBSET = Path("results/m5/asv21_subset.csv")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str) -> None:
    if not (dest.exists() and md5(dest) == expected_md5):
        download(url, dest)
    got = md5(dest)
    if got != expected_md5:
        raise SystemExit(f"{dest.name}: MD5 {got}, expected {expected_md5}")
    print(f"{dest.name}: MD5 OK", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/asvspoof2021"))
    ap.add_argument("--n-per-class", type=int, default=400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys_tar = args.out / "LA-keys-full.tar.gz"
    try:
        fetch(KEYS_URL, keys_tar, KEYS_MD5)
    except (urllib.error.URLError, OSError) as e:
        print(f"keys not available ({e}); skipping ASVspoof 2021", file=sys.stderr)
        return 2
    with tarfile.open(keys_tar) as t:
        t.extractall(args.out, filter="data")

    audio_tar = args.out / "ASVspoof2021_LA_eval.tar.gz"
    if not (args.out / asvspoof2021.AUDIO).is_dir():
        fetch(AUDIO_URL, audio_tar, AUDIO_MD5)
        with tarfile.open(audio_tar) as t:
            t.extractall(args.out, filter="data")
        audio_tar.unlink()

    meta = asvspoof2021.read_metadata(args.out / asvspoof2021.METADATA)
    print(f"{len(meta)} trials, conditions: {sorted(meta['condition'].unique())}", flush=True)
    print(meta.groupby(["subset", "key"]).size().to_string(), flush=True)
    export_asv21_labels(args.out, args.out / ASV21_LABELS_FILE)

    sub = asvspoof2021.stratified_by_condition(meta, args.n_per_class, args.n_per_class, seed=0)
    missing = [u for u in sub["utt_id"] if not (args.out / asvspoof2021.AUDIO / f"{u}.flac").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} subset files missing from the audio archive, e.g. {missing[0]}")
    SUBSET.parent.mkdir(parents=True, exist_ok=True)
    sub[["utt_id", "condition", "key"]].to_csv(SUBSET, index=False)
    print(f"subset: {len(sub)} trials over {sub['condition'].nunique()} conditions -> {SUBSET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
