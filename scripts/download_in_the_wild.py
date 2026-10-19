"""Download In-the-Wild (Müller et al., 2022) from Hugging Face, unzip it and describe it.

About 8.2 GB. Licence: CC-BY-SA 4.0 on the Hugging Face card (the Fraunhofer
project page has also stated Apache 2.0); only scores and metrics derived from
it are published here, never the audio.

Usage:
    python scripts/download_in_the_wild.py --out data/in_the_wild
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_asvspoof2019 import download, sha256  # noqa: E402
from adg.in_the_wild import find_metadata, read_in_the_wild  # noqa: E402

URL = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / "release_in_the_wild.zip"

    if not any(args.out.rglob("*.csv")):
        download(URL, zip_path)
        print(f"sha256(release_in_the_wild.zip) = {sha256(zip_path)}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(args.out)
        if not args.keep_zip:
            zip_path.unlink()

    meta = find_metadata(args.out)
    print(f"metadata file: {meta}")
    print(meta.read_text(encoding="utf-8", errors="replace").splitlines()[:4])
    df = read_in_the_wild(args.out)
    print(f"{len(df)} files: {int(df['label'].sum())} bona fide, {int((1 - df['label']).sum())} spoof")
    if "speaker" in df:
        print(f"{df['speaker'].nunique()} speakers")
    rates = {}
    for p in df["path"].sample(min(200, len(df)), random_state=0):
        info = sf.info(p)
        rates[(info.samplerate, info.channels)] = rates.get((info.samplerate, info.channels), 0) + 1
    print(f"(sample rate, channels) on 200 random files: {rates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
