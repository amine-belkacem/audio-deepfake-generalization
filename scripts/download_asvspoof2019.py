"""Download ASVspoof 2019 LA from Edinburgh DataShare, unzip it and check the protocols.

Usage:
    python scripts/download_asvspoof2019.py --out data

The zip is about 7.1 GB. The download resumes if interrupted, and the SHA-256 of
the zip is printed so it can be recorded in docs/DATA.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.protocols import check_counts, check_disjoint_speakers, load_partition  # noqa: E402

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
CHUNK_BYTES = 1 << 20
PROTOCOL_SUBDIR = Path("LA") / "ASVspoof2019_LA_cm_protocols"


USER_AGENT = "audio-deepfake-generalization/0.1 (research download; github.com/amine-belkacem)"
RETRY_CODES = {403, 429, 500, 502, 503, 504}


def open_with_retry(req, attempts: int = 5, first_wait_s: float = 30.0,
                    opener=urllib.request.urlopen, sleep=time.sleep):
    """urlopen with exponential back-off (30 s, 60 s, 120 s, ...).

    Added after a Kaggle run got "HTTP 403 Forbidden" from DataShare while the
    same URL answered normally from my laptop: the refusal was temporary, and
    one failed request should not cost a whole GPU session.
    """
    for k in range(attempts):
        try:
            return opener(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or k == attempts - 1:
                raise
            reason = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if k == attempts - 1:
                raise
            reason = str(e.reason)
        wait = first_wait_s * 2 ** k
        print(f"{reason}, retry {k + 2}/{attempts} in {wait:.0f} s", flush=True)
        sleep(wait)
    raise RuntimeError("unreachable")


def download(url: str, dest: Path) -> None:
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"
    with open_with_retry(urllib.request.Request(url, headers=headers)) as resp:
        if done and resp.status != 206:
            print("server ignored the resume request, starting over")
            done = 0
        total = done + int(resp.headers.get("Content-Length", 0))
        with open(dest, "ab" if done else "wb") as f:
            while chunk := resp.read(CHUNK_BYTES):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--keep-zip", action="store_true", help="do not delete LA.zip after extraction")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / "LA.zip"
    protocol_dir = args.out / PROTOCOL_SUBDIR

    if not protocol_dir.exists():
        download(URL, zip_path)
        print(f"sha256(LA.zip) = {sha256(zip_path)}")
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"corrupted member in zip: {bad}", file=sys.stderr)
                return 1
            zf.extractall(args.out)
        if not args.keep_zip:
            zip_path.unlink()

    partitions = {p: load_partition(protocol_dir, p) for p in ("train", "dev", "eval")}
    for name, df in partitions.items():
        check_counts(df, name)
        print(f"{name}: {len(df)} trials, counts match the evaluation plan")
    check_disjoint_speakers(*partitions.values())
    print("speakers are disjoint across partitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
