"""Check that a copy of ASVspoof 2019 LA matches the official release I downloaded.

Why: on Kaggle, Edinburgh DataShare answered HTTP 403 to every request, so
milestone 3 uses a community mirror attached as a Kaggle input. A third-party
copy is only usable if it is byte-identical to the original, at least on what
can be checked cheaply.

    --make   on my laptop (official download): write the reference file
    default  anywhere else: compare a copy against that reference, exit 1 on mismatch

The reference holds SHA-256 digests of the protocol files, the organisers' ASV
score file and 60 audio files (20 per partition, fixed seed). This is a sample,
not a proof: together with the protocol counts and the existence check of all
121,461 files it makes a silently altered copy very unlikely, nothing more.

Usage:
    python scripts/verify_asvspoof_copy.py --la-root data/LA --make
    python scripts/verify_asvspoof_copy.py --la-root /kaggle/input/.../LA
    python scripts/verify_asvspoof_copy.py --find /kaggle/input   # print the LA root of a mounted copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.protocols import PARTITION_FILES, check_counts, check_disjoint_speakers, load_partition  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "asvspoof2019_la_sha256.json"
AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}
PROTO = Path("ASVspoof2019_LA_cm_protocols")
ASV_SCORES = Path("ASVspoof2019_LA_asv_scores") / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
N_AUDIO_PER_PARTITION = 20
SEED = 2019


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def find_la_root(search_root: Path) -> Path:
    """The folder that contains ASVspoof2019_LA_cm_protocols/<train protocol>."""
    hits = sorted(search_root.rglob(PARTITION_FILES["train"]))
    hits = [h for h in hits if h.parent.name == PROTO.name]
    if len(hits) != 1:
        raise SystemExit(f"expected one LA train protocol under {search_root}, found {[str(h) for h in hits]}")
    return hits[0].parent.parent


def checked_files(la_root: Path) -> dict[str, Path]:
    """Relative name -> path, for every file covered by the reference.

    Names use "/" on every OS (as_posix): the reference is made on Windows and
    checked on Linux, and the first version, keyed with str(Path), would have
    matched nothing on Kaggle.
    """
    files = {(PROTO / name).as_posix(): la_root / PROTO / name for name in PARTITION_FILES.values()}
    files[ASV_SCORES.as_posix()] = la_root / ASV_SCORES
    rng = np.random.default_rng(SEED)
    for part, audio_dir in AUDIO_DIRS.items():
        ids = load_partition(la_root / PROTO, part)["utt_id"].to_numpy()
        for u in sorted(rng.choice(ids, N_AUDIO_PER_PARTITION, replace=False)):
            rel = Path(audio_dir) / "flac" / f"{u}.flac"
            files[rel.as_posix()] = la_root / rel
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path)
    ap.add_argument("--find", type=Path, help="print the LA root found under this folder and exit")
    ap.add_argument("--make", action="store_true")
    args = ap.parse_args()

    if args.find is not None:
        print(find_la_root(args.find))
        return 0
    if args.la_root is None:
        ap.error("--la-root is required")

    partitions = {p: load_partition(args.la_root / PROTO, p) for p in AUDIO_DIRS}
    for p, df in partitions.items():
        check_counts(df, p)
    check_disjoint_speakers(*partitions.values())
    missing = [u for p, df in partitions.items() for u in df["utt_id"]
               if not (args.la_root / AUDIO_DIRS[p] / "flac" / f"{u}.flac").is_file()]
    if missing:
        print(f"MISMATCH: {len(missing)} audio files missing, first {missing[0]}")
        return 1
    print(f"protocol counts OK, speakers disjoint, all {sum(map(len, partitions.values()))} audio files present")

    files = checked_files(args.la_root)
    digests = {rel: sha256(path) for rel, path in files.items()}
    if args.make:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps({"source": "official LA.zip, doi:10.7488/ds/2555", "seed": SEED,
                                         "sha256": digests}, indent=1))
        print(f"reference with {len(digests)} digests written to {REFERENCE}")
        return 0

    ref = json.loads(REFERENCE.read_text())["sha256"]
    bad = [rel for rel in ref if digests.get(rel) != ref[rel]]
    if bad or set(digests) != set(ref):
        print(f"MISMATCH on {len(bad)} of {len(ref)} files, first: {bad[:3]}")
        return 1
    print(f"all {len(ref)} SHA-256 digests match the official release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
