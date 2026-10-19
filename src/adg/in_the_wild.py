"""Reading the In-the-Wild dataset (Müller et al., Interspeech 2022).

The archive layout is not documented on the download page, so this reader
makes as few assumptions as possible and fails loudly on anything unexpected:
it looks for exactly one metadata CSV, needs a file column and a label column,
and only accepts label spellings it knows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BONAFIDE_SPELLINGS = {"bona-fide", "bonafide", "bona fide", "bona_fide", "real", "genuine"}
SPOOF_SPELLINGS = {"spoof", "fake", "deepfake"}
FILE_COLUMNS = ("file", "filename", "path", "audio")
LABEL_COLUMNS = ("label", "key", "class")


class InTheWildError(ValueError):
    pass


def find_metadata(root: str | Path) -> Path:
    # Zips made on macOS carry "__MACOSX/._name" shadow files; they are not data.
    csvs = sorted(p for p in Path(root).rglob("*.csv")
                  if "__MACOSX" not in p.parts and not p.name.startswith("._"))
    if len(csvs) != 1:
        raise InTheWildError(f"expected exactly one metadata CSV under {root}, found {[str(c) for c in csvs]}")
    return csvs[0]


def _pick(columns, candidates, what):
    lowered = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise InTheWildError(f"no {what} column among {list(columns)} (tried {list(candidates)})")


def read_in_the_wild(root: str | Path) -> pd.DataFrame:
    """One row per file: utt_id, path, label (1 = bona fide), speaker if present.

    Paths in the CSV are resolved relative to the CSV's folder. Every file must
    exist; a partially extracted archive is an error, not a smaller test set.
    """
    meta_path = find_metadata(root)
    meta = pd.read_csv(meta_path)
    file_col = _pick(meta.columns, FILE_COLUMNS, "file")
    label_col = _pick(meta.columns, LABEL_COLUMNS, "label")

    raw = meta[label_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw) - BONAFIDE_SPELLINGS - SPOOF_SPELLINGS)
    if unknown:
        raise InTheWildError(f"unknown label values {unknown} in {meta_path.name}")

    paths = [meta_path.parent / str(f) for f in meta[file_col]]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise InTheWildError(f"{len(missing)} audio files listed in {meta_path.name} are missing, "
                             f"first: {missing[0]}")

    df = pd.DataFrame({
        "utt_id": [f"ITW_{Path(str(f)).stem}" for f in meta[file_col]],
        "path": [str(p) for p in paths],
        "label": raw.isin(BONAFIDE_SPELLINGS).astype("int8").to_numpy(),
    })
    if "speaker" in {c.lower() for c in meta.columns}:
        df["speaker"] = meta[_pick(meta.columns, ("speaker",), "speaker")].to_numpy()
    if df["utt_id"].duplicated().any():
        raise InTheWildError("duplicated file names in the metadata")
    # Same columns as an ASVspoof protocol, so the evaluation code works unchanged.
    df["attack"] = df["label"].map({1: "-", 0: "itw"})
    df["key"] = df["label"].map({1: "bonafide", 0: "spoof"})
    return df
