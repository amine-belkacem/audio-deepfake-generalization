"""One place that knows where the labels of every evaluation set come from."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .in_the_wild import read_in_the_wild
from .protocols import load_partition

ASVSPOOF_PARTITIONS = ("train", "dev", "eval")
LABELS_FILE = "itw_labels.csv"
LABEL_COLUMNS = ["utt_id", "attack", "key", "label"]


def export_itw_labels(itw_root: str | Path, out_path: str | Path) -> int:
    """Write the In-the-Wild label table (no audio, no paths) for use without the audio."""
    df = read_in_the_wild(itw_root)[LABEL_COLUMNS]
    df.to_csv(out_path, index=False)
    return len(df)


def load_labels(partition: str, la_root: str | Path = "data/LA",
                itw_root: str | Path = "data/in_the_wild") -> pd.DataFrame:
    """Protocol-like table (utt_id, attack, key, label) for an ASVspoof partition or 'itw'."""
    if partition in ASVSPOOF_PARTITIONS:
        return load_partition(Path(la_root) / "ASVspoof2019_LA_cm_protocols", partition)
    if partition == "itw":
        # On my laptop only the label table exported from Kaggle exists, not the audio.
        exported = Path(itw_root) / LABELS_FILE
        if exported.is_file():
            df = pd.read_csv(exported, dtype={"utt_id": str, "attack": str, "key": str})
            if set(df["key"]) - {"bonafide", "spoof"} or (df["label"] != (df["key"] == "bonafide")).any():
                raise ValueError(f"{exported}: inconsistent key / label columns")
            return df
        return read_in_the_wild(itw_root)
    raise ValueError(f"unknown partition {partition!r}")


def load_labels_from_index(partition: str, feat_utt_ids: list[str], la_root, itw_root) -> pd.DataFrame:
    """Labels aligned with the row order of a feature file; raises if they differ."""
    labels = load_labels(partition, la_root, itw_root)
    if labels["utt_id"].tolist() != list(feat_utt_ids):
        raise ValueError(f"{partition}: feature rows are not in the same order as the labels")
    return labels
