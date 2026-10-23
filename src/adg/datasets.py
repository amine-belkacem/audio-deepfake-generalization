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


def base_partition(name: str) -> str:
    """'eval' -> 'eval', 'eval_trim40' -> 'eval', 'evalsub_band1000-2000' -> 'eval'."""
    head = name.split("_", 1)[0]
    return head[:-3] if head.endswith("sub") else head


def load_labels_from_index(partition: str, feat_utt_ids: list[str], la_root, itw_root) -> pd.DataFrame:
    """Labels in the row order of a feature file.

    A full partition must match the protocol order exactly. A derived one
    (subset and / or transform, e.g. 'evalsub_band1000-2000') may hold any subset
    of its base partition, in any order, but every id must exist exactly once.
    """
    labels = load_labels(base_partition(partition), la_root, itw_root)
    ids = list(feat_utt_ids)
    if partition in (*ASVSPOOF_PARTITIONS, "itw"):
        if labels["utt_id"].tolist() != ids:
            raise ValueError(f"{partition}: feature rows are not in the same order as the labels")
        return labels
    if len(set(ids)) != len(ids):
        raise ValueError(f"{partition}: duplicated utterance ids in the feature index")
    unknown = set(ids) - set(labels["utt_id"])
    if unknown:
        raise ValueError(f"{partition}: {len(unknown)} ids not in {base_partition(partition)}, e.g. {sorted(unknown)[:3]}")
    return labels.set_index("utt_id").loc[ids].reset_index()
