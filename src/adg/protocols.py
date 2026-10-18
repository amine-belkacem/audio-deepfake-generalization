"""Reading and checking the ASVspoof 2019 LA countermeasure (CM) protocols.

A protocol line looks like:
    LA_0039 LA_E_2834763 - A11 spoof
    speaker utterance    - attack key
The attack field is "-" for bona fide trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BONAFIDE = "bonafide"
SPOOF = "spoof"
PROTOCOL_COLUMNS = ["speaker", "utt_id", "unused", "attack", "key"]

PARTITION_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

TRAIN_ATTACKS = frozenset(f"A{i:02d}" for i in range(1, 7))
EVAL_ATTACKS = frozenset(f"A{i:02d}" for i in range(7, 20))
# A16 and A19 reuse the algorithms of A04 and A06 (training attacks), so on the
# eval set they are the only "known" generators. Everything else is unseen.
KNOWN_EVAL_ATTACKS = frozenset({"A16", "A19"})
UNKNOWN_EVAL_ATTACKS = EVAL_ATTACKS - KNOWN_EVAL_ATTACKS


@dataclass(frozen=True)
class PartitionCounts:
    bonafide: int
    spoof_per_attack: int
    attacks: frozenset[str]

    @property
    def spoof(self) -> int:
        return self.spoof_per_attack * len(self.attacks)

    @property
    def total(self) -> int:
        return self.bonafide + self.spoof


# Official counts from the ASVspoof 2019 evaluation plan. If a download is
# truncated or the wrong protocol file is picked up, check_counts() catches it.
EXPECTED_COUNTS = {
    "train": PartitionCounts(bonafide=2580, spoof_per_attack=3800, attacks=TRAIN_ATTACKS),
    "dev": PartitionCounts(bonafide=2548, spoof_per_attack=3716, attacks=TRAIN_ATTACKS),
    "eval": PartitionCounts(bonafide=7355, spoof_per_attack=4914, attacks=EVAL_ATTACKS),
}


class ProtocolError(ValueError):
    """Raised when a protocol file does not match the expected format or counts."""


def read_protocol(path: str | Path) -> pd.DataFrame:
    """Load one CM protocol file into a DataFrame (one row per trial)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=PROTOCOL_COLUMNS,
                     dtype=str, engine="python")
    if df.isna().any().any():
        bad = df[df.isna().any(axis=1)].index[0] + 1
        raise ProtocolError(f"{path.name}: line {bad} does not have 5 fields")

    bad_keys = set(df["key"]) - {BONAFIDE, SPOOF}
    if bad_keys:
        raise ProtocolError(f"{path.name}: unexpected keys {sorted(bad_keys)}")
    if df["utt_id"].duplicated().any():
        dup = df.loc[df["utt_id"].duplicated(), "utt_id"].iloc[0]
        raise ProtocolError(f"{path.name}: duplicated utterance id {dup}")

    df = df.drop(columns="unused")
    df["label"] = (df["key"] == BONAFIDE).astype("int8")  # 1 = bona fide
    return df


def load_partition(protocol_dir: str | Path, partition: str) -> pd.DataFrame:
    if partition not in PARTITION_FILES:
        raise ValueError(f"unknown partition {partition!r}, expected one of {sorted(PARTITION_FILES)}")
    return read_protocol(Path(protocol_dir) / PARTITION_FILES[partition])


def check_counts(df: pd.DataFrame, partition: str) -> None:
    """Compare a loaded partition with the official counts. Raises on mismatch."""
    expected = EXPECTED_COUNTS[partition]
    errors = []

    n_bona = int((df["key"] == BONAFIDE).sum())
    if n_bona != expected.bonafide:
        errors.append(f"bona fide: got {n_bona}, expected {expected.bonafide}")

    spoof_attacks = df.loc[df["key"] == SPOOF, "attack"].value_counts()
    if set(spoof_attacks.index) != expected.attacks:
        errors.append(f"attacks: got {sorted(spoof_attacks.index)}, expected {sorted(expected.attacks)}")
    for attack, count in spoof_attacks.items():
        if count != expected.spoof_per_attack:
            errors.append(f"{attack}: got {count} trials, expected {expected.spoof_per_attack}")

    if errors:
        raise ProtocolError(f"{partition} partition does not match the official counts: " + "; ".join(errors))


def check_disjoint_speakers(*partitions: pd.DataFrame) -> None:
    """ASVspoof 2019 LA uses disjoint speakers across partitions. Leakage here
    would make every result optimistic, so it is checked explicitly."""
    seen: dict[str, int] = {}
    for i, df in enumerate(partitions):
        for spk in set(df["speaker"]):
            if spk in seen and seen[spk] != i:
                raise ProtocolError(f"speaker {spk} appears in partitions {seen[spk]} and {i}")
            seen[spk] = i
