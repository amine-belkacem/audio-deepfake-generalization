"""ASVspoof 2021 LA evaluation set: real telephony and VoIP transmissions.

Trial metadata (keys/LA/CM/trial_metadata.txt in LA-keys-full.tar.gz), one line
per trial, whitespace separated:
    speaker trial codec transmission attack key trim subset
    LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
The audio is in ASVspoof2021_LA_eval/flac/<trial>.flac (16 kHz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["speaker", "utt_id", "codec", "transmission", "attack", "key", "trim", "subset"]
METADATA = Path("keys") / "LA" / "CM" / "trial_metadata.txt"
AUDIO = Path("ASVspoof2021_LA_eval") / "flac"


def read_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, dtype=str)
    if df.isna().any().any():
        raise ValueError(f"{path}: some lines do not have {len(COLUMNS)} fields")
    bad = set(df["key"]) - {"bonafide", "spoof"}
    if bad:
        raise ValueError(f"{path}: unexpected keys {sorted(bad)}")
    df["label"] = (df["key"] == "bonafide").astype("int8")
    df["condition"] = df["codec"] + "/" + df["transmission"]
    return df


def stratified_by_condition(df: pd.DataFrame, n_bona: int, n_spoof: int, seed: int = 0,
                            subset: str = "eval") -> pd.DataFrame:
    """Up to n_bona bona fide and n_spoof spoofed trials per codec/transmission condition.

    Only trials of the given scoring subset ("eval" in the 2021 keys) are used.
    Equal counts per condition keep every channel visible in the results.
    """
    rng = np.random.default_rng(seed)
    pool = df[df["subset"] == subset]
    if pool.empty:
        raise ValueError(f"no trial with subset == {subset!r}")
    keep = []
    for _, group in pool.groupby("condition", sort=True):
        for label, n in ((1, n_bona), (0, n_spoof)):
            idx = group.index[group["label"] == label].to_numpy()
            keep.append(rng.choice(idx, min(n, idx.size), replace=False))
    return df.loc[np.sort(np.concatenate(keep))]
