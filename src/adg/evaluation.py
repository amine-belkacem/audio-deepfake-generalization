"""Turning a score file into the numbers reported in the README.

A score file is a CSV with one row per eval trial: utt_id, score. It is joined
with the protocol so that every metric can be broken down by attack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import AsvErrorRates, asv_error_rates, bootstrap_eer, compute_eer, min_tdcf_2019
from .protocols import KNOWN_EVAL_ATTACKS, UNKNOWN_EVAL_ATTACKS

ASV_SCORE_FILE = Path("ASVspoof2019_LA_asv_scores") / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
ASV_KEYS = {"target", "nontarget", "spoof"}


def load_asv_rates(la_root: str | Path) -> AsvErrorRates:
    """ASV error rates of the organisers' ASV system, at its own EER threshold.

    This is how the official t-DCF script sets the ASV operating point: the ASV
    threshold is the EER threshold between target and non-target trials.
    File format (one trial per line): <speaker> <source> <key> <score>,
    with key in {target, nontarget, spoof}.
    """
    path = Path(la_root) / ASV_SCORE_FILE
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["speaker", "source", "key", "score"])
    if set(df["key"]) != ASV_KEYS:
        raise ValueError(f"{path.name}: keys {sorted(set(df['key']))}, expected {sorted(ASV_KEYS)}")
    tar = df.loc[df["key"] == "target", "score"].to_numpy()
    non = df.loc[df["key"] == "nontarget", "score"].to_numpy()
    spf = df.loc[df["key"] == "spoof", "score"].to_numpy()
    _, threshold = compute_eer(tar, non)
    return asv_error_rates(tar, non, spf, threshold)


def join_scores(scores: pd.DataFrame, protocol: pd.DataFrame) -> pd.DataFrame:
    merged = protocol.merge(scores, on="utt_id", how="left", validate="one_to_one")
    missing = int(merged["score"].isna().sum())
    if missing:
        raise ValueError(f"{missing} protocol trials have no score")
    return merged


def evaluate(merged: pd.DataFrame, asv: AsvErrorRates, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per condition: all trials, known attacks, unknown attacks, each attack.

    For a subset, the bona fide trials are always all of them; only the spoof
    side is restricted. That is the standard way to report per-attack EER.
    """
    bona = merged.loc[merged["label"] == 1, "score"].to_numpy()
    spoof = merged[merged["label"] == 0]

    groups = {"pooled": spoof,
              "known (A16, A19)": spoof[spoof["attack"].isin(KNOWN_EVAL_ATTACKS)],
              "unknown (11 attacks)": spoof[spoof["attack"].isin(UNKNOWN_EVAL_ATTACKS)]}
    for attack in sorted(spoof["attack"].unique()):
        groups[attack] = spoof[spoof["attack"] == attack]

    rows = []
    for name, sub in groups.items():
        s = sub["score"].to_numpy()
        boot = bootstrap_eer(bona, s, n_resamples=n_boot, seed=seed)
        tdcf, _ = min_tdcf_2019(bona, s, asv)
        rows.append({"condition": name, "n_spoof": s.size,
                     "eer_pct": 100 * boot.estimate, "eer_low_pct": 100 * boot.low,
                     "eer_high_pct": 100 * boot.high, "min_tdcf": tdcf})
    return pd.DataFrame(rows)
