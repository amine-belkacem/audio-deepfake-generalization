"""Tests for the score-file evaluation (join with protocol, per-attack breakdown)."""

import numpy as np
import pandas as pd
import pytest

from adg.evaluation import ASV_SCORE_FILE, evaluate, join_scores, load_asv_rates
from adg.metrics import AsvErrorRates


def test_load_asv_rates_hand_computed(tmp_path):
    # targets [3, 2, 1], nontargets [0, -1, 2.5]
    # The ASV EER threshold is t = 2: FRR = 1/3 (1 < 2), FAR = 1/3 (2.5 >= 2).
    # spoofs [2.2, 0.5]: at t = 2, 0.5 is rejected -> Pmiss_spoof = 1/2.
    path = tmp_path / ASV_SCORE_FILE
    path.parent.mkdir(parents=True)
    rows = [("LA_0001", "bonafide", "target", s) for s in (3, 2, 1)]
    rows += [("LA_0001", "bonafide", "nontarget", s) for s in (0, -1, 2.5)]
    rows += [("LA_0001", "A07", "spoof", s) for s in (2.2, 0.5)]
    path.write_text("\n".join(" ".join(map(str, r)) for r in rows) + "\n")

    r = load_asv_rates(tmp_path)
    for name, got, expected in [("Pmiss", r.p_miss, 1 / 3), ("Pfa", r.p_fa, 1 / 3),
                                ("Pmiss_spoof", r.p_miss_spoof, 1 / 2)]:
        assert abs(got - expected) < 1e-12, f"ECHEC {name} ASV : obtenu {got}, attendu {expected}"


def toy_merged():
    # 4 bona fide at 0.9, 0.8, 0.7, 0.3; A07 is easy (all spoofs low),
    # A16 overlaps with bona fide.
    return pd.DataFrame({
        "utt_id": [f"u{i}" for i in range(8)],
        "attack": ["-", "-", "-", "-", "A07", "A07", "A16", "A16"],
        "label": [1, 1, 1, 1, 0, 0, 0, 0],
        "score": [0.9, 0.8, 0.7, 0.3, 0.1, 0.2, 0.85, 0.75],
    })


def test_per_attack_eer_hand_computed():
    # A07 vs bona fide: spoof {0.1, 0.2} all below bona fide {0.3..0.9} -> EER 0.
    # A16 vs bona fide: spoof {0.85, 0.75}.
    #   t = 0.8: FRR = 2/4 (0.3, 0.7), FAR = 1/2 (0.85) -> |diff| = 0 -> EER = 50 %.
    asv = AsvErrorRates(p_fa=0.01, p_miss=0.02, p_miss_spoof=0.5)
    table = evaluate(toy_merged(), asv, n_boot=20).set_index("condition")
    for cond, expected in [("A07", 0.0), ("A16", 50.0)]:
        got = table.loc[cond, "eer_pct"]
        assert abs(got - expected) < 1e-9, f"ECHEC EER {cond} : obtenu {got}, attendu {expected}"
    assert table.loc["known (A16, A19)", "n_spoof"] == 2, "ECHEC : A16 doit compter comme attaque connue"
    assert table.loc["unknown (11 attacks)", "n_spoof"] == 2, "ECHEC : A07 doit compter comme attaque inconnue"


def test_missing_scores_are_rejected():
    merged = toy_merged()
    scores = merged[["utt_id", "score"]].iloc[:-1]  # drop one trial
    with pytest.raises(ValueError, match="1 protocol trials have no score"):
        join_scores(scores, merged.drop(columns="score"))
