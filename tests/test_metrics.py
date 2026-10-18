"""Tests for the evaluation metrics.

Every expected value below is worked out by hand in the comments, so a reader
can check it without running anything.
"""

import numpy as np
import pytest

from adg.metrics import (
    AsvErrorRates,
    CostModel,
    asv_error_rates,
    bootstrap_eer,
    compute_eer,
    det_curve,
    min_tdcf_2019,
)

# Toy scores reused in several tests (higher = more bona fide).
BONA = [0.9, 0.8, 0.7, 0.3]
SPOOF = [0.6, 0.4, 0.2, 0.1]


def check_close(got, expected, tol=1e-9, what="value"):
    assert abs(got - expected) <= tol, f"ECHEC {what} : obtenu {got}, attendu {expected}"


def test_eer_hand_computed():
    # Accept when score >= t.
    # t = 0.6: bona fide rejected = {0.3}       -> FRR = 1/4 = 0.25
    #          spoof accepted     = {0.6}       -> FAR = 1/4 = 0.25
    # FRR == FAR, so EER = 0.25 at t = 0.6.
    eer, thr = compute_eer(BONA, SPOOF)
    check_close(eer, 0.25, what="EER")
    check_close(thr, 0.6, what="seuil EER")


def test_eer_perfect_separation_is_zero():
    # Every bona fide score is above every spoof score: at t = 0.7,
    # FRR = 0/3 and FAR = 0/3, so EER = 0.
    eer, _ = compute_eer([0.7, 0.8, 0.9], [0.1, 0.2, 0.3])
    check_close(eer, 0.0, what="EER sÃ©paration parfaite")


def test_eer_inverted_scores_is_one():
    # Bona fide [0.1, 0.2, 0.3], spoof [0.7, 0.8, 0.9]: the detector is
    # perfectly wrong. Walking the thresholds:
    #   t = 0.1: FRR = 0,   FAR = 1  -> |diff| = 1
    #   t = 0.2: FRR = 1/3, FAR = 1  -> |diff| = 2/3
    #   t = 0.3: FRR = 2/3, FAR = 1  -> |diff| = 1/3
    #   t = 0.7: FRR = 1,   FAR = 1  -> |diff| = 0   <- chosen, EER = 1
    eer, _ = compute_eer([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    check_close(eer, 1.0, what="EER scores inversÃ©s")


def test_eer_ties_are_not_split():
    # All four scores equal 0.5. Only two thresholds exist in practice:
    # t = 0.5 -> accept all: FRR = 0, FAR = 1
    # t = +inf -> reject all: FRR = 1, FAR = 0
    # |FRR - FAR| = 1 at both, the first one is kept: EER = (0 + 1) / 2 = 0.5.
    eer, _ = compute_eer([0.5, 0.5], [0.5, 0.5])
    check_close(eer, 0.5, what="EER avec scores Ã©gaux")


def _eer_brute_force(bona, spoof):
    """Independent reference: loop over candidate thresholds one by one."""
    bona, spoof = np.asarray(bona), np.asarray(spoof)
    candidates = sorted(set(bona) | set(spoof)) + [np.inf]
    best = None
    for t in candidates:
        frr = np.mean(bona < t)
        far = np.mean(spoof >= t)
        if best is None or abs(frr - far) < best[0]:
            best = (abs(frr - far), (frr + far) / 2)
    return best[1]


@pytest.mark.parametrize("seed", range(5))
def test_eer_matches_brute_force_on_random_scores(seed):
    rng = np.random.default_rng(seed)
    bona = rng.normal(1.0, 1.0, 300).round(2)   # rounding forces ties
    spoof = rng.normal(-1.0, 1.0, 900).round(2)
    got, _ = compute_eer(bona, spoof)
    expected = _eer_brute_force(bona, spoof)
    check_close(got, expected, what=f"EER alÃ©atoire (graine {seed})")


def test_det_curve_is_monotonic():
    frr, far, _ = det_curve(BONA, SPOOF)
    assert np.all(np.diff(frr) >= 0), f"ECHEC FRR non croissant : {frr}"
    assert np.all(np.diff(far) <= 0), f"ECHEC FAR non dÃ©croissant : {far}"
    check_close(frr[-1], 1.0, what="FRR au seuil +inf")
    check_close(far[-1], 0.0, what="FAR au seuil +inf")


@pytest.mark.parametrize("bad", [[], [0.1, np.nan], [np.inf, 0.2]])
def test_eer_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        compute_eer(bad, SPOOF)


def test_asv_error_rates_hand_computed():
    # Threshold 0.5, accept when score >= 0.5.
    # targets    [0.9, 0.6, 0.4]      -> 0.4 rejected     -> Pmiss       = 1/3
    # nontargets [0.1, 0.2, 0.7, 0.3] -> 0.7 accepted     -> Pfa         = 1/4
    # spoofs     [0.8, 0.2]           -> 0.2 rejected     -> Pmiss_spoof = 1/2
    r = asv_error_rates([0.9, 0.6, 0.4], [0.1, 0.2, 0.7, 0.3], [0.8, 0.2], 0.5)
    check_close(r.p_miss, 1 / 3, what="Pmiss ASV")
    check_close(r.p_fa, 1 / 4, what="Pfa ASV")
    check_close(r.p_miss_spoof, 1 / 2, what="Pmiss_spoof ASV")


def test_min_tdcf_hand_computed():
    # Cost model ASVspoof 2019:
    #   Ptar = 0.95 * 0.99 = 0.9405, Pnon = 0.95 * 0.01 = 0.0095, Pspoof = 0.05
    #   Cmiss_asv = 1, Cfa_asv = 10, Cmiss_cm = 1, Cfa_cm = 10
    # ASV rates chosen for the test: Pfa = 0.01, Pmiss = 0.02, Pmiss_spoof = 0.5
    #   C1 = 0.9405 * (1 - 1 * 0.02) - 0.0095 * 10 * 0.01 = 0.92169 - 0.00095 = 0.92074
    #   C2 = 10 * 0.05 * (1 - 0.5) = 0.25
    #   normalised t-DCF = (0.92074 * Pmiss_cm + 0.25 * Pfa_cm) / 0.25
    #                    = 3.68296 * Pmiss_cm + Pfa_cm
    # On the toy CM scores, the operating points (Pmiss_cm, Pfa_cm) are:
    #   t=-inf (0, 1)    -> 1.0
    #   t=0.1  (0, 1)    -> 1.0
    #   t=0.2  (0, 0.75) -> 0.75
    #   t=0.3  (0, 0.5)  -> 0.5      <- minimum
    #   t=0.4  (0.25, 0.5)  -> 1.42074
    #   t=0.6  (0.25, 0.25) -> 1.17074
    #   t=0.7  (0.25, 0)    -> 0.92074
    #   higher thresholds only increase Pmiss_cm.
    asv = AsvErrorRates(p_fa=0.01, p_miss=0.02, p_miss_spoof=0.5)
    tdcf, thr = min_tdcf_2019(BONA, SPOOF, asv)
    check_close(tdcf, 0.5, what="min t-DCF")
    check_close(thr, 0.3, what="seuil min t-DCF")


def test_min_tdcf_never_above_one():
    # Normalisation by min(C1, C2) means a CM that accepts or rejects
    # everything gets exactly 1, so the minimum can never exceed 1.
    rng = np.random.default_rng(0)
    asv = AsvErrorRates(p_fa=0.02, p_miss=0.03, p_miss_spoof=0.4)
    tdcf, _ = min_tdcf_2019(rng.normal(size=200), rng.normal(size=200), asv)
    assert tdcf <= 1.0 + 1e-12, f"ECHEC min t-DCF : obtenu {tdcf}, attendu <= 1"


def test_cost_model_rejects_bad_priors():
    with pytest.raises(ValueError):
        CostModel(p_target=0.5, p_nontarget=0.5, p_spoof=0.5)


def test_bootstrap_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    bona, spoof = rng.normal(1, 1, 500), rng.normal(-1, 1, 2000)
    a = bootstrap_eer(bona, spoof, n_resamples=200, seed=42)
    b = bootstrap_eer(bona, spoof, n_resamples=200, seed=42)
    assert a == b, f"ECHEC bootstrap non reproductible : {a} != {b}"
    assert a.low <= a.estimate <= a.high, (
        f"ECHEC intervalle : obtenu [{a.low}, {a.high}], estimation {a.estimate} hors intervalle")
    # Two unit-variance Gaussians 2 apart: theoretical EER = Phi(-1) = 0.1587.
    assert a.low < 0.1587 < a.high, (
        f"ECHEC intervalle : obtenu [{a.low:.4f}, {a.high:.4f}], attendu contenant 0.1587")
