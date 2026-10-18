"""Evaluation metrics: EER, ASVspoof 2019 min t-DCF and bootstrap confidence intervals.

Score convention used everywhere in this project: a HIGHER score means
"more likely bona fide". A trial is accepted as bona fide when score >= threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ASVspoof 2019 cost model, taken from the official evaluation plan (section on t-DCF).
SPOOF_PRIOR = 0.05
ASV_TARGET_SHARE = 0.99


@dataclass(frozen=True)
class CostModel:
    p_target: float = (1.0 - SPOOF_PRIOR) * ASV_TARGET_SHARE
    p_nontarget: float = (1.0 - SPOOF_PRIOR) * (1.0 - ASV_TARGET_SHARE)
    p_spoof: float = SPOOF_PRIOR
    c_miss_asv: float = 1.0
    c_fa_asv: float = 10.0
    c_miss_cm: float = 1.0
    c_fa_cm: float = 10.0

    def __post_init__(self) -> None:
        priors = (self.p_target, self.p_nontarget, self.p_spoof)
        if any(p < 0 for p in priors) or abs(sum(priors) - 1.0) > 1e-10:
            raise ValueError(f"priors must be non-negative and sum to 1, got {priors}")
        costs = (self.c_miss_asv, self.c_fa_asv, self.c_miss_cm, self.c_fa_cm)
        if any(c < 0 for c in costs):
            raise ValueError(f"costs must be non-negative, got {costs}")


ASVSPOOF2019_COSTS = CostModel()


@dataclass(frozen=True)
class AsvErrorRates:
    p_fa: float          # zero-effort impostors accepted by the ASV system
    p_miss: float        # genuine targets rejected by the ASV system
    p_miss_spoof: float  # spoofed trials rejected by the ASV system


def _as_scores(name: str, scores) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def det_curve(bonafide_scores, spoof_scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Miss rate (FRR) and false-alarm rate (FAR) at every distinct threshold.

    Thresholds are the distinct observed scores plus +inf, so tied scores are
    handled correctly (a tie never gets split between accept and reject).
    Returns (frr, far, thresholds), with frr increasing and far decreasing.
    """
    bona = np.sort(_as_scores("bonafide_scores", bonafide_scores))
    spoof = np.sort(_as_scores("spoof_scores", spoof_scores))

    thresholds = np.concatenate([np.unique(np.concatenate([bona, spoof])), [np.inf]])
    # side="left" counts scores strictly below the threshold, i.e. rejected trials.
    frr = np.searchsorted(bona, thresholds, side="left") / bona.size
    far = 1.0 - np.searchsorted(spoof, thresholds, side="left") / spoof.size
    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores) -> tuple[float, float]:
    """Equal error rate (as a fraction, not a percentage) and its threshold.

    Same convention as the official ASVspoof scripts: take the operating point
    where |FRR - FAR| is smallest and report the mean of the two rates there.
    """
    frr, far, thresholds = det_curve(bonafide_scores, spoof_scores)
    idx = int(np.argmin(np.abs(frr - far)))
    return float((frr[idx] + far[idx]) / 2.0), float(thresholds[idx])


def asv_error_rates(target_scores, nontarget_scores, spoof_scores, threshold: float) -> AsvErrorRates:
    """ASV error rates at a fixed threshold (accept when score >= threshold)."""
    tar = _as_scores("target_scores", target_scores)
    non = _as_scores("nontarget_scores", nontarget_scores)
    spf = _as_scores("spoof_scores", spoof_scores)
    return AsvErrorRates(
        p_fa=float(np.mean(non >= threshold)),
        p_miss=float(np.mean(tar < threshold)),
        p_miss_spoof=float(np.mean(spf < threshold)),
    )


def min_tdcf_2019(bonafide_cm, spoof_cm, asv: AsvErrorRates,
                  costs: CostModel = ASVSPOOF2019_COSTS) -> tuple[float, float]:
    """Minimum normalised t-DCF, ASVspoof 2019 definition.

    t-DCF(s) = C1 * Pmiss_cm(s) + C2 * Pfa_cm(s), normalised by min(C1, C2) so
    that a CM which accepts or rejects everything scores 1.
    Returns (min_tdcf, cm_threshold_at_min).
    """
    c1 = (costs.p_target * (costs.c_miss_cm - costs.c_miss_asv * asv.p_miss)
          - costs.p_nontarget * costs.c_fa_asv * asv.p_fa)
    c2 = costs.c_fa_cm * costs.p_spoof * (1.0 - asv.p_miss_spoof)
    if c1 <= 0 or c2 <= 0:
        raise ValueError(f"t-DCF weights must be positive, got C1={c1}, C2={c2}; "
                         "check the ASV error rates")

    frr, far, thresholds = det_curve(bonafide_cm, spoof_cm)
    # The curve from det_curve starts at the lowest observed score. Add the
    # "accept everything" point (Pmiss = 0, Pfa = 1) so the curve is complete.
    frr = np.concatenate([[0.0], frr])
    far = np.concatenate([[1.0], far])
    thresholds = np.concatenate([[-np.inf], thresholds])

    tdcf_norm = (c1 * frr + c2 * far) / min(c1, c2)
    idx = int(np.argmin(tdcf_norm))
    return float(tdcf_norm[idx]), float(thresholds[idx])


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    n_resamples: int


def bootstrap_eer(bonafide_scores, spoof_scores, n_resamples: int = 1000,
                  confidence: float = 0.95, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI on the EER.

    Bona fide and spoof trials are resampled separately, which keeps the class
    ratio fixed. ASVspoof is heavily imbalanced (about 1 bona fide for 9 spoofs),
    and a pooled resample would let that ratio drift between draws.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    bona = _as_scores("bonafide_scores", bonafide_scores)
    spoof = _as_scores("spoof_scores", spoof_scores)
    rng = np.random.default_rng(seed)

    eers = np.empty(n_resamples)
    for i in range(n_resamples):
        eers[i], _ = compute_eer(rng.choice(bona, bona.size), rng.choice(spoof, spoof.size))

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(eers, [alpha, 1.0 - alpha])
    estimate, _ = compute_eer(bona, spoof)
    return BootstrapResult(estimate, float(low), float(high), n_resamples)
