"""Tests for the chunked-EM diagonal GMM."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from adg.gmm import DiagGMM, fit_gmm, utterance_llr


def two_clusters(n=6000, seed=0):
    # 30 % of frames around (-5, -5, -5), 70 % around (+5, +5, +5), unit variance.
    rng = np.random.default_rng(seed)
    n_a = int(0.3 * n)
    a = rng.normal(-5.0, 1.0, (n_a, 3))
    b = rng.normal(+5.0, 1.0, (n - n_a, 3))
    x = np.vstack([a, b])
    return x[rng.permutation(n)]


def test_frame_log_likelihood_matches_scipy():
    # Independent reference: scipy densities in the ORIGINAL space. A component
    # with standardised mean mu and variance v corresponds, in x space, to mean
    # offset + scale * mu and variance scale^2 * v.
    rng = np.random.default_rng(3)
    gmm = DiagGMM(weights=np.array([0.25, 0.75]),
                  means=np.array([[0.0, 1.0], [-1.0, 0.5]]),
                  variances=np.array([[1.0, 0.5], [2.0, 0.3]]),
                  offset=np.array([10.0, -3.0]), scale=np.array([2.0, 0.5]))
    x = rng.normal(size=(50, 2)) * gmm.scale + gmm.offset

    per_comp = []
    for k in range(2):
        mean_x = gmm.offset + gmm.scale * gmm.means[k]
        std_x = gmm.scale * np.sqrt(gmm.variances[k])
        per_comp.append(np.log(gmm.weights[k]) + norm.logpdf(x, mean_x, std_x).sum(1))
    expected = logsumexp(np.stack(per_comp, 1), axis=1)

    got = gmm.frame_log_likelihood(x, chunk=7)  # odd chunk size on purpose
    worst = float(np.max(np.abs(got - expected)))
    assert worst < 1e-4, f"ECHEC log-vraisemblance : écart max obtenu {worst:.2e}, attendu < 1e-4"


def test_em_never_decreases_likelihood():
    # EM guarantees a non-decreasing likelihood. A small tolerance covers float32
    # rounding and the variance floor.
    gmm = fit_gmm(two_clusters(), n_components=4, n_iter=8, n_init_frames=2000, verbose=False)
    steps = np.diff(gmm.history)
    assert np.all(steps > -1e-5), f"ECHEC EM : la log-vraisemblance baisse, pas obtenus {steps}"


def test_em_recovers_two_clusters():
    gmm = fit_gmm(two_clusters(), n_components=2, n_iter=10, n_init_frames=2000, verbose=False)
    means_x = gmm.offset + gmm.scale * gmm.means
    order = np.argsort(means_x[:, 0])
    got_means = means_x[order, 0]
    got_weights = gmm.weights[order]
    assert np.allclose(got_means, [-5.0, 5.0], atol=0.1), f"ECHEC moyennes : obtenu {got_means}, attendu [-5, 5]"
    assert np.allclose(got_weights, [0.3, 0.7], atol=0.02), f"ECHEC poids : obtenu {got_weights}, attendu [0.3, 0.7]"


def test_chunk_size_does_not_change_the_fit():
    x = two_clusters()
    a = fit_gmm(x, n_components=3, n_iter=3, n_init_frames=2000, chunk=500, verbose=False)
    b = fit_gmm(x, n_components=3, n_iter=3, n_init_frames=2000, chunk=100_000, verbose=False)
    worst = float(np.max(np.abs(a.means - b.means)))
    assert worst < 1e-4, f"ECHEC invariance par bloc : écart max des moyennes {worst:.2e}, attendu < 1e-4"


def single_gaussian(mean):
    return DiagGMM(np.array([1.0]), np.array([[mean]]), np.array([[1.0]]),
                   offset=np.zeros(1), scale=np.ones(1))


def test_utterance_llr_hand_computed():
    # bona = N(0, 1), spoof = N(1, 1). Per frame:
    #   llr(x) = -x^2/2 + (x - 1)^2/2 = 1/2 - x
    # utterance 1 = frames [0, 1] -> (0.5 + (-0.5)) / 2 = 0
    # utterance 2 = frame  [2]    -> 0.5 - 2 = -1.5
    frames = np.array([[0.0], [1.0], [2.0]])
    got = utterance_llr(frames, np.array([0, 2]), np.array([2, 1]), single_gaussian(0.0), single_gaussian(1.0))
    assert np.allclose(got, [0.0, -1.5], atol=1e-6), f"ECHEC LLR : obtenu {got}, attendu [0, -1.5]"


def test_utterance_llr_rejects_gaps_in_index():
    frames = np.zeros((4, 1))
    with pytest.raises(ValueError, match="contiguous"):
        utterance_llr(frames, np.array([0, 3]), np.array([2, 1]), single_gaussian(0.0), single_gaussian(1.0))
