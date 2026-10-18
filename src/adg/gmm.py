"""Diagonal-covariance GMM trained with chunked EM.

Why not sklearn.mixture.GaussianMixture on the full data: its E-step keeps an
(n_frames x n_components) responsibility matrix. For the spoof class that is
about 7.5 M x 512 doubles = 30 GB. Here the E-step runs chunk by chunk and only
the sufficient statistics (N_k, sum z, sum z^2) are accumulated, so memory stays
at O(chunk x components). sklearn is still used, on a random subsample, to get a
sensible starting point.

Frames are standardised (z = (x - offset) / scale) before anything else. The
fast E-step expands (z - mu)^2 into z^2 - 2 z mu + mu^2; with raw LFCCs (c0 is
large) that difference of big float32 numbers loses most of its precision.
Log-likelihoods are reported in the original x space by adding the Jacobian
term -sum(log scale), so two models with different standardisations remain
directly comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class DiagGMM:
    weights: np.ndarray    # (K,)
    means: np.ndarray      # (K, D), in standardised space
    variances: np.ndarray  # (K, D), in standardised space
    offset: np.ndarray     # (D,)
    scale: np.ndarray      # (D,)
    history: list[float] = field(default_factory=list)  # mean log-lik (z space) per EM iteration

    @property
    def n_components(self) -> int:
        return self.weights.shape[0]

    def standardise(self, frames: np.ndarray) -> torch.Tensor:
        z = (np.asarray(frames, dtype=np.float64) - self.offset) / self.scale
        return torch.from_numpy(z.astype(np.float32))

    def _torch_params(self):
        inv_var = torch.from_numpy(1.0 / self.variances).float()
        mu_inv_var = torch.from_numpy(self.means / self.variances).float()
        # Constant part of log(w_k N(z | mu_k, Sigma_k)), one value per component.
        const = (np.log(self.weights)
                 - 0.5 * (self.means.shape[1] * LOG_2PI + np.log(self.variances).sum(1)
                          + (self.means ** 2 / self.variances).sum(1)))
        return inv_var, mu_inv_var, torch.from_numpy(const).float()

    @staticmethod
    def _component_log_probs(z: torch.Tensor, params) -> torch.Tensor:
        """log(w_k N(z_n | k)) for a chunk, shape (n, K), as two matrix products."""
        inv_var, mu_inv_var, const = params
        return const - 0.5 * (z * z) @ inv_var.T + z @ mu_inv_var.T

    def frame_log_likelihood(self, frames: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """log p(x_n) in the ORIGINAL feature space, for every frame, shape (n,)."""
        params = self._torch_params()
        log_jacobian = -float(np.log(self.scale).sum())
        out = np.empty(frames.shape[0], dtype=np.float64)
        with torch.inference_mode():
            for i in range(0, frames.shape[0], chunk):
                z = self.standardise(frames[i:i + chunk])
                out[i:i + chunk] = torch.logsumexp(self._component_log_probs(z, params), 1).double().numpy()
        return out + log_jacobian

    def save(self, path: str | Path) -> None:
        np.savez(path, weights=self.weights, means=self.means, variances=self.variances,
                 offset=self.offset, scale=self.scale, history=np.array(self.history))

    @classmethod
    def load(cls, path: str | Path) -> "DiagGMM":
        d = np.load(path)
        return cls(d["weights"], d["means"], d["variances"], d["offset"], d["scale"],
                   list(d["history"]))


def _global_stats(frames: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = frames.shape
    s1, s2 = np.zeros(d), np.zeros(d)
    for i in range(0, n, chunk):
        x = np.asarray(frames[i:i + chunk], dtype=np.float64)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean, std


def fit_gmm(frames: np.ndarray, n_components: int = 512, n_iter: int = 10,
            n_init_frames: int = 200_000, var_floor: float = 1e-3,
            chunk: int = 200_000, seed: int = 0, verbose: bool = True) -> DiagGMM:
    """Fit a diagonal GMM on all frames (array or memmap, shape (n, D)).

    var_floor is expressed in standardised units (the global variance is 1),
    so 1e-3 means "no component narrower than 0.1 % of the global variance".
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    n, d = frames.shape
    if n < n_components:
        raise ValueError(f"{n} frames is not enough for {n_components} components")

    offset, scale = _global_stats(frames, chunk)

    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(n, size=min(n_init_frames, n), replace=False))
    sub = ((np.asarray(frames[pick], dtype=np.float64) - offset) / scale)
    init = GaussianMixture(n_components, covariance_type="diag", max_iter=20,
                           init_params="k-means++", random_state=seed, reg_covar=var_floor).fit(sub)
    gmm = DiagGMM(init.weights_.copy(), init.means_.copy(),
                  np.maximum(init.covariances_, var_floor), offset, scale)

    for it in range(n_iter):
        params = gmm._torch_params()
        n_k = torch.zeros(n_components, dtype=torch.float64)
        s1 = torch.zeros(n_components, d, dtype=torch.float64)
        s2 = torch.zeros(n_components, d, dtype=torch.float64)
        total_ll = 0.0
        with torch.inference_mode():
            for i in range(0, n, chunk):
                z = gmm.standardise(frames[i:i + chunk])
                logp = gmm._component_log_probs(z, params)
                ll = torch.logsumexp(logp, 1, keepdim=True)
                resp = torch.exp(logp - ll)  # posterior of each component, (n, K)
                total_ll += float(ll.double().sum())
                n_k += resp.sum(0).double()
                s1 += (resp.T @ z).double()
                s2 += (resp.T @ (z * z)).double()

        n_k_np, s1_np, s2_np = n_k.numpy(), s1.numpy(), s2.numpy()
        # A component that owns almost no data gets its old parameters back
        # instead of a 0/0 update.
        alive = n_k_np > 1e-3 * n / n_components
        means, variances = gmm.means.copy(), gmm.variances.copy()
        means[alive] = s1_np[alive] / n_k_np[alive, None]
        variances[alive] = s2_np[alive] / n_k_np[alive, None] - means[alive] ** 2
        variances = np.maximum(variances, var_floor)
        weights = np.where(alive, n_k_np, gmm.weights * n)
        gmm = DiagGMM(weights / weights.sum(), means, variances, offset, scale,
                      gmm.history + [total_ll / n])
        if verbose:
            print(f"  EM {it + 1}/{n_iter}: mean log-lik {total_ll / n:.4f}, "
                  f"inactive components {int((~alive).sum())}", flush=True)
    return gmm


def utterance_llr(frames: np.ndarray, starts: np.ndarray, counts: np.ndarray,
                  bona: DiagGMM, spoof: DiagGMM, chunk: int = 200_000) -> np.ndarray:
    """Per-utterance score: mean over frames of log p(x|bona) - log p(x|spoof).

    The frames of a partition are stored contiguously, so the frame scores are
    computed in one pass and then averaged per utterance with np.add.reduceat.
    """
    starts, counts = np.asarray(starts), np.asarray(counts)
    if np.any(counts <= 0):
        raise ValueError("every utterance must have at least one frame")
    if starts[0] != 0 or np.any(starts[1:] != starts[:-1] + counts[:-1]):
        raise ValueError("utterances must be contiguous and in index order")
    if starts[-1] + counts[-1] != frames.shape[0]:
        raise ValueError(f"index covers {starts[-1] + counts[-1]} frames, array has {frames.shape[0]}")
    frame_llr = bona.frame_log_likelihood(frames, chunk) - spoof.frame_log_likelihood(frames, chunk)
    return np.add.reduceat(frame_llr, starts) / counts
