"""The two trained classifiers of milestone 3.

Both output one logit per utterance, positive = bona fide, the same orientation as
every score in this project.
"""

from __future__ import annotations

import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """Split dim 1 (channels, or features for 2-D input) in two halves and keep the max.

    This is the activation of Light CNN (Wu et al., 2018), used by the LCNN
    countermeasure of Lavrentyeva et al. (ASVspoof 2019). It halves the channels
    and acts as a learned feature selector.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2:
            raise ValueError(f"MaxFeatureMap needs an even channel count, got {x.shape[1]}")
        a, b = x.chunk(2, dim=1)
        return torch.maximum(a, b)


def _mfm_conv(c_in: int, c_out: int, k: int) -> nn.Sequential:
    # conv to 2 * c_out channels, MFM brings it back to c_out
    return nn.Sequential(nn.Conv2d(c_in, 2 * c_out, k, padding=k // 2), MaxFeatureMap())


class LCNN(nn.Module):
    """Light CNN on (batch, n_features, n_frames) LFCC input.

    A reduced version of the ASVspoof 2019 LCNN: same block structure (MFM
    convolutions, 1x1 bottlenecks, max pooling), half the channels of the
    original, and an average over the remaining time axis instead of a flatten,
    so it accepts any length. 173,777 parameters (counted, see the tests).
    """

    def __init__(self, n_features: int = 60, dropout: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(
            _mfm_conv(1, 32, 5), nn.MaxPool2d(2),
            _mfm_conv(32, 32, 1), nn.BatchNorm2d(32), _mfm_conv(32, 48, 3), nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            _mfm_conv(48, 48, 1), nn.BatchNorm2d(48), _mfm_conv(48, 64, 3), nn.MaxPool2d(2),
            _mfm_conv(64, 64, 1), nn.BatchNorm2d(64), _mfm_conv(64, 32, 3), nn.BatchNorm2d(32),
            _mfm_conv(32, 32, 1), nn.BatchNorm2d(32), _mfm_conv(32, 32, 3), nn.MaxPool2d(2),
        )
        freq_after = n_features // 16  # four 2x2 poolings
        if freq_after < 1:
            raise ValueError(f"n_features={n_features} is too small for four poolings")
        self.head = nn.Sequential(
            nn.Linear(32 * freq_after, 2 * 80), MaxFeatureMap(), nn.BatchNorm1d(80),
            nn.Dropout(dropout), nn.Linear(80, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, features, frames), got {tuple(x.shape)}")
        h = self.body(x.unsqueeze(1))       # (B, C, F', T')
        h = h.mean(dim=3).flatten(1)        # average over time -> (B, C * F')
        return self.head(h).squeeze(1)


class WeightedLayerMLP(nn.Module):
    """MLP on a learned softmax-weighted sum of the pooled wav2vec 2.0 layers.

    Input (batch, n_layers, dim). The layer weights are a softmax over n_layers
    free parameters, so they stay positive, sum to 1 and can be read directly
    as "how much the classifier uses each layer".
    """

    def __init__(self, n_layers: int = 12, dim: int = 1536, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))  # start from a uniform average
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.layer_logits.numel():
            raise ValueError(f"expected (batch, {self.layer_logits.numel()}, dim), got {tuple(x.shape)}")
        mixed = torch.einsum("l,bld->bd", self.layer_weights(), x)
        return self.mlp(mixed).squeeze(1)
