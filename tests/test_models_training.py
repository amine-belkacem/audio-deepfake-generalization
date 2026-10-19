"""Tests for the milestone 3 models and the training loop."""

import numpy as np
import pytest
import torch

from adg.metrics import compute_eer
from adg.models import LCNN, MaxFeatureMap, WeightedLayerMLP
from adg.training import (
    FrameCropDataset,
    PooledDataset,
    TrainConfig,
    chunked_mean_std,
    score,
    set_seed,
    train_model,
)


def test_max_feature_map_hand_computed():
    # channels [a, b, c, d] -> max(a, c), max(b, d)
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    got = MaxFeatureMap()(x)
    assert torch.equal(got, torch.tensor([[3.0, 5.0]])), f"ECHEC MFM : obtenu {got}, attendu [[3, 5]]"


def test_lcnn_accepts_any_length():
    model = LCNN().eval()
    for frames in (100, 400, 731):
        out = model(torch.randn(3, 60, frames))
        assert out.shape == (3,), f"ECHEC LCNN T={frames} : forme obtenue {tuple(out.shape)}, attendu (3,)"


def test_lcnn_parameter_count():
    # Pins the number quoted in the LCNN docstring and in the docs.
    n = sum(p.numel() for p in LCNN().parameters())
    assert n == 173_777, f"ECHEC taille LCNN : {n} paramètres, attendu 173777"


def test_layer_weights_start_uniform_and_sum_to_one():
    m = WeightedLayerMLP(n_layers=12, dim=8)
    w = m.layer_weights().detach()
    assert torch.allclose(w, torch.full((12,), 1 / 12)), f"ECHEC poids initiaux : obtenu {w}, attendu 1/12 partout"
    assert abs(float(w.sum()) - 1) < 1e-6, f"ECHEC somme des poids : obtenu {float(w.sum())}, attendu 1"


def test_mlp_rejects_wrong_layer_count():
    with pytest.raises(ValueError, match="expected"):
        WeightedLayerMLP(n_layers=12, dim=8)(torch.randn(2, 11, 8))


def test_chunked_mean_std_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(5003, 4, 3)).astype(np.float16)
    mean, std = chunked_mean_std(x, chunk=700)
    ref = x.astype(np.float64)
    assert np.allclose(mean, ref.mean(0), atol=1e-5), "ECHEC moyenne par blocs"
    assert np.allclose(std, ref.std(0), atol=1e-5), "ECHEC écart-type par blocs"


def test_short_utterance_is_tiled_hand_computed():
    # One utterance of 3 frames [0, 1, 2], crop of 7 frames -> 0 1 2 0 1 2 0
    frames = np.arange(3, dtype=np.float32).reshape(3, 1)
    ds = FrameCropDataset(frames, np.array([0]), np.array([3]), np.array([1]),
                          mean=np.zeros(1, np.float32), std=np.ones(1, np.float32), n_frames=7)
    got = ds.crop(0)[:, 0].tolist()
    assert got == [0, 1, 2, 0, 1, 2, 0], f"ECHEC répétition : obtenu {got}, attendu [0, 1, 2, 0, 1, 2, 0]"


def test_eval_crop_is_the_beginning_and_train_crop_stays_inside():
    frames = np.arange(10, dtype=np.float32).reshape(10, 1)
    kw = dict(starts=np.array([0]), counts=np.array([10]), labels=np.array([1]),
              mean=np.zeros(1, np.float32), std=np.ones(1, np.float32), n_frames=4)
    ev = FrameCropDataset(frames, **kw).crop(0)[:, 0].tolist()
    assert ev == [0, 1, 2, 3], f"ECHEC recadrage éval : obtenu {ev}, attendu [0, 1, 2, 3]"
    tr = FrameCropDataset(frames, train=True, seed=1, **kw)
    for _ in range(20):
        c = tr.crop(0)[:, 0]
        assert np.all(np.diff(c) == 1) and 0 <= c[0] <= 6, f"ECHEC recadrage aléatoire : obtenu {c}"


def toy_pooled(n_bona, n_spoof, seed):
    # Only layer 7 carries the class (shift 3 on dim 0); the other layers are noise.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_bona + n_spoof, 12, 8)).astype(np.float16)
    x[:n_bona, 7, 0] += 3.0
    y = np.r_[np.ones(n_bona), np.zeros(n_spoof)]
    return x, y


def test_mlp_learns_and_finds_the_useful_layer():
    xt, yt = toy_pooled(200, 1800, seed=0)
    xd, yd = toy_pooled(200, 1800, seed=1)
    mean, std = chunked_mean_std(xt)
    model = WeightedLayerMLP(n_layers=12, dim=8, hidden=16)
    res = train_model(model, PooledDataset(xt, yt, mean, std), PooledDataset(xd, yd, mean, std),
                      TrainConfig(epochs=15, lr=1e-2, batch_size=64, patience=15), verbose=False)
    # Shift 3 sigma on one dim -> theoretical EER Phi(-1.5) = 6.7 %; allow some slack.
    assert res.best_dev_eer < 0.10, f"ECHEC apprentissage MLP : EER dev obtenu {res.best_dev_eer:.3f}, attendu < 0.10"
    w = model.layer_weights().detach().numpy()
    assert int(np.argmax(w)) == 7, f"ECHEC couche utile : poids max sur la couche {int(np.argmax(w))}, attendu 7 ; poids {w.round(3)}"


def test_training_is_reproducible():
    xt, yt = toy_pooled(64, 192, seed=2)
    mean, std = chunked_mean_std(xt)
    cfg = TrainConfig(epochs=3, lr=1e-2, batch_size=32, seed=5)
    runs = []
    for _ in range(2):
        set_seed(cfg.seed)  # weight init happens at construction, before train_model seeds
        m = WeightedLayerMLP(n_layers=12, dim=8, hidden=8)
        train_model(m, PooledDataset(xt, yt, mean, std), PooledDataset(xt, yt, mean, std), cfg, verbose=False)
        runs.append(score(m, PooledDataset(xt, yt, mean, std)))
    assert np.array_equal(runs[0], runs[1]), "ECHEC reproductibilité : deux entraînements à graine égale diffèrent"


def test_lcnn_learns_a_spectral_difference():
    # Bona fide crops have +2 on feature rows 20-29 everywhere: an easy pattern
    # a CNN must pick up within a few epochs.
    rng = np.random.default_rng(3)
    n_utt, n_frames = 160, 40
    labels = np.r_[np.ones(40), np.zeros(120)]
    frames = rng.normal(size=(n_utt * n_frames, 60)).astype(np.float16)
    for i in range(40):
        frames[i * n_frames:(i + 1) * n_frames, 20:30] += 2.0
    starts = np.arange(n_utt) * n_frames
    counts = np.full(n_utt, n_frames)
    mean, std = np.zeros(60, np.float32), np.ones(60, np.float32)
    ds = FrameCropDataset(frames, starts, counts, labels, mean, std, n_frames=32, train=True)
    dev = FrameCropDataset(frames, starts, counts, labels, mean, std, n_frames=32)
    res = train_model(LCNN(), ds, dev, TrainConfig(epochs=8, lr=3e-3, batch_size=16, patience=8), verbose=False)
    assert res.best_dev_eer < 0.05, f"ECHEC apprentissage LCNN : EER obtenu {res.best_dev_eer:.3f}, attendu < 0.05"
