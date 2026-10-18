"""Tests for the layer-wise linear probes."""

import numpy as np
import pandas as pd
import pytest

from adg.metrics import compute_eer
from adg.probes import fit_layer_probe, load_split, probe_scores


def synthetic(n_bona, n_spoof, shift, dim=8, seed=0):
    """Bona fide ~ N(shift, 1) on dim 0, spoof ~ N(0, 1); other dims are noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_bona + n_spoof, dim)).astype(np.float32)
    x[:n_bona, 0] += shift
    y = np.r_[np.ones(n_bona, int), np.zeros(n_spoof, int)]
    return x, y


def test_probe_separates_informative_layer():
    # Shift of 6 standard deviations on one dim: the classes barely overlap,
    # the linear probe should reach an EER close to 0.
    xt, yt = synthetic(200, 1800, shift=6.0, seed=1)
    xd, yd = synthetic(200, 1800, shift=6.0, seed=2)
    probe = fit_layer_probe(0, xt, yt, xd, yd)
    assert probe.dev_eer < 0.01, f"ECHEC couche informative : EER dev obtenu {probe.dev_eer:.3f}, attendu < 0.01"


def test_probe_is_at_chance_on_noise_layer():
    # No shift at all: nothing to learn, EER must stay around 50 %.
    xt, yt = synthetic(300, 2700, shift=0.0, seed=3)
    xd, yd = synthetic(300, 2700, shift=0.0, seed=4)
    probe = fit_layer_probe(0, xt, yt, xd, yd)
    assert 0.4 < probe.dev_eer < 0.6, f"ECHEC couche bruit : EER dev obtenu {probe.dev_eer:.3f}, attendu entre 0.4 et 0.6"


def test_scores_are_oriented_bona_fide_high():
    # Theoretical EER for a 2-sigma shift on one dim: Phi(-1) = 0.1587. A probe
    # with flipped sign would give ~0.84 instead.
    xt, yt = synthetic(500, 4500, shift=2.0, seed=5)
    xe, ye = synthetic(2000, 2000, shift=2.0, seed=6)
    probe = fit_layer_probe(0, xt, yt, xt, yt)
    s = probe_scores(probe.model, xe)
    eer, _ = compute_eer(s[ye == 1], s[ye == 0])
    assert abs(eer - 0.1587) < 0.02, f"ECHEC orientation des scores : EER obtenu {eer:.4f}, attendu 0.1587 ± 0.02"


def test_chosen_c_comes_from_the_grid():
    xt, yt = synthetic(100, 900, shift=3.0, seed=7)
    probe = fit_layer_probe(0, xt, yt, xt, yt, c_grid=(0.01, 1.0))
    assert probe.c in (0.01, 1.0), f"ECHEC choix de C : obtenu {probe.c}, attendu 0.01 ou 1.0"


def test_load_split_checks_row_order(tmp_path):
    np.save(tmp_path / "dev.npy", np.zeros((2, 12, 4), dtype=np.float16))
    pd.DataFrame({"utt_id": ["a", "b"]}).to_csv(tmp_path / "dev_utt_ids.csv", index=False)
    protocol = pd.DataFrame({"utt_id": ["b", "a"]})
    with pytest.raises(ValueError, match="protocol order"):
        load_split(tmp_path, "dev", protocol)


def test_load_split_refuses_unfinished_extraction(tmp_path):
    np.save(tmp_path / "dev.npy", np.zeros((2, 12, 4), dtype=np.float16))
    np.save(tmp_path / "dev.done.npy", np.array([True, False]))
    pd.DataFrame({"utt_id": ["a", "b"]}).to_csv(tmp_path / "dev_utt_ids.csv", index=False)
    with pytest.raises(ValueError, match="not finished"):
        load_split(tmp_path, "dev", pd.DataFrame({"utt_id": ["a", "b"]}))
