"""Tests for band occlusion and the stratified subset."""

import numpy as np
import pandas as pd
import pytest

from adg.bands import BANDS_HZ, mask_band, stratified_subset

SR = 16000
# The first and last 384 samples are affected by the STFT reflection padding
# (see src/adg/bands.py); interior checks skip 512 samples on each side.
EDGE = 512


def sine(f, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * f * t)


def rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def test_bands_cover_0_to_8_khz():
    assert BANDS_HZ[0] == (0, 1000) and BANDS_HZ[-1] == (7000, 8000) and len(BANDS_HZ) == 8, (
        f"ECHEC bandes : obtenu {BANDS_HZ}")


def test_tone_inside_the_band_is_removed():
    # 1500 Hz tone, band 1-2 kHz removed: the residual must be at least 40 dB down.
    # Bin spacing is 16000 / 512 = 31.25 Hz, the tone sits on bin 48, far from the band edges.
    # First version measured over the whole signal and got 35.2 dB: the edges
    # dominated the residual. In the interior it is ~143 dB.
    x = sine(1500)
    y = mask_band(x, SR, 1000, 2000)
    drop = rms_db(x[EDGE:-EDGE]) - rms_db(y[EDGE:-EDGE])
    assert drop > 40, f"ECHEC atténuation dans la bande : obtenu {drop:.1f} dB, attendu > 40 dB"


def test_tone_outside_the_band_is_kept():
    # 3000 Hz tone, band 1-2 kHz removed: STFT/iSTFT with a Hann window at 75 %
    # overlap reconstructs perfectly, so the tone must come back almost unchanged.
    x = sine(3000)
    y = mask_band(x, SR, 1000, 2000)
    err = np.max(np.abs(x - y)[EDGE:-EDGE])
    assert err < 1e-5, f"ECHEC hors bande : écart max obtenu {err:.2e}, attendu < 1e-5"


def test_edge_effect_stays_within_384_samples():
    # Pins the measured extent of the edge effect quoted in the module docstring.
    x = sine(3000)
    err = np.abs(x - mask_band(x, SR, 4000, 5000))
    assert err[384:-384].max() < 1e-5, f"ECHEC effet de bord : écart {err[384:-384].max():.2e} au-delà de 384 échantillons"


def test_length_is_preserved():
    x = sine(440, seconds=0.737)
    y = mask_band(x, SR, 0, 1000)
    assert y.shape == x.shape, f"ECHEC longueur : obtenu {y.shape}, attendu {x.shape}"


def test_top_band_includes_nyquist():
    # A tone at exactly 8 kHz is the Nyquist bin: masking 7-8 kHz must remove it.
    x = 0.5 * np.cos(np.pi * np.arange(SR))  # +0.5, -0.5, ... = 8 kHz
    drop = rms_db(x[EDGE:-EDGE]) - rms_db(mask_band(x, SR, 7000, 8000)[EDGE:-EDGE])
    assert drop > 40, f"ECHEC bin de Nyquist : atténuation {drop:.1f} dB, attendu > 40 dB"


def test_band_outside_nyquist_is_rejected():
    with pytest.raises(ValueError):
        mask_band(sine(440), SR, 7000, 9000)


def test_stratified_subset_counts_hand_computed():
    # 50 bona fide, attacks A07 x 40 and A08 x 30; ask 10 bona + 5 per attack -> 20 rows.
    proto = pd.DataFrame({
        "utt_id": [f"u{i}" for i in range(120)],
        "label": [1] * 50 + [0] * 70,
        "attack": ["-"] * 50 + ["A07"] * 40 + ["A08"] * 30,
    })
    sub = stratified_subset(proto, n_bona=10, n_per_attack=5, seed=0)
    counts = sub.groupby("attack").size().to_dict()
    assert counts == {"-": 10, "A07": 5, "A08": 5}, f"ECHEC effectifs : obtenu {counts}, attendu -:10, A07:5, A08:5"
    assert sub.index.is_monotonic_increasing, "ECHEC : l'ordre du protocole doit être conservé"
    again = stratified_subset(proto, n_bona=10, n_per_attack=5, seed=0)
    assert sub.index.equals(again.index), "ECHEC : même graine, sous-ensemble différent"
