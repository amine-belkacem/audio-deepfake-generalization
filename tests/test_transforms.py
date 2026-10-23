"""Tests for the named audio transforms and their use in feature extraction."""

import numpy as np
import pytest
import soundfile as sf
import torch

from adg.features import LfccExtractor, extract_files, open_partition
from adg.transforms import apply, folder_name, validate

SR = 16000


def tone_with_silence(lead_s, tone_s, trail_s):
    t = np.arange(int(SR * tone_s)) / SR
    return np.concatenate([np.zeros(int(SR * lead_s)), 0.5 * np.sin(2 * np.pi * 440 * t),
                           np.zeros(int(SR * trail_s))]).astype(np.float32)


@pytest.mark.parametrize("spec", ["none", "trim40", "band:1000-2000"])
def test_valid_specs(spec):
    assert validate(spec) == spec, f"ECHEC spec valide refusée : {spec}"


@pytest.mark.parametrize("spec", ["trim", "band:1000", "lowpass", "trim-40"])
def test_invalid_specs(spec):
    with pytest.raises(ValueError):
        validate(spec)


def test_folder_name():
    got = folder_name("band:1000-2000")
    assert got == "band1000-2000", f"ECHEC nom de dossier : obtenu {got}, attendu band1000-2000"


def test_trim_removes_the_silence():
    # 0.5 s + 1 s + 0.3 s: about 1 s left, within one 20 ms frame on each side.
    x = torch.from_numpy(tone_with_silence(0.5, 1.0, 0.3))
    y = apply(x, SR, "trim40")
    assert abs(y.numel() - SR) <= 2 * 320, f"ECHEC découpe : {y.numel()} échantillons, attendu {SR} ± 640"


def test_band_keeps_the_length():
    x = torch.from_numpy(tone_with_silence(0.1, 0.5, 0.1))
    y = apply(x, SR, "band:3000-4000")
    assert y.shape == x.shape, f"ECHEC longueur après masquage : {tuple(y.shape)}, attendu {tuple(x.shape)}"


@pytest.mark.parametrize("workers", [1, 2])
def test_lfcc_extraction_on_trimmed_audio(tmp_path, workers):
    # One file: 0.5 s silence, 1 s tone, 0.5 s silence (2 s). Without trimming:
    # 32000 // 160 + 1 = 201 frames. Trimmed to ~16000 samples: ~101 frames.
    p = tmp_path / "a.flac"
    sf.write(p, tone_with_silence(0.5, 1.0, 0.5), SR)
    extract_files([p], ["a"], tmp_path / "full", LfccExtractor(), progress=False, workers=workers)
    extract_files([p], ["a"], tmp_path / "trim", LfccExtractor(), progress=False, workers=workers,
                  transform="trim40")
    full = open_partition(tmp_path / "full")[1]["n_frames"].iloc[0]
    trimmed = open_partition(tmp_path / "trim")[1]["n_frames"].iloc[0]
    assert full == 201, f"ECHEC sans découpe : {full} trames, attendu 201"
    assert abs(trimmed - 101) <= 4, f"ECHEC avec découpe : {trimmed} trames, attendu 101 ± 4"
