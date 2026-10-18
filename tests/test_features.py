"""Tests for the LFCC front-end and the frame storage."""

import math

import numpy as np
import pytest
import soundfile as sf
import torch

from adg.features import LfccConfig, LfccExtractor, extract_partition, load_audio, open_partition

SR = 16000


def sine(seconds, freq=440.0, amp=0.3):
    t = torch.arange(int(SR * seconds)) / SR
    return (amp * torch.sin(2 * math.pi * freq * t)).float()


def test_frame_geometry_hand_computed():
    cfg = LfccConfig()
    # 20 ms at 16 kHz = 320 samples, 10 ms = 160 samples.
    assert cfg.win_length == 320, f"ECHEC fenêtre : obtenu {cfg.win_length}, attendu 320"
    assert cfg.hop_length == 160, f"ECHEC pas : obtenu {cfg.hop_length}, attendu 160"
    # 20 static + 20 delta + 20 delta-delta.
    assert cfg.dim == 60, f"ECHEC dimension : obtenu {cfg.dim}, attendu 60"
    # Centred STFT: floor(n / hop) + 1 frames. 1 s -> 16000 // 160 + 1 = 101.
    assert cfg.n_frames(16000) == 101, f"ECHEC nb trames : obtenu {cfg.n_frames(16000)}, attendu 101"


def test_extractor_shape_and_determinism():
    ext = LfccExtractor()
    a, b = ext(sine(1.0)), ext(sine(1.0))
    assert a.shape == (101, 60), f"ECHEC forme : obtenu {a.shape}, attendu (101, 60)"
    assert np.all(np.isfinite(a)), "ECHEC : valeurs non finies dans les LFCC"
    assert np.array_equal(a, b), "ECHEC : deux extractions du même signal diffèrent"


def test_gain_only_moves_c0():
    # Doubling the amplitude multiplies every filter energy by 4, i.e. adds
    # ln(4) to each of the 20 log energies. torchaudio uses an orthonormal DCT-II,
    # whose first basis vector is 1/sqrt(20) everywhere, so:
    #   c0 shift = 20 * ln(4) / sqrt(20) = 2 ln(2) sqrt(20) = 6.1993
    # The other basis vectors sum to zero, so c1..c19 do not move, and the
    # deltas of a constant shift are zero, so dims 20..59 do not move either.
    # White noise, not a sine: torchaudio takes log(E + 1e-6), and with a pure
    # tone the empty high bands sit on that 1e-6 floor and do not shift with
    # gain (first version of this test used a sine and measured 2.09, not 6.20).
    ext = LfccExtractor()
    x = 0.1 * torch.randn(SR, generator=torch.Generator().manual_seed(0))
    ref, loud = ext(x), ext(2.0 * x)
    expected_shift = 2 * math.log(2) * math.sqrt(20)
    got_shift = float(np.mean(loud[:, 0] - ref[:, 0]))
    assert abs(got_shift - expected_shift) < 1e-3, (
        f"ECHEC décalage c0 : obtenu {got_shift:.5f}, attendu {expected_shift:.5f}")
    worst = float(np.max(np.abs(loud[:, 1:] - ref[:, 1:])))
    assert worst < 1e-3, f"ECHEC autres coefficients : écart max obtenu {worst:.2e}, attendu < 1e-3"


def test_wrong_sample_rate_is_rejected(tmp_path):
    p = tmp_path / "x.flac"
    sf.write(p, np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="sample rate 8000"):
        load_audio(p, SR)


def test_extract_partition_roundtrip(tmp_path):
    audio = tmp_path / "flac"
    audio.mkdir()
    lengths = {"u1": 0.5, "u2": 1.0, "u3": 0.25}
    for i, (utt, sec) in enumerate(lengths.items()):
        sf.write(audio / f"{utt}.flac", sine(sec, freq=300 + 100 * i).numpy(), SR)

    ext = LfccExtractor()
    extract_partition(list(lengths), audio, tmp_path / "feats", ext, progress=False)
    frames, index = open_partition(tmp_path / "feats")

    # 8000 // 160 + 1 = 51, 16000 // 160 + 1 = 101, 4000 // 160 + 1 = 26
    assert list(index["n_frames"]) == [51, 101, 26], f"ECHEC nb trames : obtenu {list(index['n_frames'])}, attendu [51, 101, 26]"
    assert list(index["start"]) == [0, 51, 152], f"ECHEC débuts : obtenu {list(index['start'])}, attendu [0, 51, 152]"
    assert frames.shape == (178, 60), f"ECHEC forme : obtenu {frames.shape}, attendu (178, 60)"

    # Stored as float16: about 3 significant digits, so compare with rtol 1e-3.
    direct = ext(load_audio(audio / "u2.flac", SR))
    stored = np.asarray(frames[51:152], dtype=np.float32)
    assert np.allclose(stored, direct, rtol=1e-3, atol=1e-2), (
        f"ECHEC stockage float16 : écart max {np.max(np.abs(stored - direct)):.3e}")


def test_first_run_preset_matches_torchaudio_lfcc():
    # Independent reference: torchaudio's own LFCC transform, which is what the
    # first milestone-1 run used (Hann window, f_min = 0, log(E + 1e-6)).
    import torchaudio

    from adg.features import PRESETS

    cfg = PRESETS["first_run"]
    ref_tf = torchaudio.transforms.LFCC(
        sample_rate=SR, n_filter=20, n_lfcc=20, log_lf=True,
        speckwargs={"n_fft": 512, "win_length": 320, "hop_length": 160, "center": True})
    x = 0.1 * torch.randn(SR, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        ref = ref_tf(x).T.numpy()  # (n_frames, 20), static part only
    got = LfccExtractor(cfg)(x)[:, :20]
    worst = float(np.max(np.abs(got - ref)))
    assert worst < 1e-3, f"ECHEC comparaison torchaudio : écart max obtenu {worst:.2e}, attendu < 1e-3"


def test_b02_preset_settings():
    from adg.features import PRESETS

    cfg = PRESETS["asvspoof2019_b02"]
    got = (cfg.window, cfg.f_min, cfg.n_filter, cfg.n_lfcc, cfg.win_ms, cfg.hop_ms, cfg.dim)
    expected = ("hamming", 30.0, 20, 20, 20.0, 10.0, 60)
    assert got == expected, f"ECHEC preset B02 : obtenu {got}, attendu {expected}"


def test_parallel_extraction_matches_sequential(tmp_path):
    audio = tmp_path / "flac"
    audio.mkdir()
    utts = [f"u{i}" for i in range(6)]
    for i, utt in enumerate(utts):
        sf.write(audio / f"{utt}.flac", sine(0.3 + 0.1 * i, freq=200 + 50 * i).numpy(), SR)

    ext = LfccExtractor()
    extract_partition(utts, audio, tmp_path / "seq", ext, progress=False, workers=1)
    extract_partition(utts, audio, tmp_path / "par", ext, progress=False, workers=2)
    seq, _ = open_partition(tmp_path / "seq")
    par, _ = open_partition(tmp_path / "par")
    # Not bit-exact: workers run torch with 1 thread, so float sums are done in a
    # different order and a few values round to the neighbouring float16. The
    # first run of this test measured a max gap of 2.4e-4 (relative 1.4e-3), i.e.
    # one float16 step (2^-10 ~ 9.8e-4 relative). Allow at most two steps.
    seq, par = seq.astype(np.float32), par.astype(np.float32)
    rel = float(np.max(np.abs(seq - par) / np.maximum(np.abs(seq), 1e-2)))
    assert rel < 2 * 2.0 ** -10, (
        f"ECHEC extraction parallèle : écart relatif max obtenu {rel:.2e}, attendu < {2 * 2.0 ** -10:.2e}")


def test_missing_audio_is_reported(tmp_path):
    (tmp_path / "flac").mkdir()
    with pytest.raises(FileNotFoundError, match="1 audio files missing"):
        extract_partition(["nope"], tmp_path / "flac", tmp_path / "f", LfccExtractor(), progress=False)
