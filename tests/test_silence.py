"""Tests for silence measurement and trimming, on synthetic signals with known silence."""

import numpy as np
import pytest

from adg.silence import SpeechBounds, frame_range, frame_rms_db, speech_bounds, trim

SR = 16000
FRAME = 320   # 20 ms
HOP = 160     # 10 ms


def tone_with_silence(lead_s, tone_s, trail_s, noise_db=None, seed=0):
    """lead_s of silence, a 440 Hz tone of amplitude 0.5, trail_s of silence.

    noise_db: optional white noise floor, in dB relative to the tone RMS.
    """
    t = np.arange(int(SR * tone_s)) / SR
    x = np.concatenate([np.zeros(int(SR * lead_s)), 0.5 * np.sin(2 * np.pi * 440 * t), np.zeros(int(SR * trail_s))])
    if noise_db is not None:
        tone_rms = 0.5 / np.sqrt(2)
        x = x + np.random.default_rng(seed).normal(0, tone_rms * 10 ** (noise_db / 20), x.size)
    return x


def test_rms_db_of_a_full_scale_sine_hand_computed():
    # RMS of a sine of amplitude 1 is 1/sqrt(2): 10*log10(0.5) = -3.01 dB.
    t = np.arange(SR) / SR
    db = frame_rms_db(np.sin(2 * np.pi * 1000 * t), SR)
    assert np.allclose(db, -3.0103, atol=0.01), f"ECHEC RMS sinus : obtenu {db[:3]}, attendu -3.01 dB"
    # 1 s, 20 ms frames, 10 ms hop: 1 + (16000 - 320) // 160 = 99 frames
    assert db.size == 99, f"ECHEC nb trames : obtenu {db.size}, attendu 99"


def test_silence_lengths_hand_computed():
    # 0.5 s + 1 s tone + 0.3 s: lead 8000 samples, trail 4800 samples.
    # Detection works on frames of 20 ms with a 10 ms hop, so the edges are
    # only known to within one frame (320 samples).
    b = speech_bounds(tone_with_silence(0.5, 1.0, 0.3), SR)
    assert abs(b.lead_samples - 8000) <= FRAME, f"ECHEC silence début : obtenu {b.lead_samples}, attendu 8000 ± {FRAME}"
    assert abs(b.trail_samples - 4800) <= FRAME, f"ECHEC silence fin : obtenu {b.trail_samples}, attendu 4800 ± {FRAME}"


def test_noise_floor_below_threshold_is_still_silence():
    # Noise 50 dB under the tone, threshold 40 dB: the noisy edges count as silence.
    b = speech_bounds(tone_with_silence(0.4, 1.0, 0.4, noise_db=-50), SR, top_db=40)
    assert abs(b.lead_samples - 6400) <= FRAME, f"ECHEC bruit -50 dB : silence début {b.lead_samples}, attendu 6400 ± {FRAME}"


def test_noise_floor_above_threshold_is_not_trimmed():
    # Noise only 30 dB under the tone, threshold 40 dB: the edges are not silent
    # by this definition, so (almost) nothing is removed.
    b = speech_bounds(tone_with_silence(0.4, 1.0, 0.4, noise_db=-30), SR, top_db=40)
    assert b.lead_samples <= FRAME, f"ECHEC bruit -30 dB : silence début {b.lead_samples}, attendu <= {FRAME}"


def test_inner_pause_is_kept():
    # tone, 0.5 s pause, tone: only the edges may be trimmed.
    x = np.concatenate([tone_with_silence(0.2, 0.5, 0.5), tone_with_silence(0, 0.5, 0.2)])
    b = speech_bounds(x, SR)
    kept = (b.end - b.start) / SR
    assert abs(kept - 1.5) < 0.03, f"ECHEC pause interne : durée gardée {kept:.3f} s, attendu 1.5 s"


def test_digital_silence_keeps_the_whole_file():
    b = speech_bounds(np.zeros(SR), SR)
    assert (b.start, b.end) == (0, SR), f"ECHEC silence numérique : obtenu {(b.start, b.end)}, attendu (0, {SR})"


def test_trim_and_frame_range():
    x = tone_with_silence(0.5, 1.0, 0.3)
    b = speech_bounds(x, SR)
    y = trim(x, b)
    assert y.size == b.end - b.start, "ECHEC longueur après découpe"
    # Hand case: speech in samples [8000, 24000), 1.8 s file -> 28800 // 160 + 1 = 181 centred frames.
    # Frame k is centred on 160 k: kept frames are k = 50 (8000) to k = 149 (23840), i.e. range (50, 150).
    first, last = frame_range(SpeechBounds(8000, 24000, 28800), n_frames=181, hop=HOP)
    assert (first, last) == (50, 150), f"ECHEC plage de trames : obtenu {(first, last)}, attendu (50, 150)"


def test_trim_rejects_mismatched_bounds():
    with pytest.raises(ValueError):
        trim(np.zeros(100), SpeechBounds(0, 50, 200))
