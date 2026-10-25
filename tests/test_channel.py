"""Tests for the simulated telephone / VoIP channels."""

import subprocess

import numpy as np
import pytest
import torch

from adg.channel import (
    CODECS,
    ffmpeg_codec,
    ffmpeg_exe,
    float_to_int16,
    g711_ulaw,
    has_encoder,
    packet_loss,
    resample_roundtrip,
    ulaw_decode,
    ulaw_encode,
)
from adg.transforms import apply, folder_name, validate

SR = 16000


def sine(f, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def band_energy_db(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-20)


# ---------- G.711

def test_ulaw_codewords_hand_computed():
    # By hand, with the 14-bit rule (sample >> 2, thresholds half way between levels).
    # The first reconstruction levels are 0, 8, 16 (16-bit units) = 0, 2, 4 in 14-bit units;
    # thresholds (0 + 8 + 4) >> 3 = 1 and (8 + 16 + 4) >> 3 = 3.
    #   0      -> 14-bit 0, level 0, positive: code 0xFF
    #   -1     -> 14-bit floor(-1 / 4) = -1, magnitude 1 >= threshold 1 -> second level, negative:
    #             0x7E (decodes to -8). The Sun 16-bit rule would give 0x7F (-0) here.
    #   32767  -> largest positive level (exponent 7, mantissa 15): 0x80
    got = ulaw_encode(np.array([0, -1, 32767], dtype=np.int16)).tolist()
    assert got == [0xFF, 0x7E, 0x80], f"ECHEC codes mu-law : obtenu {[hex(v) for v in got]}, attendu ['0xff', '0x7e', '0x80']"


def test_ulaw_matches_ffmpeg_byte_for_byte():
    # Independent reference: ffmpeg's own pcm_mulaw encoder on the same int16 samples.
    # The first encoder (Sun bit tricks on 16-bit samples) failed here on 358 of
    # 21,200 bytes; see _ulaw_table for why.
    rng = np.random.default_rng(0)
    pcm = np.concatenate([rng.integers(-32768, 32768, 20000), np.arange(-600, 600)]).astype(np.int16)
    ref = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "8000",
                          "-ac", "1", "-i", "pipe:0", "-c:a", "pcm_mulaw", "-f", "mulaw", "pipe:1"],
                         input=pcm.tobytes(), capture_output=True, check=True).stdout
    mine = ulaw_encode(pcm).tobytes()
    diff = sum(a != b for a, b in zip(mine, ref))
    assert len(ref) == len(mine) and diff == 0, f"ECHEC mu-law vs ffmpeg : {diff} octets différents sur {len(ref)}"


def test_ulaw_roundtrip_error_is_below_one_segment_step():
    # In segment e the levels are 2^(e+3) apart. The error is not always below half a
    # step: near a segment boundary a sample can be coded by the neighbour segment, and
    # the 14-bit rounding adds up to 3 (measured max: half a step + 2^(e+1) - 1). It is
    # always below one full step of the chosen segment, over the whole int16 range.
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    code = ulaw_encode(pcm)
    err = np.abs(ulaw_decode(code).astype(np.int64) - pcm.astype(np.int64))
    exponent = ((code.astype(np.int64) ^ 0xFF) >> 4) & 0x07
    worst = int((err - 2 ** (exponent + 3)).max())
    assert worst < 0, f"ECHEC erreur mu-law : dépasse un pas de segment de {worst + 1} au pire"


def test_g711_snr_of_a_sine_is_about_38_db():
    # G.711 mu-law on a strong sine gives an SNR of roughly 38 dB (8-bit log quantiser).
    x = sine(1000, amp=0.5)
    y = g711_ulaw(x)
    inner = slice(800, -800)  # skip resampling edge effects
    snr = 10 * np.log10(np.sum(x[inner] ** 2) / np.sum((x[inner] - y[inner]) ** 2))
    assert 33 < snr < 42, f"ECHEC RSB G.711 : obtenu {snr:.1f} dB, attendu environ 38 dB (33 à 42)"


# ---------- telephone band

def test_tel8k_removes_everything_above_4_khz():
    x = sine(1000) + sine(6000)
    y = resample_roundtrip(x)
    drop_6k = band_energy_db(x, 5500, 6500) - band_energy_db(y, 5500, 6500)
    drop_1k = band_energy_db(x, 900, 1100) - band_energy_db(y, 900, 1100)
    assert drop_6k > 40, f"ECHEC bande téléphonique : 6 kHz atténué de {drop_6k:.1f} dB, attendu > 40 dB"
    assert drop_1k < 0.5, f"ECHEC bande téléphonique : 1 kHz atténué de {drop_1k:.2f} dB, attendu < 0.5 dB"


@pytest.mark.parametrize("fn", [resample_roundtrip, g711_ulaw])
def test_length_is_preserved(fn):
    x = sine(440, seconds=0.737)
    assert fn(x).shape == x.shape, f"ECHEC longueur {fn.__name__}"


# ---------- packet loss

def test_packet_loss_rate_and_frames():
    # 100 s of signal = 5000 frames of 20 ms. At 10 %, the lost share must be
    # 0.10 +- 3 standard deviations: sqrt(0.1 * 0.9 / 5000) = 0.0042 -> [0.087, 0.113].
    x = np.ones(SR * 100, dtype=np.float32)
    y = packet_loss(x, 0.10, seed=1)
    frames = y.reshape(-1, 320)
    lost = np.mean(np.all(frames == 0, axis=1))
    assert 0.087 < lost < 0.113, f"ECHEC taux de perte : obtenu {lost:.4f}, attendu 0.10 ± 0.013"
    partial = np.mean(np.any(frames == 0, axis=1) & ~np.all(frames == 0, axis=1))
    assert partial == 0, "ECHEC : une trame ne doit jamais être perdue partiellement"


def test_packet_loss_is_reproducible_per_file():
    x = sine(300)
    assert np.array_equal(packet_loss(x, 0.05), packet_loss(x, 0.05)), "ECHEC : même fichier, pertes différentes"
    assert not np.array_equal(packet_loss(x, 0.05), packet_loss(sine(301), 0.05)), \
        "ECHEC : deux fichiers différents ont perdu exactement les mêmes trames"


# ---------- ffmpeg codecs

@pytest.mark.parametrize("codec", sorted(CODECS))
def test_ffmpeg_codecs_keep_length_and_signal(codec):
    encoder = CODECS[codec][0]
    if not has_encoder(encoder):
        pytest.skip(f"{encoder} not in this ffmpeg build")
    x = sine(500, seconds=1.0, amp=0.3)
    y = ffmpeg_codec(x, codec)
    assert y.shape == x.shape, f"ECHEC longueur {codec} : {y.shape}, attendu {x.shape}"
    # A 500 Hz tone survives every codec: its band keeps most of its energy.
    kept = band_energy_db(y, 450, 550) - band_energy_db(x, 450, 550)
    assert kept > -6, f"ECHEC {codec} : le ton à 500 Hz perd {-kept:.1f} dB, attendu moins de 6 dB"


# ---------- transform strings

@pytest.mark.parametrize("spec", ["tel8k", "g711", "gsm", "opus12k", "loss5", "trim40+g711", "trim40+loss10"])
def test_channel_specs_are_valid(spec):
    assert validate(spec) == spec


@pytest.mark.parametrize("spec", ["loss0", "loss100", "opus7k", "none+g711", "g711+"])
def test_bad_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        validate(spec)


def test_chained_folder_name():
    assert folder_name("trim40+g711") == "trim40-g711", "ECHEC nom de dossier pour une chaîne"


def test_chain_applies_left_to_right():
    # 0.5 s silence + tone + 0.5 s silence: trimming first leaves ~1 s, then G.711 keeps that length.
    x = np.concatenate([np.zeros(8000, np.float32), sine(440), np.zeros(8000, np.float32)])
    y = apply(torch.from_numpy(x), SR, "trim40+g711")
    assert abs(y.numel() - SR) <= 640, f"ECHEC chaîne : {y.numel()} échantillons, attendu {SR} ± 640"


def test_randchan_is_deterministic_and_spreads_over_the_channels():
    # Same file -> same output; 200 different files -> every one of the 10 channels used.
    import zlib
    from adg.transforms import CHANNELS

    x = sine(300, seconds=0.5)
    a = apply(torch.from_numpy(x), SR, "randchan")
    b = apply(torch.from_numpy(x), SR, "randchan")
    assert torch.equal(a, b), "ECHEC randchan : même fichier, sorties différentes"
    picks = {CHANNELS[zlib.crc32(sine(200 + k, seconds=0.1).tobytes()) % len(CHANNELS)] for k in range(200)}
    assert picks == set(CHANNELS), f"ECHEC randchan : canaux jamais tirés {sorted(set(CHANNELS) - picks)}"


def test_each_codec_picks_a_binary_that_has_it(monkeypatch):
    # Simulates Kaggle: two builds, each missing one codec. Every encoder must be
    # routed to a build that has it, and an encoder nobody has must raise.
    import adg.channel as ch

    fake = {"ubuntu": frozenset({"libgsm", "libopus"}), "imageio": frozenset({"libopencore_amrnb", "libopus"})}
    monkeypatch.setattr(ch, "ffmpeg_candidates", lambda: ("ubuntu", "imageio"))
    monkeypatch.setattr(ch, "_encoders", lambda exe: fake[exe])
    ch.ffmpeg_for.cache_clear()
    try:
        got = {e: ch.ffmpeg_for(e) for e in ("libgsm", "libopencore_amrnb", "libopus")}
        expected = {"libgsm": "ubuntu", "libopencore_amrnb": "imageio", "libopus": "ubuntu"}
        assert got == expected, f"ECHEC choix du binaire : obtenu {got}, attendu {expected}"
        with pytest.raises(RuntimeError, match="no ffmpeg with a libfdk_aac"):
            ch.ffmpeg_for("libfdk_aac")
    finally:
        ch.ffmpeg_for.cache_clear()
