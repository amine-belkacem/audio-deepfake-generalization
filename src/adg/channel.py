"""Simulated telephone and VoIP channels.

Every function takes a 16 kHz float waveform in [-1, 1] and returns a 16 kHz
waveform of the same length, so a degraded file goes through the exact same
feature extraction as a clean one.

    resample_roundtrip  16 kHz -> 8 kHz -> 16 kHz: the telephone band, nothing else
    g711_ulaw           8 kHz + G.711 mu-law (ITU-T G.711), implemented here
    ffmpeg_codec        GSM 06.10, AMR-NB, Opus... through an ffmpeg binary
    packet_loss         20 ms frames set to zero at random, no concealment

G.711 is written out by hand because it is short and testable against ffmpeg's
own pcm_mulaw encoder (tests/test_channel.py). The other codecs are far too
complex to re-implement; they go through ffmpeg, located with imageio-ffmpeg
so the same binary is used on my laptop and on Kaggle.
"""

from __future__ import annotations

import shutil
import subprocess
import zlib
from functools import lru_cache

import numpy as np
import torch
import torchaudio

SR = 16000
TEL_SR = 8000

# G.711 mu-law decoder bias (ITU-T G.711, classic reference code).
ULAW_BIAS = 0x84   # 132


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Cut or zero-pad the end so the output has exactly n samples."""
    if y.size >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - y.size, dtype=y.dtype)])


def resample_roundtrip(x: np.ndarray, sr: int = SR, mid_sr: int = TEL_SR) -> np.ndarray:
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y = torchaudio.functional.resample(torchaudio.functional.resample(t, sr, mid_sr), mid_sr, sr)
    return _fit_length(y.numpy(), t.numel())


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


@lru_cache(maxsize=1)
def _ulaw_table() -> np.ndarray:
    """Encoder table indexed by the 14-bit sample (16-bit sample >> 2).

    G.711 mu-law works on 14-bit linear samples, with decision thresholds half
    way between two reconstruction levels. My first encoder was the classic Sun
    bit-manipulation code on 16-bit samples; it differed from ffmpeg by one
    code on 851 of the 65,536 inputs, all at decision boundaries. Building the
    table the way ffmpeg does (libavcodec pcm_tablegen) matches it on all of them.
    """
    dec = ulaw_decode(np.arange(256, dtype=np.uint8)).astype(np.int64)
    mask = 0xFF
    table = np.zeros(16384, dtype=np.uint8)
    table[8192] = mask
    j = 1
    for i in range(127):
        threshold = (dec[i ^ mask] + dec[(i + 1) ^ mask] + 4) >> 3  # midpoint, in 14-bit units
        table[8192 - threshold + 1:8192 - j + 1] = i ^ (mask ^ 0x80)  # negative side
        table[8192 + j:8192 + threshold] = i ^ mask                   # positive side
        j = max(j, threshold)
    table[8192 - 8191:8192 - j + 1] = 127 ^ (mask ^ 0x80)
    table[8192 + j:] = 127 ^ mask
    table[0] = table[1]
    return table


def ulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """16-bit linear PCM -> 8-bit G.711 mu-law codewords (uint8)."""
    return _ulaw_table()[(pcm.astype(np.int32) + 32768) >> 2]


def ulaw_decode(code: np.ndarray) -> np.ndarray:
    """8-bit G.711 mu-law codewords -> 16-bit linear PCM."""
    u = ~code.astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + ULAW_BIAS) << exponent) - ULAW_BIAS
    return np.where(sign != 0, -mag, mag).astype(np.int16)


def g711_ulaw(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Telephone band + mu-law quantisation, back at the original rate and length."""
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    narrow = torchaudio.functional.resample(t, sr, TEL_SR).numpy()
    coded = ulaw_decode(ulaw_encode(float_to_int16(narrow))).astype(np.float32) / 32768.0
    y = torchaudio.functional.resample(torch.from_numpy(coded), TEL_SR, sr)
    return _fit_length(y.numpy(), t.numel())


def packet_loss(x: np.ndarray, rate: float, sr: int = SR, frame_ms: float = 20.0,
                seed: int | None = None) -> np.ndarray:
    """Zero each 20 ms frame independently with probability `rate`.

    Without a seed, the seed is a CRC of the signal itself: the same file always
    loses the same frames, which keeps every extraction reproducible without
    having to pass file names around.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"loss rate must be in [0, 1), got {rate}")
    y = np.array(x, dtype=np.float32, copy=True)
    frame = int(round(sr * frame_ms / 1000))
    n_frames = -(-y.size // frame)
    if seed is None:
        seed = zlib.crc32(y.tobytes())
    lost = np.random.default_rng(seed).random(n_frames) < rate
    for k in np.flatnonzero(lost):
        y[k * frame:(k + 1) * frame] = 0.0
    return y


@lru_cache(maxsize=1)
def ffmpeg_candidates() -> tuple[str, ...]:
    """Every ffmpeg binary available, in order of preference.

    ADG_FFMPEG_PATHS (several paths separated by os.pathsep) if set, then the
    imageio-ffmpeg binary, then the one on PATH. Several are kept because no
    single build has every codec everywhere: on Kaggle the Ubuntu ffmpeg has
    GSM and Opus but not AMR-NB, the imageio-ffmpeg build has AMR-NB and Opus
    but not GSM.
    """
    import os
    found: list[str] = []
    if os.environ.get("ADG_FFMPEG_PATHS"):
        found += [p for p in os.environ["ADG_FFMPEG_PATHS"].split(os.pathsep) if p]
    try:
        import imageio_ffmpeg
        found.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        found.append(shutil.which("ffmpeg"))
    unique = tuple(dict.fromkeys(found))
    if not unique:
        raise RuntimeError("no ffmpeg binary: pip install imageio-ffmpeg")
    return unique


def ffmpeg_exe() -> str:
    """The preferred ffmpeg (used where any build will do, e.g. pcm_mulaw in the tests)."""
    return ffmpeg_candidates()[0]


@lru_cache(maxsize=None)
def _encoders(exe: str) -> frozenset[str]:
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return frozenset(line.split()[1] for line in out.splitlines() if len(line.split()) > 1)


@lru_cache(maxsize=None)
def ffmpeg_for(encoder: str) -> str:
    """First ffmpeg that has this encoder. The same binary also decodes, so a
    codec never mixes two builds."""
    for exe in ffmpeg_candidates():
        if encoder in _encoders(exe):
            return exe
    raise RuntimeError(f"no ffmpeg with a {encoder} encoder among {list(ffmpeg_candidates())}")


def has_encoder(name: str) -> bool:
    return any(name in _encoders(exe) for exe in ffmpeg_candidates())


# name -> (encoder, encoder sample rate, extra encoder args, container)
CODECS = {
    "gsm": ("libgsm", 8000, [], "gsm"),
    "amr12k": ("libopencore_amrnb", 8000, ["-b:a", "12.2k"], "amr"),
    "opus6k": ("libopus", 16000, ["-b:a", "6k"], "ogg"),
    "opus12k": ("libopus", 16000, ["-b:a", "12k"], "ogg"),
    "opus24k": ("libopus", 16000, ["-b:a", "24k"], "ogg"),
}


def ffmpeg_codec(x: np.ndarray, codec: str, sr: int = SR) -> np.ndarray:
    """Encode then decode with ffmpeg, back to sr Hz and to the original length."""
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}, expected one of {sorted(CODECS)}")
    encoder, enc_sr, extra, container = CODECS[codec]
    pcm = np.asarray(x, dtype=np.float32).tobytes()
    base = [ffmpeg_for(encoder), "-hide_banner", "-loglevel", "error"]
    enc = subprocess.run(base + ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                                 "-ar", str(enc_sr), "-c:a", encoder, *extra, "-f", container, "pipe:1"],
                         input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(base + ["-f", container, "-i", "pipe:0", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                                 "pipe:1"], input=enc, capture_output=True, check=True).stdout
    return _fit_length(np.frombuffer(dec, dtype=np.float32).copy(), len(pcm) // 4)
