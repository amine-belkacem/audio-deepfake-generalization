"""Leading and trailing silence: measurement and trimming.

ASVspoof 2019 LA is known to leak the label through the length of the silence
before and after the speech (Müller et al., 2021, arXiv:2106.12914). This
module measures that silence so it can be (1) used alone as a "detector" to
quantify the leak and (2) removed before scoring to see what the real
detectors lose.

A frame is "speech" when its RMS energy is within `top_db` dB of the loudest
frame of the same file. The threshold is relative on purpose: absolute levels
differ between files, and a fixed dBFS threshold would trim quiet speakers.
Only the edges are trimmed; pauses inside the utterance are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 20.0
HOP_MS = 10.0
DEFAULT_TOP_DB = 40.0
DIGITAL_SILENCE_DB = -100.0  # loudest frame below this: the file has no signal at all
_EPS = 1e-12


@dataclass(frozen=True)
class SpeechBounds:
    start: int     # first sample of speech
    end: int       # one past the last sample of speech
    n_samples: int

    @property
    def lead_samples(self) -> int:
        return self.start

    @property
    def trail_samples(self) -> int:
        return self.n_samples - self.end


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS) -> np.ndarray:
    """RMS energy in dB of each frame (non-centred frames, last partial frame dropped)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    frame, hop = int(round(sr * frame_ms / 1000)), int(round(sr * hop_ms / 1000))
    if x.size < frame:
        return np.array([10 * np.log10(np.mean(x ** 2) + _EPS)]) if x.size else np.array([])
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + _EPS)


def speech_bounds(x: np.ndarray, sr: int, top_db: float = DEFAULT_TOP_DB,
                  frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS) -> SpeechBounds:
    """Sample range from the first to the last frame within top_db of the peak frame.

    A file with no frame above the threshold (digital silence) keeps its full
    length, so nothing downstream ever receives an empty signal.
    """
    if top_db <= 0:
        raise ValueError(f"top_db must be positive, got {top_db}")
    x = np.asarray(x).ravel()
    n = x.size
    db = frame_rms_db(x, sr, frame_ms, hop_ms)
    if db.size == 0 or db.max() < DIGITAL_SILENCE_DB:
        return SpeechBounds(0, n, n)
    active = np.flatnonzero(db >= db.max() - top_db)
    frame, hop = int(round(sr * frame_ms / 1000)), int(round(sr * hop_ms / 1000))
    start = int(active[0] * hop)
    end = int(min(n, active[-1] * hop + frame))
    return SpeechBounds(start, end, n)


def trim(x: np.ndarray, bounds: SpeechBounds) -> np.ndarray:
    if bounds.n_samples != np.asarray(x).size:
        raise ValueError(f"bounds computed for {bounds.n_samples} samples, signal has {np.asarray(x).size}")
    return np.asarray(x)[bounds.start:bounds.end]


def frame_range(bounds: SpeechBounds, n_frames: int, hop: int) -> tuple[int, int]:
    """Matching range of feature frames for a centred STFT with this hop (frame k is centred on k*hop).

    Used to drop silence from already extracted LFCC frames instead of
    re-extracting: frames whose centre lies inside [start, end) are kept.
    """
    first = int(np.ceil(bounds.start / hop))
    last = int(np.ceil(bounds.end / hop))  # exclusive
    first, last = min(first, n_frames - 1), min(max(last, first + 1), n_frames)
    return first, last
