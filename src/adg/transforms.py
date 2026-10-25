"""Audio transforms applied before feature extraction, named by a short string.

    none            the signal as it is
    trim<N>         leading / trailing silence removed, threshold N dB below the loudest frame
    band:<lo>-<hi>  content between lo and hi Hz removed (see adg.bands)
    tel8k           16 -> 8 -> 16 kHz (telephone band only)
    g711            telephone band + G.711 mu-law
    gsm, amr12k     GSM 06.10, AMR-NB 12.2 kbit/s (through ffmpeg)
    opus<N>k        Opus at N kbit/s (through ffmpeg)
    loss<P>         P % of the 20 ms frames set to zero
    randchan        one of CHANNELS, picked per file from a CRC of the signal (augmentation)

Steps can be chained with "+", applied left to right: "trim40+g711" trims the
silence, then sends the result through G.711. One string per condition keeps
the extraction scripts, the Kaggle notebook and the output folder names in sync.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import torch

from .bands import mask_band
from .channel import CODECS, ffmpeg_codec, g711_ulaw, packet_loss, resample_roundtrip
from .silence import speech_bounds, trim

_TRIM = re.compile(r"^trim(\d+)$")
_BAND = re.compile(r"^band:(\d+)-(\d+)$")
_LOSS = re.compile(r"^loss(\d+)$")
# The ten simulated channels of milestone 5, in the order used for reporting.
CHANNELS = ("tel8k", "g711", "gsm", "amr12k", "opus6k", "opus12k", "opus24k", "loss1", "loss5", "loss10")
_SIMPLE = {"tel8k", "g711", "randchan", *CODECS}


def _validate_step(step: str) -> None:
    if step == "none" or step in _SIMPLE or _TRIM.match(step) or _BAND.match(step):
        return
    m = _LOSS.match(step)
    if m and 0 < int(m.group(1)) < 100:
        return
    raise ValueError(f"unknown transform {step!r}; expected none, trim<dB>, band:<lo>-<hi>, "
                     f"loss<percent> or one of {sorted(_SIMPLE)}")


def validate(spec: str) -> str:
    steps = spec.split("+")
    for step in steps:
        _validate_step(step)
    if "none" in steps and len(steps) > 1:
        raise ValueError(f"'none' cannot be chained: {spec!r}")
    return spec


def folder_name(spec: str) -> str:
    """Filesystem-friendly name: 'band:1000-2000' -> 'band1000-2000', 'trim40+g711' -> 'trim40-g711'."""
    return validate(spec).replace(":", "").replace("+", "-")


def _apply_step(arr: np.ndarray, sr: int, step: str) -> np.ndarray:
    if step == "none":
        return arr
    if m := _TRIM.match(step):
        return trim(arr, speech_bounds(arr, sr, top_db=float(m.group(1))))
    if m := _BAND.match(step):
        return mask_band(arr, sr, float(m.group(1)), float(m.group(2)))
    if m := _LOSS.match(step):
        return packet_loss(arr, int(m.group(1)) / 100, sr=sr)
    if step == "randchan":
        # Deterministic per file: the same file always gets the same channel, and
        # over a partition every channel gets about one tenth of the files.
        channel = CHANNELS[zlib.crc32(np.ascontiguousarray(arr, dtype=np.float32).tobytes()) % len(CHANNELS)]
        return _apply_step(arr, sr, channel)
    if step == "tel8k":
        return resample_roundtrip(arr, sr)
    if step == "g711":
        return g711_ulaw(arr, sr)
    return ffmpeg_codec(arr, step, sr)


def apply(x: torch.Tensor, sr: int, spec: str) -> torch.Tensor:
    validate(spec)
    if spec == "none":
        return x
    arr = x.numpy()
    for step in spec.split("+"):
        arr = _apply_step(arr, sr, step)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
