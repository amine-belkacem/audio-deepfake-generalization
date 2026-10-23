"""Audio transforms applied before feature extraction, named by a short string.

    none            the signal as it is
    trim<N>         leading / trailing silence removed, threshold N dB below the loudest frame
    band:<lo>-<hi>  content between lo and hi Hz removed (see adg.bands)

One string per condition keeps the extraction scripts, the Kaggle notebook and
the output folder names in sync.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .bands import mask_band
from .silence import speech_bounds, trim

_TRIM = re.compile(r"^trim(\d+)$")
_BAND = re.compile(r"^band:(\d+)-(\d+)$")


def validate(spec: str) -> str:
    if spec == "none" or _TRIM.match(spec) or _BAND.match(spec):
        return spec
    raise ValueError(f"unknown transform {spec!r}; expected 'none', 'trim<dB>' or 'band:<lo>-<hi>'")


def folder_name(spec: str) -> str:
    """Filesystem-friendly name: 'band:1000-2000' -> 'band1000-2000'."""
    return validate(spec).replace(":", "")


def apply(x: torch.Tensor, sr: int, spec: str) -> torch.Tensor:
    validate(spec)
    if spec == "none":
        return x
    arr = x.numpy()
    if m := _TRIM.match(spec):
        out = trim(arr, speech_bounds(arr, sr, top_db=float(m.group(1))))
    else:
        lo, hi = map(float, _BAND.match(spec).groups())
        out = mask_band(arr, sr, lo, hi)
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
