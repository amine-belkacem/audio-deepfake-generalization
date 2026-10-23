"""Frequency-band occlusion: remove one band from the signal and see what a detector loses.

The signal goes through an STFT, the bins of one band are set to zero, and the
inverse STFT rebuilds a waveform of the same length. Working on the waveform
(not on the features) means the same masked audio can be fed to every system,
LFCC and wav2vec 2.0 alike.

Known edge effect: with center=True the STFT pads the signal by reflection, and
the kink at the reflection point is broadband. Removing a band also removes
that part of the kink, so the first and last ~24 ms (384 samples, measured)
are slightly altered even outside the band. In the interior, a tone outside the
band comes back within 1e-7 and a tone inside it is attenuated by ~143 dB
(measured on a 1500 Hz tone, band 1-2 kHz). 24 ms out of a 3-4 s utterance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

N_FFT = 512
HOP = 128
BANDS_HZ = tuple((lo, lo + 1000) for lo in range(0, 8000, 1000))  # eight 1 kHz bands up to Nyquist


def mask_band(x: np.ndarray, sr: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Waveform with the content between f_lo and f_hi (Hz) removed. Same length as x."""
    if not 0 <= f_lo < f_hi <= sr / 2:
        raise ValueError(f"band [{f_lo}, {f_hi}] Hz is not inside [0, {sr / 2}] Hz")
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    window = torch.hann_window(N_FFT)
    spec = torch.stft(t, N_FFT, HOP, window=window, center=True, return_complex=True)
    freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / sr)
    band = (freqs >= f_lo) & (freqs < f_hi)
    if f_hi >= sr / 2:
        band |= freqs >= f_lo  # include the Nyquist bin in the top band
    spec[band] = 0
    return torch.istft(spec, N_FFT, HOP, window=window, center=True, length=t.numel()).numpy()


def stratified_subset(protocol: pd.DataFrame, n_bona: int, n_per_attack: int, seed: int = 0) -> pd.DataFrame:
    """n_bona bona fide trials plus n_per_attack trials of every attack, sampled without replacement.

    Equal counts per attack keep every attack visible in the pooled EER, which
    is what an occlusion study needs. Rows keep the protocol order.
    """
    rng = np.random.default_rng(seed)
    keep = []
    bona = protocol.index[protocol["label"] == 1].to_numpy()
    if n_bona > bona.size:
        raise ValueError(f"asked for {n_bona} bona fide trials, only {bona.size} available")
    keep.append(rng.choice(bona, n_bona, replace=False))
    for attack, idx in protocol[protocol["label"] == 0].groupby("attack").groups.items():
        idx = np.asarray(idx)
        if n_per_attack > idx.size:
            raise ValueError(f"{attack}: asked for {n_per_attack} trials, only {idx.size} available")
        keep.append(rng.choice(idx, n_per_attack, replace=False))
    return protocol.loc[np.sort(np.concatenate(keep))]
