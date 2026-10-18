"""LFCC front-end and frame storage.

Frames of a whole partition are stored in one float16 .npy file (memory-mapped
when read back) plus an index CSV giving, for each utterance, where its frames
start and how many there are. The eval set alone is ~23 M frames, which would
not fit in RAM as float32 on an 8 GB laptop.

The LFCC chain is written out by hand (STFT -> linear filterbank -> log -> DCT)
rather than calling torchaudio.transforms.LFCC, because the ASVspoof 2019
baseline uses settings that transform does not expose: a Hamming window, a
30 Hz lower band edge and a log floor far below torchaudio's fixed 1e-6.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

INDEX_COLUMNS = ["utt_id", "start", "n_frames"]
WINDOWS = {"hann": torch.hann_window, "hamming": torch.hamming_window}


@dataclass(frozen=True)
class LfccConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_ms: float = 20.0
    hop_ms: float = 10.0
    window: str = "hamming"
    f_min: float = 30.0
    n_filter: int = 20
    n_lfcc: int = 20
    log_floor: float = 2.2204e-16  # float64 machine epsilon, as in the reference code
    with_deltas: bool = True
    delta_win: int = 5  # torchaudio default; covers +-2 frames

    def __post_init__(self) -> None:
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {sorted(WINDOWS)}, got {self.window!r}")
        if not 0 <= self.f_min < self.sample_rate / 2:
            raise ValueError(f"f_min must be in [0, {self.sample_rate / 2}), got {self.f_min}")
        if self.win_length > self.n_fft:
            raise ValueError(f"window ({self.win_length}) longer than n_fft ({self.n_fft})")

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000))

    @property
    def dim(self) -> int:
        return self.n_lfcc * (3 if self.with_deltas else 1)

    def n_frames(self, n_samples: int) -> int:
        """Frame count produced by torchaudio's centred STFT."""
        return n_samples // self.hop_length + 1


# The two configurations used in milestone 1. "first_run" is what torchaudio's
# LFCC transform does with default settings; "asvspoof2019_b02" follows the
# published description of the official baseline.
PRESETS = {
    "first_run": LfccConfig(window="hann", f_min=0.0, log_floor=1e-6),
    "asvspoof2019_b02": LfccConfig(),
}


def load_audio(path: str | Path, expected_sr: int) -> torch.Tensor:
    """Read a mono file as a float32 tensor of shape (n_samples,)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr}, expected {expected_sr}")
    if data.ndim != 1:
        raise ValueError(f"{path}: expected mono audio, got shape {data.shape}")
    return torch.from_numpy(data)


class LfccExtractor:
    def __init__(self, config: LfccConfig = LfccConfig()):
        self.config = config
        self._spec = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length,
            window_fn=WINDOWS[config.window], power=2.0, center=True)
        # (n_freqs, n_filter): triangular filters, linearly spaced from f_min to Nyquist.
        self._fbank = torchaudio.functional.linear_fbanks(
            n_freqs=config.n_fft // 2 + 1, f_min=config.f_min, f_max=config.sample_rate / 2,
            n_filter=config.n_filter, sample_rate=config.sample_rate)
        # (n_filter, n_lfcc): orthonormal DCT-II, same as torchaudio's LFCC.
        self._dct = torchaudio.functional.create_dct(config.n_lfcc, config.n_filter, norm="ortho")

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> np.ndarray:
        """(n_samples,) waveform -> (n_frames, dim) float32 features."""
        if waveform.ndim != 1:
            raise ValueError(f"expected a 1-D waveform, got shape {tuple(waveform.shape)}")
        power = self._spec(waveform)                     # (n_freqs, n_frames)
        band_energy = power.T @ self._fbank              # (n_frames, n_filter)
        static = torch.log(band_energy + self.config.log_floor) @ self._dct  # (n_frames, n_lfcc)
        if self.config.with_deltas:
            s = static.T  # compute_deltas works along the last axis (time)
            d1 = torchaudio.functional.compute_deltas(s, win_length=self.config.delta_win)
            d2 = torchaudio.functional.compute_deltas(d1, win_length=self.config.delta_win)
            static = torch.cat([s, d1, d2], dim=0).T
        return static.contiguous().numpy()


# Worker-process state for the parallel extraction (one extractor per process).
_worker_extractor: LfccExtractor | None = None


def _init_worker(config: LfccConfig) -> None:
    global _worker_extractor
    torch.set_num_threads(1)  # parallelism comes from the processes, not from torch
    _worker_extractor = LfccExtractor(config)


def _extract_one(path: Path) -> np.ndarray:
    return _worker_extractor(load_audio(path, _worker_extractor.config.sample_rate)).astype(np.float16)


def extract_partition(utt_ids: list[str], audio_dir: str | Path, out_dir: str | Path,
                      extractor: LfccExtractor, progress: bool = True, workers: int = 1) -> None:
    """Extract features for every utterance into out_dir/frames.npy + index.csv.

    Two passes: first read the file headers to size the output exactly, then
    fill a memory-mapped array. Nothing larger than one utterance is held in RAM.
    With workers > 1, files are decoded and processed in separate processes; the
    main process only writes the results into the memmap, in order.
    """
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = extractor.config

    paths = [audio_dir / f"{u}.flac" for u in utt_ids]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files missing, first: {missing[0]}")

    counts = np.array([cfg.n_frames(sf.info(str(p)).frames) for p in paths], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    frames = np.lib.format.open_memmap(out_dir / "frames.npy", mode="w+", dtype=np.float16,
                                       shape=(int(counts.sum()), cfg.dim))

    if workers > 1:
        pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,))
        results = pool.map(_extract_one, paths, chunksize=64)
    else:
        pool = None
        results = (extractor(load_audio(p, cfg.sample_rate)) for p in paths)

    iterator = zip(paths, starts, counts, results)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(paths), desc=out_dir.name, mininterval=10)
    try:
        for path, start, n, feats in iterator:
            if feats.shape[0] != n:
                raise RuntimeError(f"{path.name}: got {feats.shape[0]} frames, expected {n}")
            frames[start:start + n] = feats
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    frames.flush()
    del frames

    pd.DataFrame({"utt_id": utt_ids, "start": starts, "n_frames": counts}).to_csv(
        out_dir / "index.csv", index=False)
    pd.Series(asdict(cfg)).to_json(out_dir / "config.json", indent=2)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def open_partition(feat_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-mapped frames and their index, as written by extract_partition."""
    feat_dir = Path(feat_dir)
    frames = np.load(feat_dir / "frames.npy", mmap_mode="r")
    index = pd.read_csv(feat_dir / "index.csv")
    if list(index.columns) != INDEX_COLUMNS:
        raise ValueError(f"{feat_dir}/index.csv: unexpected columns {list(index.columns)}")
    if int(index["n_frames"].sum()) != frames.shape[0]:
        raise ValueError(f"{feat_dir}: index covers {index['n_frames'].sum()} frames, "
                         f"array has {frames.shape[0]}")
    return frames, index
