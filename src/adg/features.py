"""LFCC front-end and frame storage.

Frames of a whole partition are stored in one float16 .npy file (memory-mapped
when read back) plus an index CSV giving, for each utterance, where its frames
start and how many there are. The eval set alone is ~23 M frames, which would
not fit in RAM as float32 on an 8 GB laptop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

INDEX_COLUMNS = ["utt_id", "start", "n_frames"]


@dataclass(frozen=True)
class LfccConfig:
    # Values follow the ASVspoof 2019 LFCC-GMM baseline: 20 ms windows,
    # 10 ms hop, 20 linear filters, 20 cepstral coefficients + deltas.
    sample_rate: int = 16000
    n_fft: int = 512
    win_ms: float = 20.0
    hop_ms: float = 10.0
    n_filter: int = 20
    n_lfcc: int = 20
    with_deltas: bool = True
    delta_win: int = 5  # torchaudio default; covers +-2 frames

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
        if config.win_length > config.n_fft:
            raise ValueError(f"window ({config.win_length}) longer than n_fft ({config.n_fft})")
        self.config = config
        self._lfcc = torchaudio.transforms.LFCC(
            sample_rate=config.sample_rate,
            n_filter=config.n_filter,
            n_lfcc=config.n_lfcc,
            log_lf=True,
            speckwargs={
                "n_fft": config.n_fft,
                "win_length": config.win_length,
                "hop_length": config.hop_length,
                "center": True,
            },
        )

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> np.ndarray:
        """(n_samples,) waveform -> (n_frames, dim) float32 features."""
        if waveform.ndim != 1:
            raise ValueError(f"expected a 1-D waveform, got shape {tuple(waveform.shape)}")
        static = self._lfcc(waveform)  # (n_lfcc, n_frames)
        if self.config.with_deltas:
            d1 = torchaudio.functional.compute_deltas(static, win_length=self.config.delta_win)
            d2 = torchaudio.functional.compute_deltas(d1, win_length=self.config.delta_win)
            static = torch.cat([static, d1, d2], dim=0)
        return static.T.contiguous().numpy()


def extract_partition(utt_ids: list[str], audio_dir: str | Path, out_dir: str | Path,
                      extractor: LfccExtractor, progress: bool = True) -> None:
    """Extract features for every utterance into out_dir/frames.npy + index.csv.

    Two passes: first read the file headers to size the output exactly, then
    fill a memory-mapped array. Nothing larger than one utterance is held in RAM.
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

    iterator = zip(paths, starts, counts)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(paths), desc=out_dir.name, mininterval=5)
    for path, start, n in iterator:
        feats = extractor(load_audio(path, cfg.sample_rate))
        if feats.shape[0] != n:
            raise RuntimeError(f"{path.name}: got {feats.shape[0]} frames, expected {n}")
        frames[start:start + n] = feats
    frames.flush()
    del frames

    pd.DataFrame({"utt_id": utt_ids, "start": starts, "n_frames": counts}).to_csv(
        out_dir / "index.csv", index=False)
    pd.Series(asdict(cfg)).to_json(out_dir / "config.json", indent=2)


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
