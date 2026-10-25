"""Layer-wise wav2vec 2.0 representations, pooled over time.

For each utterance and each of the 12 transformer layers, I keep the mean and the
standard deviation over time of the 768-dim frame vectors, so one utterance
becomes a (12, 1536) float16 array (37 KB). Keeping every frame of every layer
would take several hundred GB for ASVspoof 2019 LA.

Utterances go through the model one at a time, without padding. The first block
of wav2vec 2.0 BASE normalises each channel over the whole signal (GroupNorm),
so zero-padding a short file to batch it with a longer one would change its
features. One file at a time is slower but exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import load_audio
from .transforms import apply as apply_transform
from .transforms import validate as validate_transform

SAMPLE_RATE = 16000


def pool_layers(layer_outputs: list[torch.Tensor]) -> np.ndarray:
    """List of L tensors of shape (1, T, D) -> (L, 2D) array: [mean over time, std over time].

    std is the population standard deviation (divides by T), so a single-frame
    input gives std = 0 instead of NaN.
    """
    if not layer_outputs:
        raise ValueError("no layer outputs given")
    stacked = torch.stack([h.squeeze(0) for h in layer_outputs])  # (L, T, D)
    if stacked.ndim != 3:
        raise ValueError(f"expected (1, T, D) tensors, got shape {tuple(layer_outputs[0].shape)}")
    mean = stacked.mean(dim=1)
    std = stacked.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=1).float().cpu().numpy()


class Wav2Vec2LayerExtractor:
    """Frozen wav2vec 2.0 model returning pooled features for every transformer layer."""

    def __init__(self, model: torch.nn.Module | None = None, device: str = "cpu",
                 normalize_waveform: bool = False):
        if model is None:
            import torchaudio
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            if bundle.sample_rate != SAMPLE_RATE:
                raise ValueError(f"model expects {bundle.sample_rate} Hz, data is {SAMPLE_RATE} Hz")
            model = bundle.get_model()
            # BASE was pre-trained on raw waveforms; only the "large lv60" models
            # expect a per-utterance layer norm on the input.
            normalize_waveform = bool(getattr(bundle, "_normalize_waveform", False))
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.normalize_waveform = normalize_waveform

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> np.ndarray:
        """(n_samples,) waveform -> (n_layers, 2 * hidden) float32 array."""
        if waveform.ndim != 1:
            raise ValueError(f"expected a 1-D waveform, got shape {tuple(waveform.shape)}")
        x = waveform.to(self.device).unsqueeze(0)
        if self.normalize_waveform:
            x = torch.nn.functional.layer_norm(x, x.shape)
        layers, _ = self.model.extract_features(x)
        return pool_layers(layers)


def _load_for_ssl(job: tuple[str, bool, str]) -> np.ndarray:
    """Worker side of extract_to_memmap: read + transform one file (CPU work, ffmpeg included)."""
    path, resample, transform = job
    torch.set_num_threads(1)
    return apply_transform(load_audio(path, SAMPLE_RATE, resample=resample), SAMPLE_RATE, transform).numpy()


def extract_to_memmap(paths: list[Path], out_path: str | Path, extractor: Wav2Vec2LayerExtractor,
                      n_layers: int, feat_dim: int, checkpoint_every: int = 500,
                      progress: bool = True, resample: bool = False, transform: str = "none",
                      workers: int = 1) -> None:
    """Fill out_path (.npy, float16, shape (N, n_layers, feat_dim)) one file at a time.

    A companion "<out>.done.npy" boolean mask records which rows are finished, so
    a Kaggle session that hits its time limit can be resumed where it stopped.
    With workers > 1, reading and transforming the audio (codecs go through
    ffmpeg, which is slow) runs in worker processes while the model runs here;
    without it the GPU would wait for ffmpeg most of the time.
    """
    validate_transform(transform)
    out_path = Path(out_path)
    done_path = out_path.with_suffix(".done.npy")
    shape = (len(paths), n_layers, feat_dim)

    if out_path.exists() and done_path.exists():
        feats = np.load(out_path, mmap_mode="r+")
        if feats.shape != shape:
            raise ValueError(f"{out_path} has shape {feats.shape}, expected {shape}")
        done = np.load(done_path)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=shape)
        done = np.zeros(len(paths), dtype=bool)

    todo = np.flatnonzero(~done)
    iterator = todo
    if progress:
        from tqdm import tqdm
        iterator = tqdm(todo, desc=out_path.stem, mininterval=30)

    jobs = [(str(paths[i]), resample, transform) for i in todo]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(workers)
        waves = pool.map(_load_for_ssl, jobs, chunksize=16)
    else:
        pool = None
        waves = map(_load_for_ssl, jobs)

    try:
        for k, (i, wave) in enumerate(zip(iterator, waves), start=1):
            pooled = extractor(torch.from_numpy(wave))
            if pooled.shape != shape[1:]:
                raise RuntimeError(f"{paths[i].name}: got {pooled.shape}, expected {shape[1:]}")
            feats[i] = pooled
            done[i] = True
            if k % checkpoint_every == 0:
                feats.flush()
                np.save(done_path, done)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    feats.flush()
    np.save(done_path, done)
    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} rows of {out_path.name} are still missing")
