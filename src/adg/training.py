"""Datasets, training loop and scoring for the milestone 3 classifiers.

Model selection uses the dev EER (attacks A01-A06 only). The epoch with the
lowest dev EER is kept; eval is only touched once, by score() on the final model.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import compute_eer


def chunked_mean_std(x: np.ndarray, chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over axis 0 of a large (possibly memory-mapped) array, in float64."""
    n = x.shape[0]
    s1 = np.zeros(x.shape[1:])
    s2 = np.zeros(x.shape[1:])
    for i in range(0, n, chunk):
        b = np.asarray(x[i:i + chunk], dtype=np.float64)
        s1 += b.sum(0)
        s2 += (b * b).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class PooledDataset(Dataset):
    """Rows of a (N, n_layers, dim) array, standardised with train statistics."""

    def __init__(self, feats: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        if len(feats) != len(labels):
            raise ValueError(f"{len(feats)} feature rows for {len(labels)} labels")
        self.feats = feats  # kept as float16 (memmap or array): float32 copies would not fit in RAM
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        x = (np.asarray(self.feats[i], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.labels[i])


class FrameCropDataset(Dataset):
    """Fixed-length LFCC crops (n_frames x dim), transposed to (dim, n_frames).

    Training: a random crop per utterance and per epoch. Evaluation: the first
    n_frames, so scores are deterministic. Utterances shorter than n_frames are
    tiled (repeated), the usual choice in ASVspoof systems, rather than padded
    with zeros, which would look like an extra block of silence.
    """

    def __init__(self, frames: np.ndarray, starts: np.ndarray, counts: np.ndarray, labels: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, n_frames: int = 400, train: bool = False, seed: int = 0):
        if not (len(starts) == len(counts) == len(labels)):
            raise ValueError("starts, counts and labels must have the same length")
        self.frames, self.starts, self.counts = frames, np.asarray(starts), np.asarray(counts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean, self.std = mean, std
        self.n_frames, self.train = n_frames, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def crop(self, i: int) -> np.ndarray:
        start, count = int(self.starts[i]), int(self.counts[i])
        if count >= self.n_frames:
            off = int(self.rng.integers(0, count - self.n_frames + 1)) if self.train else 0
            x = self.frames[start + off:start + off + self.n_frames]
        else:
            x = np.tile(self.frames[start:start + count], (int(np.ceil(self.n_frames / count)), 1))[:self.n_frames]
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def __getitem__(self, i):
        return torch.from_numpy(self.crop(i).T.copy()), torch.tensor(self.labels[i])


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 5     # stop after this many epochs without a better dev EER
    seed: int = 0


@dataclass
class TrainResult:
    best_epoch: int
    best_dev_eer: float
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def score(model: nn.Module, dataset: Dataset, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return np.concatenate([model(x.to(device)).float().cpu().numpy() for x, _ in loader])


def train_model(model: nn.Module, train_ds: Dataset, dev_ds: Dataset, cfg: TrainConfig,
                device: str = "cpu", verbose: bool = True) -> TrainResult:
    """Adam + BCE with logits, early stopping on the dev EER. The model is left
    with the weights of its best dev epoch."""
    set_seed(cfg.seed)
    model.to(device)
    labels = np.asarray(train_ds.labels)
    n_bona, n_spoof = int(labels.sum()), int((1 - labels).sum())
    if n_bona == 0 or n_spoof == 0:
        raise ValueError(f"training set needs both classes, got {n_bona} bona fide and {n_spoof} spoof")
    # Bona fide is the positive class and ~9x rarer: weight it so both classes
    # contribute equally to the loss.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_spoof / n_bona, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen, drop_last=True)
    dev_labels = np.asarray(dev_ds.labels)

    result = TrainResult(best_epoch=-1, best_dev_eer=float("inf"))
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
            n += len(y)
        s = score(model, dev_ds, device=device)
        dev_eer, _ = compute_eer(s[dev_labels == 1], s[dev_labels == 0])
        result.history.append({"epoch": epoch, "train_loss": total / n, "dev_eer": dev_eer,
                               "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"  epoch {epoch:2d}: loss {total / n:.4f}, dev EER {100 * dev_eer:.3f} %, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
        if dev_eer < result.best_dev_eer:
            result.best_epoch, result.best_dev_eer = epoch, dev_eer
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - result.best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return result
