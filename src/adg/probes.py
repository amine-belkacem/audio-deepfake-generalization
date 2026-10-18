"""Linear probes on frozen wav2vec 2.0 layers.

One logistic regression per layer, on the pooled (mean, std) vector of that
layer. A probe is deliberately weak: if a linear model separates bona fide from
spoof on layer k, the information is already explicit in that layer, not
something a deep classifier had to build.

The regularisation strength C is picked on the dev set (attacks A01-A06, same
as training), never on eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import compute_eer

C_GRID = (1e-3, 1e-2, 1e-1, 1.0)


def load_split(feat_dir: str | Path, partition: str, protocol: pd.DataFrame) -> np.ndarray:
    """Memory-mapped (N, n_layers, dim) features, checked against the protocol order."""
    feat_dir = Path(feat_dir)
    feats = np.load(feat_dir / f"{partition}.npy", mmap_mode="r")
    ids = pd.read_csv(feat_dir / f"{partition}_utt_ids.csv")["utt_id"].tolist()
    done_path = feat_dir / f"{partition}.done.npy"
    if done_path.exists() and not np.load(done_path).all():
        raise ValueError(f"{partition}: extraction is not finished")
    if ids != protocol["utt_id"].tolist():
        raise ValueError(f"{partition}: feature rows are not in protocol order")
    if feats.shape[0] != len(ids):
        raise ValueError(f"{partition}: {feats.shape[0]} feature rows for {len(ids)} utterances")
    return feats


def make_probe(c: float, seed: int = 0) -> Pipeline:
    # class_weight="balanced": train has ~9 spoofs per bona fide trial; without
    # it the probe would mostly learn the class prior.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                            random_state=seed))


def probe_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Log-odds of bona fide (label 1), so higher = more bona fide, as everywhere."""
    return model.decision_function(np.asarray(x, dtype=np.float32))


@dataclass
class LayerProbe:
    layer: int
    c: float
    dev_eer: float
    model: Pipeline


def fit_layer_probe(layer: int, x_train, y_train, x_dev, y_dev,
                    c_grid=C_GRID, seed: int = 0) -> LayerProbe:
    """Train one probe per C on train, keep the C with the lowest dev EER."""
    xt = np.asarray(x_train, dtype=np.float32)
    xd = np.asarray(x_dev, dtype=np.float32)
    y_dev = np.asarray(y_dev)
    best = None
    for c in c_grid:
        model = make_probe(c, seed).fit(xt, y_train)
        s = probe_scores(model, xd)
        eer, _ = compute_eer(s[y_dev == 1], s[y_dev == 0])
        # Ties go to the stronger regularisation (smaller C), listed first.
        if best is None or eer < best.dev_eer:
            best = LayerProbe(layer, c, eer, model)
    return best
