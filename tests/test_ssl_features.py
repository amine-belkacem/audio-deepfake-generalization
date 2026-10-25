"""Tests for the wav2vec 2.0 layer extraction.

They use torchaudio's wav2vec 2.0 BASE architecture with random weights: same
shapes and same code path as the real model, no 360 MB download in CI.
"""

import math

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from adg.ssl_features import Wav2Vec2LayerExtractor, extract_to_memmap, pool_layers

SR = 16000


@pytest.fixture(scope="module")
def extractor():
    torch.manual_seed(0)
    return Wav2Vec2LayerExtractor(model=torchaudio.models.wav2vec2_base())


def noise(seconds, seed):
    return 0.1 * torch.randn(int(SR * seconds), generator=torch.Generator().manual_seed(seed))


def test_pool_layers_hand_computed():
    # One layer, T = 3 frames, D = 2:
    #   frames (1, 4), (3, 4), (5, 4)
    #   mean = (3, 4)
    #   std (population) = sqrt(((1-3)^2 + 0 + (5-3)^2) / 3) = sqrt(8/3) = 1.63299 for dim 0, 0 for dim 1
    h = torch.tensor([[[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]]])
    got = pool_layers([h])
    expected = np.array([[3.0, 4.0, math.sqrt(8 / 3), 0.0]])
    assert np.allclose(got, expected, atol=1e-6), f"ECHEC pooling : obtenu {got}, attendu {expected}"


def test_pool_layers_single_frame_has_zero_std():
    got = pool_layers([torch.ones(1, 1, 3)])
    assert np.all(got[0, 3:] == 0), f"ECHEC écart-type sur 1 trame : obtenu {got[0, 3:]}, attendu 0"


def test_extractor_output_shape(extractor):
    # 12 transformer layers, 768-dim hidden state -> mean + std = 1536.
    got = extractor(noise(1.0, seed=1))
    assert got.shape == (12, 1536), f"ECHEC forme : obtenu {got.shape}, attendu (12, 1536)"
    assert np.all(np.isfinite(got)), "ECHEC : valeurs non finies"


def test_zero_padding_changes_the_features(extractor):
    # This is the reason files are processed one at a time: padding a file with
    # silence changes its pooled features. If this test ever fails, batching
    # with padding would become safe and could speed up extraction.
    x = noise(1.0, seed=2)
    padded = torch.cat([x, torch.zeros(SR)])
    a, b = extractor(x), extractor(padded)
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    assert rel > 1e-2, f"ECHEC : le padding ne change presque rien (écart relatif {rel:.2e}), hypothèse à revoir"


def test_extraction_resumes_after_interruption(tmp_path, extractor):
    paths = []
    for i in range(4):
        p = tmp_path / f"u{i}.flac"
        sf.write(p, noise(0.5, seed=10 + i).numpy(), SR)
        paths.append(p)
    out = tmp_path / "w2v.npy"

    # Full run, for reference.
    extract_to_memmap(paths, tmp_path / "ref.npy", extractor, 12, 1536, progress=False)
    ref = np.load(tmp_path / "ref.npy")

    # Simulated interruption: only rows 0 and 2 are marked done, rows 1 and 3
    # are garbage. A resumed run must recompute exactly those two rows.
    feats = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(4, 12, 1536))
    feats[:] = 0
    feats[0], feats[2] = ref[0], ref[2]
    feats.flush()
    del feats
    np.save(out.with_suffix(".done.npy"), np.array([True, False, True, False]))

    extract_to_memmap(paths, out, extractor, 12, 1536, progress=False)
    got = np.load(out)
    assert np.array_equal(got, ref), "ECHEC reprise : résultat différent d'une extraction complète"
    assert np.load(out.with_suffix(".done.npy")).all(), "ECHEC reprise : toutes les lignes devraient être faites"


def test_shape_mismatch_on_resume_is_rejected(tmp_path, extractor):
    out = tmp_path / "w2v.npy"
    np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(3, 12, 1536)).flush()
    np.save(out.with_suffix(".done.npy"), np.zeros(3, dtype=bool))
    p = tmp_path / "a.flac"
    sf.write(p, noise(0.2, seed=0).numpy(), SR)
    with pytest.raises(ValueError, match="expected"):
        extract_to_memmap([p], out, extractor, 12, 1536, progress=False)


def test_parallel_loading_gives_the_same_features(tmp_path, extractor):
    # Workers only read and transform the audio; the model runs in the main process,
    # so the result must be bit-identical to the sequential run.
    paths = []
    for i in range(5):
        p = tmp_path / f"u{i}.flac"
        sf.write(p, noise(0.4, seed=30 + i).numpy(), SR)
        paths.append(p)
    extract_to_memmap(paths, tmp_path / "seq.npy", extractor, 12, 1536, progress=False, transform="g711")
    extract_to_memmap(paths, tmp_path / "par.npy", extractor, 12, 1536, progress=False, transform="g711", workers=2)
    assert np.array_equal(np.load(tmp_path / "seq.npy"), np.load(tmp_path / "par.npy")), \
        "ECHEC chargement parallèle : caractéristiques différentes"
