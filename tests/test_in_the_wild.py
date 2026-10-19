"""Tests for the In-the-Wild reader and for resampled audio loading."""

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from adg.features import LfccExtractor, extract_files, load_audio, open_partition, resampled_length
from adg.in_the_wild import InTheWildError, read_in_the_wild


def make_archive(tmp_path, labels=("bona-fide", "spoof", "bona-fide"), sr=16000):
    root = tmp_path / "release_in_the_wild"
    root.mkdir()
    rows = []
    for i, lab in enumerate(labels):
        sf.write(root / f"{i}.wav", np.zeros(sr // 2, dtype=np.float32), sr)
        rows.append(f"{i}.wav,Speaker {i % 2},{lab}")
    (root / "meta.csv").write_text("file,speaker,label\n" + "\n".join(rows) + "\n")
    return tmp_path


def test_reader_maps_labels_and_ids(tmp_path):
    df = read_in_the_wild(make_archive(tmp_path))
    assert list(df["label"]) == [1, 0, 1], f"ECHEC étiquettes : obtenu {list(df['label'])}, attendu [1, 0, 1]"
    assert list(df["utt_id"]) == ["ITW_0", "ITW_1", "ITW_2"], f"ECHEC ids : obtenu {list(df['utt_id'])}"
    assert list(df["key"]) == ["bonafide", "spoof", "bonafide"], f"ECHEC clés : obtenu {list(df['key'])}"


def test_unknown_label_is_rejected(tmp_path):
    with pytest.raises(InTheWildError, match="unknown label"):
        read_in_the_wild(make_archive(tmp_path, labels=("bona-fide", "maybe")))


def test_missing_audio_is_rejected(tmp_path):
    root = make_archive(tmp_path)
    (root / "release_in_the_wild" / "1.wav").unlink()
    with pytest.raises(InTheWildError, match="1 audio files"):
        read_in_the_wild(root)


def test_two_csv_files_are_rejected(tmp_path):
    root = make_archive(tmp_path)
    (root / "release_in_the_wild" / "other.csv").write_text("a\n1\n")
    with pytest.raises(InTheWildError, match="exactly one"):
        read_in_the_wild(root)


def test_resample_length_formula_matches_torchaudio():
    for sr in (8000, 22050, 44100, 48000):
        for n in (1, 7, 44099, 96001):
            got = resampled_length(n, sr, 16000)
            ref = torchaudio.functional.resample(torch.zeros(n), sr, 16000).numel()
            assert got == ref, f"ECHEC longueur rééchantillonnée (n={n}, sr={sr}) : obtenu {got}, attendu {ref}"


def test_strict_loader_refuses_44k_and_lenient_one_resamples(tmp_path):
    p = tmp_path / "x.wav"
    # 1 s at 44.1 kHz, stereo -> 16000 mono samples after resampling
    sf.write(p, np.zeros((44100, 2), dtype=np.float32), 44100)
    with pytest.raises(ValueError):
        load_audio(p, 16000)
    x = load_audio(p, 16000, resample=True)
    assert x.shape == (16000,), f"ECHEC rééchantillonnage : forme obtenue {tuple(x.shape)}, attendu (16000,)"


def test_lfcc_extraction_with_resampling(tmp_path):
    # 0.5 s at 22.05 kHz -> 11025 samples -> 8000 at 16 kHz -> 8000 // 160 + 1 = 51 frames
    p = tmp_path / "a.wav"
    t = np.arange(11025) / 22050
    sf.write(p, (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 22050)
    extract_files([p], ["a"], tmp_path / "f", LfccExtractor(), progress=False, resample=True)
    frames, index = open_partition(tmp_path / "f")
    assert list(index["n_frames"]) == [51], f"ECHEC nb trames : obtenu {list(index['n_frames'])}, attendu [51]"


def test_exported_labels_are_used_without_audio(tmp_path):
    # On Kaggle: export the label table. On the laptop: no audio, only that table.
    from adg.datasets import LABELS_FILE, export_itw_labels, load_labels

    root = make_archive(tmp_path)
    laptop = tmp_path / "laptop_itw"
    laptop.mkdir()
    n = export_itw_labels(root, laptop / LABELS_FILE)
    df = load_labels("itw", itw_root=laptop)
    assert n == 3 and list(df["label"]) == [1, 0, 1], f"ECHEC export étiquettes : obtenu {n} lignes, {list(df['label'])}"
    assert "path" not in df.columns, "ECHEC : la table exportée ne doit pas contenir de chemins"


def test_macos_shadow_csv_is_ignored(tmp_path):
    root = make_archive(tmp_path)
    shadow = root / "__MACOSX" / "release_in_the_wild"
    shadow.mkdir(parents=True)
    (shadow / "._meta.csv").write_bytes(b"\x00\x05binary")
    df = read_in_the_wild(root)
    assert len(df) == 3, f"ECHEC fichier __MACOSX : obtenu {len(df)} lignes, attendu 3"
