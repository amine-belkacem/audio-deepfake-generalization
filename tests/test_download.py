"""Tests for the retry logic of the dataset downloads (no network involved)."""

import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_asvspoof2019 import open_with_retry  # noqa: E402


def http_error(code):
    return urllib.error.HTTPError("https://example.org", code, "refused", {}, None)


def fake_opener(failures):
    """Raises the given errors in order, then returns "ok"."""
    calls = []

    def opener(req, timeout):
        calls.append(req)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return "ok"
    return opener, calls


def test_two_403_then_success_waits_30_then_60_seconds():
    opener, calls = fake_opener([http_error(403), http_error(403)])
    waits = []
    got = open_with_retry("req", attempts=5, first_wait_s=30, opener=opener, sleep=waits.append)
    assert got == "ok", f"ECHEC : obtenu {got!r}, attendu 'ok'"
    assert len(calls) == 3, f"ECHEC nb tentatives : obtenu {len(calls)}, attendu 3"
    # back-off 30 * 2^0 = 30 s, then 30 * 2^1 = 60 s
    assert waits == [30, 60], f"ECHEC attentes : obtenu {waits}, attendu [30, 60]"


def test_404_is_not_retried():
    opener, calls = fake_opener([http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        open_with_retry("req", opener=opener, sleep=lambda s: None)
    assert len(calls) == 1, f"ECHEC : une 404 ne doit pas être retentée, {len(calls)} tentatives"


def test_gives_up_after_the_last_attempt():
    opener, calls = fake_opener([http_error(403)] * 10)
    with pytest.raises(urllib.error.HTTPError):
        open_with_retry("req", attempts=3, opener=opener, sleep=lambda s: None)
    assert len(calls) == 3, f"ECHEC nb tentatives : obtenu {len(calls)}, attendu 3"


def test_find_la_root_in_a_mounted_copy(tmp_path):
    from verify_asvspoof_copy import find_la_root

    proto = tmp_path / "asvpoof-2019-dataset" / "LA" / "LA" / "ASVspoof2019_LA_cm_protocols"
    proto.mkdir(parents=True)
    (proto / "ASVspoof2019.LA.cm.train.trn.txt").write_text("")
    got = find_la_root(tmp_path)
    expected = tmp_path / "asvpoof-2019-dataset" / "LA" / "LA"
    assert got == expected, f"ECHEC recherche du dossier LA : obtenu {got}, attendu {expected}"


def test_find_la_root_refuses_two_copies(tmp_path):
    from verify_asvspoof_copy import find_la_root

    for name in ("a", "b"):
        proto = tmp_path / name / "ASVspoof2019_LA_cm_protocols"
        proto.mkdir(parents=True)
        (proto / "ASVspoof2019.LA.cm.train.trn.txt").write_text("")
    with pytest.raises(SystemExit, match="expected one"):
        find_la_root(tmp_path)
