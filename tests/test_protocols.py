"""Tests for protocol parsing and the official-count checks.

The real protocol files are not in the repository (licence + size), so these
tests build synthetic files with the same format.
"""

import pytest

from adg.protocols import (
    EXPECTED_COUNTS,
    KNOWN_EVAL_ATTACKS,
    UNKNOWN_EVAL_ATTACKS,
    ProtocolError,
    check_counts,
    check_disjoint_speakers,
    read_protocol,
)


def write_protocol(path, partition="train", drop_one=False):
    """Write a protocol file with exactly the official counts of `partition`."""
    exp = EXPECTED_COUNTS[partition]
    prefix = {"train": "T", "dev": "D", "eval": "E"}[partition]
    lines, n = [], 0
    for _ in range(exp.bonafide):
        lines.append(f"LA_{partition}_spk LA_{prefix}_{n:07d} - - bonafide")
        n += 1
    for attack in sorted(exp.attacks):
        for _ in range(exp.spoof_per_attack):
            lines.append(f"LA_{partition}_spk LA_{prefix}_{n:07d} - {attack} spoof")
            n += 1
    if drop_one:
        lines.pop()
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_protocol_parses_fields(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1138215 - - bonafide\nLA_0079 LA_T_1271820 - A01 spoof\n")
    df = read_protocol(p)
    assert list(df["utt_id"]) == ["LA_T_1138215", "LA_T_1271820"], f"ECHEC ids : obtenu {list(df['utt_id'])}"
    assert list(df["label"]) == [1, 0], f"ECHEC étiquettes : obtenu {list(df['label'])}, attendu [1, 0]"
    assert list(df["attack"]) == ["-", "A01"], f"ECHEC attaques : obtenu {list(df['attack'])}"


@pytest.mark.parametrize("partition", ["train", "dev", "eval"])
def test_official_counts_accepted(tmp_path, partition):
    df = read_protocol(write_protocol(tmp_path / "p.txt", partition))
    expected_total = EXPECTED_COUNTS[partition].total
    assert len(df) == expected_total, f"ECHEC total {partition} : obtenu {len(df)}, attendu {expected_total}"
    check_counts(df, partition)  # must not raise


def test_official_totals_hand_computed():
    # train: 2580 + 6 * 3800  = 2580 + 22800 = 25380
    # dev:   2548 + 6 * 3716  = 2548 + 22296 = 24844
    # eval:  7355 + 13 * 4914 = 7355 + 63882 = 71237
    for partition, expected in [("train", 25380), ("dev", 24844), ("eval", 71237)]:
        got = EXPECTED_COUNTS[partition].total
        assert got == expected, f"ECHEC total officiel {partition} : obtenu {got}, attendu {expected}"


def test_truncated_protocol_is_rejected(tmp_path):
    df = read_protocol(write_protocol(tmp_path / "p.txt", "train", drop_one=True))
    with pytest.raises(ProtocolError, match="A06: got 3799"):
        check_counts(df, "train")


def test_malformed_line_is_rejected(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1 - - bonafide\nLA_0079 LA_T_2 - A01\n")
    with pytest.raises(ProtocolError, match="line 2"):
        read_protocol(p)


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1 - - genuine\n")
    with pytest.raises(ProtocolError, match="unexpected keys"):
        read_protocol(p)


def test_duplicate_utterance_is_rejected(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1 - - bonafide\nLA_0080 LA_T_1 - A01 spoof\n")
    with pytest.raises(ProtocolError, match="duplicated"):
        read_protocol(p)


def test_speaker_leakage_is_detected(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("LA_0001 LA_T_1 - - bonafide\n")
    b.write_text("LA_0001 LA_E_1 - - bonafide\n")
    with pytest.raises(ProtocolError, match="LA_0001"):
        check_disjoint_speakers(read_protocol(a), read_protocol(b))


def test_known_unknown_split():
    # 13 eval attacks, A16 and A19 known -> 11 unknown.
    assert len(UNKNOWN_EVAL_ATTACKS) == 11, f"ECHEC nb attaques inconnues : obtenu {len(UNKNOWN_EVAL_ATTACKS)}, attendu 11"
    assert KNOWN_EVAL_ATTACKS == {"A16", "A19"}, f"ECHEC attaques connues : obtenu {sorted(KNOWN_EVAL_ATTACKS)}"
