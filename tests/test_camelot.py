import pytest

from autodj.camelot import camelot_distance, parse_key, to_camelot


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("8A", "8A"),
        ("12b", "12B"),
        ("Am", "8A"),
        ("A minor", "8A"),
        ("C", "8B"),
        ("C major", "8B"),
        ("Cmaj", "8B"),
        ("F# minor", "11A"),
        ("Gbm", "11A"),
        ("Bb major", "6B"),
        ("Ebm", "2A"),
        ("E", "12B"),
        ("B", "1B"),
        ("Db", "3B"),
    ],
)
def test_parse_key(text, code):
    assert parse_key(text) == code


def test_to_camelot_covers_wheel_once():
    codes = {to_camelot(pc, minor) for pc in range(12) for minor in (False, True)}
    assert len(codes) == 24


def test_parse_key_rejects_garbage():
    with pytest.raises(ValueError):
        parse_key("H minor")


@pytest.mark.parametrize(
    ("a", "b", "d"),
    [("8A", "8A", 0), ("8A", "9A", 1), ("8A", "8B", 1), ("1A", "12A", 1), ("8A", "2A", 6), ("8A", "10B", 3)],
)
def test_camelot_distance(a, b, d):
    assert camelot_distance(a, b) == d


@pytest.mark.parametrize(("code", "semitones", "shifted"), [
    ("8A", 0, "8A"), ("8A", 1, "3A"), ("8A", -1, "1A"), ("8A", 2, "10A"), ("8A", -2, "6A"), ("12B", 1, "7B"),
])
def test_shift_key(code, semitones, shifted):
    from autodj.camelot import shift_key

    assert shift_key(code, semitones) == shifted
