"""Musical key parsing and Camelot wheel math."""

import re

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])\s*([AB])$", re.IGNORECASE)
_NAMED_RE = re.compile(
    r"^([A-G])\s*([#♯b♭]?)\s*(major|maj|minor|min|m)?$", re.IGNORECASE
)


def to_camelot(pitch_class: int, minor: bool) -> str:
    """Camelot code for a key, e.g. (9, True) -> '8A' (A minor)."""
    # Minor keys share a number with their relative major (3 semitones up).
    tonic = (pitch_class + 3) % 12 if minor else pitch_class % 12
    number = (tonic * 7 + 7) % 12 + 1
    return f"{number}{'A' if minor else 'B'}"


def key_name(pitch_class: int, minor: bool) -> str:
    return f"{NOTE_NAMES[pitch_class % 12]} {'minor' if minor else 'major'}"


def parse_key(text: str) -> str:
    """Normalize '8A', 'Am', 'A minor', 'C#', 'Bb major' etc. to a Camelot code."""
    s = text.strip()
    if m := _CAMELOT_RE.match(s):
        return f"{int(m.group(1))}{m.group(2).upper()}"
    if m := _NAMED_RE.match(s):
        letter, accidental, quality = m.groups()
        pc = _LETTER_PC[letter.upper()]
        if accidental in ("#", "♯"):
            pc += 1
        elif accidental in ("b", "♭"):
            pc -= 1
        minor = quality is not None and quality.lower().startswith("m") and not quality.lower().startswith("maj")
        return to_camelot(pc % 12, minor)
    raise ValueError(f"Unrecognized key: {text!r}")


def _split(code: str) -> tuple[int, str]:
    code = parse_key(code)
    return int(code[:-1]), code[-1]


def camelot_distance(a: str, b: str) -> int:
    """Steps on the wheel: 0 same key, 1 adjacent or relative major/minor, more is riskier."""
    na, la = _split(a)
    nb, lb = _split(b)
    ring = abs(na - nb) % 12
    return min(ring, 12 - ring) + (la != lb)
