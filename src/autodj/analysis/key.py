"""Key estimation. Several methods, compared with `autodj eval-keys`:

- mix:     Krumhansl-Kessler profiles on the harmonic part of the full mix (no extra dependencies)
- stems:   the same on the harmonic stems (other + vocals) plus the bass line, so drums don't pollute it
- essentia / essentia-stems: Essentia's KeyExtractor with the EDM-tuned 'edma' profile
  (optional extra `keys`, AGPL-licensed)

Confidence is about *mixing* compatibility: how far the best key beats every key that's
at least 2 Camelot steps away. The relative major/minor (same Camelot number) and fifths
(+-1) don't count against it, since they mix fine; the old best-vs-runner-up margin was
near zero whenever the runner-up was the relative key, which is most of the time.
"""

import librosa
import numpy as np

from autodj.camelot import camelot_distance, key_name, parse_key, to_camelot

_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
ESSENTIA_PROFILE = "edma"

KeyResult = tuple[str, float]  # (Camelot code, confidence)


def _scores(profile: np.ndarray) -> np.ndarray:
    """Correlation with each of the 24 keys; index pc for major, 12 + pc for minor."""
    return np.array([np.corrcoef(profile, np.roll(t, pc))[0, 1] for t in (_KK_MAJOR, _KK_MINOR) for pc in range(12)])


def _code(index: int) -> str:
    return to_camelot(index % 12, index >= 12)


def mixing_confidence(scores: np.ndarray) -> float:
    """How far the best key beats the best key that's 2+ Camelot steps away from it."""
    best = int(np.argmax(scores))
    far = [s for i, s in enumerate(scores) if camelot_distance(_code(i), _code(best)) >= 2]
    return float(scores[best] - max(far))


def _decide(profile: np.ndarray) -> KeyResult:
    scores = _scores(profile)
    return _code(int(np.argmax(scores))), round(mixing_confidence(scores), 3)


def key_from_mix(y: np.ndarray, sr: int) -> KeyResult:
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
    return _decide(chroma.mean(axis=1))


def key_from_stems(harmonic: np.ndarray, bass: np.ndarray, sr: int, bass_weight: float = 0.5) -> KeyResult:
    """Chroma of the harmonic stems, plus the bass line (which mostly plays roots)."""
    h = librosa.feature.chroma_cqt(y=harmonic, sr=sr).mean(axis=1)
    b = librosa.feature.chroma_cqt(y=bass, sr=sr, fmin=librosa.note_to_hz("C1")).mean(axis=1)
    return _decide(h / (h.sum() + 1e-9) + bass_weight * b / (b.sum() + 1e-9))


def key_from_essentia(y44k: np.ndarray) -> KeyResult:
    import essentia.standard as es

    key, scale, strength = es.KeyExtractor(profileType=ESSENTIA_PROFILE, sampleRate=44100)(y44k.astype(np.float32))
    return parse_key(f"{key} {scale}"), round(float(strength), 3)


def essentia_available() -> bool:
    try:
        import essentia.standard  # noqa: F401
    except ImportError:
        return False
    return True


def describe(code: str) -> str:
    """'8A' -> 'A minor'."""
    number, letter = int(code[:-1]), code[-1]
    minor = letter == "A"
    for pc in range(12):
        if to_camelot(pc, minor) == f"{number}{letter}":
            return key_name(pc, minor)
    return code


# Kept for the analysis pipeline: (pitch_class, is_minor, confidence) from the full mix.
def estimate_key(y: np.ndarray, sr: int) -> tuple[int, bool, float]:
    code, confidence = key_from_mix(y, sr)
    minor = code.endswith("A")
    pc = next(p for p in range(12) if to_camelot(p, minor) == code)
    return pc, minor, confidence
