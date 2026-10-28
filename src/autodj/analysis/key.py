"""Global key estimation (Krumhansl-Kessler profiles on harmonic chroma)."""

import librosa
import numpy as np

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(y: np.ndarray, sr: int) -> tuple[int, bool, float]:
    """Returns (pitch_class, is_minor, confidence)."""
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
    profile = chroma.mean(axis=1)
    scores = sorted(
        (
            (np.corrcoef(profile, np.roll(template, pc))[0, 1], pc, minor)
            for pc in range(12)
            for minor, template in ((False, _MAJOR), (True, _MINOR))
        ),
        reverse=True,
    )
    (best, pc, minor), (second, *_) = scores[0], scores[1]
    return pc, minor, float(best - second)
