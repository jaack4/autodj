import numpy as np
import pytest

from autodj.models import Analysis, Section


def make_analysis(
    sections: list[tuple[str, int, int, float]],
    key: str = "8A",
    bpm: float = 126.0,
    path: str = "/music/track.mp3",
    offsets_ms: list[float] | None = None,
    key_confidence: float = 0.2,
    vocals: list[float] | None = None,
) -> Analysis:
    """A fake analysis: sections as (label, start_bar, end_bar, energy)."""
    n = sections[-1][2]
    bar = 4 * 60 / bpm
    downbeats = (0.5 + bar * np.arange(n)).tolist()
    energy = np.zeros(n)
    for _, s, e, en in sections:
        energy[s:e] = en
    return Analysis(
        version=0, path=path, sha256="0" * 64, duration=0.5 + bar * n, beat_backend="test",
        bpm=bpm, beats=(0.5 + bar / 4 * np.arange(4 * n)).tolist(), downbeats=downbeats,
        onset_offset_ms=0.0, grid_drift_ms=0.0, grid_ok=True,
        key=key, key_name="", key_confidence=key_confidence, lufs=-8.0,
        bar_energy=energy.tolist(), bar_novelty=[0.0] * n,
        bar_offset_ms=offsets_ms or [0.0] * n, bar_offset_confidence=[3.0] * n, bar_vocals=vocals,
        sections=[Section(label=l, start_bar=s, end_bar=e, start=downbeats[s], end=downbeats[e - 1] + bar, energy=en)
                  for l, s, e, en in sections],
    )


@pytest.fixture
def club_track():
    """intro 16 / drop 32 / break 16 / drop 32 / outro 16 = 112 bars."""
    return lambda **kw: make_analysis(
        [("intro", 0, 16, 0.4), ("drop", 16, 48, 0.95), ("break", 48, 64, 0.5),
         ("drop", 64, 96, 0.95), ("outro", 96, 112, 0.4)], **kw)
