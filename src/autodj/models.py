from pydantic import BaseModel


class Section(BaseModel):
    label: str  # intro | build | drop | break | body | outro
    start_bar: int
    end_bar: int  # exclusive
    start: float  # seconds
    end: float
    energy: float  # 0..1, relative to the track


class Analysis(BaseModel):
    version: int
    path: str
    sha256: str
    duration: float
    beat_backend: str

    bpm: float
    beats: list[float]
    downbeats: list[float]
    beats_per_bar: int = 4
    kick_coverage: float  # share of grid beats with a kick onset nearby
    kick_offset_ms: float
    grid_drift_ms: float  # kick-vs-grid drift across the track
    grid_ok: bool  # False if the track isn't constant-tempo (or the BPM is off)

    key: str  # Camelot, e.g. "8A"
    key_name: str
    key_confidence: float  # margin between best and runner-up key, ~0..0.3
    lufs: float

    bar_energy: list[float]
    bar_novelty: list[float]
    sections: list[Section]
