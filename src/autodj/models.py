from typing import Literal

from pydantic import BaseModel, PrivateAttr


class Section(BaseModel):
    label: str  # intro | build | drop | break | body | outro
    start_bar: int
    end_bar: int  # exclusive
    start: float  # seconds
    end: float
    energy: float  # 0..1, relative to the track
    group: str | None = None  # sections sharing a letter repeat each other (e.g. drop 1 and drop 2)


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
    onset_offset_ms: float  # median onset-vs-grid offset; constant offsets are harmless
    grid_drift_ms: float  # onset-vs-grid drift across the track
    grid_ok: bool  # False if the track isn't constant-tempo (or the BPM is off)

    key: str  # Camelot, e.g. "8A"
    key_name: str
    key_confidence: float  # how far the best key beats keys 2+ Camelot steps away (~0..0.4)
    key_source: str = "mix"  # "stems" once re-detected from the harmonic stems
    lufs: float

    bar_energy: list[float]
    bar_novelty: list[float]
    bar_offset_ms: list[float]  # local kick-vs-grid offset; jumps mark splices or tempo changes
    bar_offset_confidence: list[float]  # < 1.5: too few kicks to measure
    sections: list[Section]
    bar_vocals: list[float] | None = None  # 0..1 vocal presence per bar, once stems are separated
    sections_source: str = "mix"  # "stems" once sections are re-detected from separated stems
    sections_mix: list[Section] | None = None  # the mix-only sections, kept for comparison
    _memo: dict = PrivateAttr(default_factory=dict)  # derived values cached by the planner

    @property
    def has_vocals_data(self) -> bool:
        return self.bar_vocals is not None

    @property
    def has_steady_beat(self) -> bool:
        """At least a third of the track has kicks the grid locks onto."""
        conf = self.bar_offset_confidence
        return sum(c >= 1.5 for c in conf) >= len(conf) / 3

    @property
    def n_bars(self) -> int:
        return len(self.downbeats)

    def section_at(self, bar: int) -> Section:
        for s in self.sections:
            if s.start_bar <= bar < s.end_bar:
                return s
        return self.sections[-1]

    def section_starts(self) -> set[int]:
        return {s.start_bar for s in self.sections} | {self.n_bars}


class PlanTrack(BaseModel):
    path: str
    start_bar: int = 0  # first track only: where playback begins
    bpm: float | None = None  # tempo the track settles at when playing solo (default: native)
    bpm_hint: float | None = None  # passed to analysis to resolve half/double time
    key: str | None = None  # user-provided key (Camelot), overrides detection when planning
    semitones: int = 0  # pitch shift for the whole track, to match keys with its neighbours
    title: str | None = None
    artist: str | None = None


class Transition(BaseModel):
    type: Literal["bass_swap_blend", "filter_sweep", "stem_blend", "echo_out", "drop_cut", "acapella_overlay"] = "bass_swap_blend"
    out_bar: int  # bar of the outgoing track where the overlap (or cut) starts
    in_bar: int = 0  # bar of the incoming track that lands on out_bar
    overlap_bars: int = 16  # 0 for echo_out / drop_cut
    swap_bar: int | None = None  # bar within the overlap where basslines swap (default: halfway)
    ramp_bars: int = 16  # after the blend, the incoming track glides to its own tempo over this many bars
    echo_delay_beats: float = 0.75  # echo_out: delay time
    echo_feedback: float = 0.55  # echo_out: level of each repeat
    notes: list[str] = []  # planner's reasons, for humans


class MixPlan(BaseModel):
    version: int = 2
    name: str = "mix"
    bpm: float | None = None  # if set, every track is locked to this tempo (no ramps)
    style: str = "smooth"  # "smooth": real blends only; "mixed": also cuts, echo outs, acapella overlays
    sample_rate: int = 44100
    target_lufs: float = -9.0  # per-track loudness before the master limiter
    tracks: list[PlanTrack]
    transitions: list[Transition]  # one per adjacent pair
    director: dict | None = None  # title, narrative, model, usage from the creative director (if used)
