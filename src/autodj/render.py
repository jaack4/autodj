"""Render a MixPlan.

Every track plays at its own tempo. During a blend the incoming track is time-stretched to
the outgoing track's tempo, then glides to its own over `ramp_bars`, like easing a pitch
fader back. Bars of the two tracks are lined up (plus a nudge for local grid offsets),
3-band EQ gains are automated, decks are rendered and summed one at a time, and the master
is limited.
"""

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pedalboard
from scipy.signal import butter, sosfiltfilt

from autodj.analysis import analyze_file
from autodj.analysis.beats import MIN_CONFIDENCE, onset_offset
from autodj.models import Analysis, MixPlan, PlanTrack, Transition
from autodj.planner import region_offset_ms

LOW_HZ = 200.0
HIGH_HZ = 2500.0
BANDS = ("low", "mid", "high")
SWAP_RAMP_S = 0.05  # bass swap completes exactly on the downbeat
CEILING_DBFS = -1.0
CURVE_POINTS = 32
TIME_MAP_STEP = 0.005  # seconds

Breakpoints = list[tuple[float, float]]  # (mix time s, gain)
TempoCurve = list[tuple[float, float]]  # (track-local input time s, stretch factor)


@dataclass
class Deck:
    """One track on the mix timeline. Times are 'input' (original audio) or 'output' (stretched)."""

    analysis: Analysis
    target_bpm: float  # tempo when playing solo
    gain: float  # loudness match to the plan's target LUFS
    curve: TempoCurve = field(default_factory=list)  # empty: constant target tempo
    offset: float = 0.0  # mix time of the deck's output t=0
    play_from: float = 0.0  # output seconds
    play_to: float = 0.0
    automation: dict[str, Breakpoints] = field(default_factory=lambda: {b: [] for b in BANDS})
    _map_in: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _map_out: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def native_bpm(self) -> float:
        return self.analysis.bpm

    def bar_input_time(self, bar: int) -> float:
        """Original-audio time of a bar start (extrapolates past the last bar)."""
        db = np.asarray(self.analysis.downbeats)
        bar_len = float(np.diff(db).mean())
        return float(db[bar] if bar < len(db) else db[-1] + (bar - len(db) + 1) * bar_len)

    def factor_at(self, t_in: float | np.ndarray) -> np.ndarray:
        if not self.curve:
            return np.full_like(np.asarray(t_in, float), self.target_bpm / self.native_bpm)
        ts, fs = zip(*self.curve)
        return np.interp(t_in, ts, fs)

    def tempo_at_bar(self, bar: int) -> float:
        return float(self.factor_at(self.bar_input_time(bar))) * self.native_bpm

    def build_time_map(self) -> None:
        """Output time of each input time: the integral of 1 / stretch factor."""
        t = np.arange(0, self.analysis.duration + 1.0, TIME_MAP_STEP)
        f = self.factor_at(t)
        self._map_in = t
        self._map_out = np.concatenate([[0.0], np.cumsum(TIME_MAP_STEP / f[:-1])])

    def to_out(self, t_in: float) -> float:
        return float(np.interp(t_in, self._map_in, self._map_out))

    def bar_time(self, bar: int) -> float:
        """Track-local output time of a bar start."""
        return self.to_out(self.bar_input_time(bar))

    def load_stretched(self, sr: int) -> np.ndarray:
        audio, _ = librosa.load(self.analysis.path, sr=sr, mono=False)
        audio = np.atleast_2d(audio).astype(np.float32)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        factor = self.factor_at(np.arange(audio.shape[1]) / sr)
        constant = np.allclose(factor, factor[0])
        # Rubber Band's R3 engine (high_quality=True) drifts up to ~50 ms within a track at 5%
        # stretch, which flams every kick. R2 holds beats within 0.2 ms at a constant factor and
        # ~4 ms through a ramp (see tests/test_render.py).
        return pedalboard.time_stretch(
            audio, sr, stretch_factor=float(factor[0]) if constant else factor.astype(np.float64), high_quality=False
        )


@dataclass
class TransitionWindow:
    transition: Transition
    start: float  # mix times
    swap: float
    end: float
    beat: float  # output seconds per beat during the blend
    out_deck: Deck
    in_deck: Deck
    nudge_ms: float = 0.0  # applied to the incoming deck
    nudge_source: str = "none"  # "measured" (closed loop) or "analysis" (fallback)
    out_audio: np.ndarray | None = None  # mono, unautomated, as placed in the mix
    in_audio: np.ndarray | None = None


def _fade(t0: float, t1: float, rising: bool) -> Breakpoints:
    """Equal-power fade."""
    x = np.linspace(0, 1, CURVE_POINTS)
    g = np.sin(x * np.pi / 2) if rising else np.cos(x * np.pi / 2)
    return list(zip(t0 + x * (t1 - t0), g))


def _bass_swap_blend(out: Deck, inc: Deck, t0: float, t1: float, ts: float) -> None:
    """Incoming highs/mids fade in until the swap; basslines swap on the swap downbeat;
    outgoing highs/mids fade out after it."""
    inc.automation["low"] += [(ts - SWAP_RAMP_S, 0.0), (ts, 1.0)]
    out.automation["low"] += [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]
    for band in ("mid", "high"):
        inc.automation[band] += _fade(t0, ts, rising=True)
        out.automation[band] += _fade(ts, t1, rising=False)


def _split_bands(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Zero-phase 3-band split that sums back to the input exactly."""
    low = sosfiltfilt(butter(4, LOW_HZ, "lowpass", fs=sr, output="sos"), x, axis=-1)
    high = sosfiltfilt(butter(4, HIGH_HZ, "highpass", fs=sr, output="sos"), x, axis=-1)
    return {"low": low, "mid": x - low - high, "high": high}


def make_deck(track: PlanTrack, plan: MixPlan) -> Deck:
    a = analyze_file(Path(track.path), bpm_hint=track.bpm_hint)
    target = plan.bpm or track.bpm or a.bpm
    return Deck(analysis=a, target_bpm=target, gain=10 ** ((plan.target_lufs - a.lufs) / 20))


def _check_transition(tr: Transition, a: Deck, b: Deck) -> None:
    n_a, n_b = a.analysis.n_bars, b.analysis.n_bars
    if not 0 <= tr.out_bar < n_a:
        raise ValueError(f"out_bar {tr.out_bar} outside {Path(a.analysis.path).name} ({n_a} bars)")
    if not 0 <= tr.in_bar < n_b or tr.in_bar + tr.overlap_bars > n_b:
        raise ValueError(f"in_bar {tr.in_bar} + overlap doesn't fit {Path(b.analysis.path).name} ({n_b} bars)")
    if tr.swap_bar is not None and not 0 < tr.swap_bar < tr.overlap_bars:
        raise ValueError("swap_bar must fall inside the overlap")


def _tempo_curves(plan: MixPlan, decks: list[Deck]) -> None:
    """Incoming decks match the outgoing tempo during the blend, then ramp to their own."""
    for i, (tr, a, b) in enumerate(zip(plan.transitions, decks, decks[1:])):
        match = a.tempo_at_bar(tr.out_bar) / b.native_bpm
        settle = b.target_bpm / b.native_bpm
        blend_end = tr.in_bar + tr.overlap_bars
        ramp_end = blend_end + tr.ramp_bars
        if i + 1 < len(plan.transitions):  # finish the ramp before b's own outgoing blend
            next_out = plan.transitions[i + 1].out_bar
            if next_out < blend_end:
                raise ValueError(f"{Path(b.analysis.path).name}: its next blend starts before this one ends")
            ramp_end = min(ramp_end, next_out)
        if abs(match - settle) < 1e-9:
            b.curve = []
        elif ramp_end == blend_end:  # no room to ramp: stay at the matched tempo
            b.curve = [(0.0, match), (1.0, match)]
        else:
            b.curve = [(0.0, match), (b.bar_input_time(blend_end), match), (b.bar_input_time(ramp_end), settle)]


def _analysis_offset_ms(d: Deck, start_bar: int, end_bar: int) -> float:
    """The analysis' estimate of a deck's local kick offset over some bars, in output ms."""
    off, _ = region_offset_ms(d.analysis, start_bar, end_bar)
    return off / float(d.factor_at(d.bar_input_time(start_bar)))


def _measured_offset_ms(d: Deck, audio: np.ndarray, start_bar: int, end_bar: int, sr: int) -> tuple[float, float]:
    """Measure a deck's kicks against its own mapped grid over some bars of the stretched audio."""
    t0, t1 = d.bar_time(start_bar), d.bar_time(end_bar)
    y = audio[:, int(round(t0 * sr)) : int(round(t1 * sr))].mean(axis=0)
    beat = (t1 - t0) / (4 * (end_bar - start_bar))
    beats = np.arange(0, len(y) / sr - beat, beat)
    return onset_offset(y, sr, beats, beat, center_ms=_analysis_offset_ms(d, start_bar, end_bar))


def place(plan: MixPlan, decks: list[Deck]) -> list[TransitionWindow]:
    """Set tempo curves and position decks on the timeline (before nudging)."""
    if len(plan.transitions) != len(decks) - 1:
        raise ValueError("A plan needs exactly one transition per adjacent pair of tracks")
    for tr, a, b in zip(plan.transitions, decks, decks[1:]):
        _check_transition(tr, a, b)
    _tempo_curves(plan, decks)
    for d in decks:
        d.build_time_map()

    first = decks[0]
    start_bar = plan.tracks[0].start_bar
    first.play_from = first.bar_time(start_bar) if start_bar else 0.0
    first.offset = -first.play_from

    windows = []
    for tr, a, b in zip(plan.transitions, decks, decks[1:]):
        t0 = a.offset + a.bar_time(tr.out_bar)
        t1 = a.offset + a.bar_time(tr.out_bar + tr.overlap_bars)
        bar = (t1 - t0) / tr.overlap_bars
        b.play_from = b.bar_time(tr.in_bar)
        b.offset = t0 - b.play_from
        ts = t0 + (tr.swap_bar if tr.swap_bar is not None else tr.overlap_bars // 2) * bar
        a.play_to = t1 - a.offset
        windows.append(TransitionWindow(tr, t0, ts, t1, bar / 4, a, b))
    decks[-1].play_to = decks[-1].to_out(decks[-1].analysis.duration)
    return windows


def _deck_signal(d: Deck, audio: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """(mix start sample, automated audio) for the part of the deck that plays."""
    s0 = int(round(d.play_from * sr))
    s1 = min(int(round(d.play_to * sr)), audio.shape[1])
    seg = audio[:, s0:s1] * d.gain
    times = d.offset + np.arange(s0, s1) / sr
    out = np.zeros_like(seg)
    for band, x in _split_bands(seg, sr).items():
        points = d.automation[band]
        if points:
            ts, gs = zip(*points)
            out += x * np.interp(times, ts, gs)
        else:
            out += x
    return int(round((d.offset + d.play_from) * sr)), out


def _capture(d: Deck, audio: np.ndarray, w: TransitionWindow, sr: int) -> np.ndarray:
    s0 = max(int(round((w.start - d.offset) * sr)), 0)
    s1 = int(round((w.end - d.offset) * sr))
    return audio[:, s0:s1].mean(axis=0)


def _shift_from(decks: list[Deck], windows: list[TransitionWindow], i: int, delta: float) -> None:
    """Move deck i and everything after it (decks and the blends they anchor) by delta seconds."""
    for d in decks[i:]:
        d.offset += delta
    for w in windows[i:]:
        w.start, w.swap, w.end = w.start + delta, w.swap + delta, w.end + delta


def _nudge(w: TransitionWindow, out_audio: np.ndarray, in_audio: np.ndarray, sr: int) -> None:
    """Closed-loop nudge: measure both decks' kicks over the overlap and line them up, like
    nudging a platter. Falls back to the analysis' estimate when either side has too few kicks."""
    tr = w.transition
    a_bars = (tr.out_bar, tr.out_bar + tr.overlap_bars)
    b_bars = (tr.in_bar, tr.in_bar + tr.overlap_bars)
    off_a, conf_a = _measured_offset_ms(w.out_deck, out_audio, *a_bars, sr)
    off_b, conf_b = _measured_offset_ms(w.in_deck, in_audio, *b_bars, sr)
    if min(conf_a, conf_b) >= MIN_CONFIDENCE:
        w.nudge_ms, w.nudge_source = off_a - off_b, "measured"
    else:
        w.nudge_ms = _analysis_offset_ms(w.out_deck, *a_bars) - _analysis_offset_ms(w.in_deck, *b_bars)
        w.nudge_source = "analysis"


def mix_alignment(plan: MixPlan, w: TransitionWindow) -> list[tuple[float, float]]:
    """(kick-vs-master-grid offset ms, confidence) for the outgoing and incoming deck during an overlap.

    The difference between the two offsets is what you'd hear as flamming.
    """
    sr = plan.sample_rate
    result = []
    for y in (w.out_audio, w.in_audio):
        beats = np.arange(0, len(y) / sr - w.beat, w.beat)
        result.append(onset_offset(y, sr, beats, w.beat, center_ms=w.nudge_ms / 2))
    return result


def _add(mix: np.ndarray, m0: int, signal: np.ndarray) -> None:
    if m0 < 0:
        signal, m0 = signal[:, -m0:], 0
    n = min(signal.shape[1], mix.shape[1] - m0)
    mix[:, m0 : m0 + n] += signal[:, :n]


def render(plan: MixPlan) -> tuple[np.ndarray, list[Deck], list[TransitionWindow]]:
    sr = plan.sample_rate
    decks = [make_deck(t, plan) for t in plan.tracks]
    windows = place(plan, decks)
    end = max(d.offset + d.play_to for d in decks)
    mix = np.zeros((2, int(np.ceil(end * sr)) + 1 + sr), np.float32)  # +1 s of room for nudges
    prev_audio = None
    for i, d in enumerate(decks):  # one deck (plus the previous one) in memory at a time
        audio = d.load_stretched(sr)
        if i > 0:
            w = windows[i - 1]
            _nudge(w, prev_audio, audio, sr)
            _shift_from(decks, windows, i, w.nudge_ms / 1000)
            w.out_audio = _capture(decks[i - 1], prev_audio, w, sr)
            w.in_audio = _capture(d, audio, w, sr)
            _bass_swap_blend(w.out_deck, d, w.start, w.end, w.swap)
            _add(mix, *_deck_signal(decks[i - 1], prev_audio, sr))  # its automation is now complete
        prev_audio = audio
    _add(mix, *_deck_signal(decks[-1], prev_audio, sr))
    mix = mix[:, : int(np.ceil(max(d.offset + d.play_to for d in decks) * sr)) + 1]
    # JUCE's limiter treats the threshold as input gain and limits to 0 dBFS, so lower the
    # output afterwards to leave headroom for inter-sample peaks and lossy encoding.
    master = pedalboard.Pedalboard([
        pedalboard.Limiter(threshold_db=-3.0, release_ms=100),
        pedalboard.Gain(gain_db=CEILING_DBFS),
    ])
    return master(mix, sr), decks, windows
