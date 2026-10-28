"""Render a MixPlan: time-stretch every track to the master tempo, line up bars,
automate 3-band EQ gains for each transition, sum, and limit."""

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pedalboard
from scipy.signal import butter, sosfiltfilt

from autodj.analysis import analyze_file
from autodj.analysis.beats import MIN_CONFIDENCE, onset_offset
from autodj.models import Analysis, MixPlan, Transition
from autodj.planner import region_offset_ms

LOW_HZ = 200.0
HIGH_HZ = 2500.0
BANDS = ("low", "mid", "high")
SWAP_RAMP_S = 0.05  # bass swap completes exactly on the downbeat
CEILING_DBFS = -1.0
CURVE_POINTS = 32

Breakpoints = list[tuple[float, float]]  # (mix time s, gain)


@dataclass
class Deck:
    """One track, stretched to the master tempo and placed on the mix timeline."""

    analysis: Analysis
    audio: np.ndarray  # (2, samples), stretched
    rate: float  # stretch factor: master bpm / native bpm
    gain: float  # loudness match to the plan's target LUFS
    offset: float = 0.0  # mix time of the track's own t=0
    play_from: float = 0.0  # track-local (stretched) seconds
    play_to: float = 0.0
    automation: dict[str, Breakpoints] = field(default_factory=lambda: {b: [] for b in BANDS})

    def bar_time(self, bar: int) -> float:
        """Track-local stretched time of a bar start (extrapolates past the last bar)."""
        db = np.asarray(self.analysis.downbeats)
        bar_len = float(np.diff(db).mean())
        t = db[bar] if bar < len(db) else db[-1] + (bar - len(db) + 1) * bar_len
        return float(t) / self.rate


def _load_deck(path: str, plan: MixPlan) -> Deck:
    a = analyze_file(Path(path))
    audio, _ = librosa.load(a.path, sr=plan.sample_rate, mono=False)
    audio = np.atleast_2d(audio)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    rate = plan.bpm / a.bpm
    # Rubber Band's R3 engine (high_quality=True) drifts up to ~50 ms within a track at 5% stretch,
    # which flams every kick. R2 holds beat positions to within 0.2 ms (see tests/test_render.py).
    stretched = pedalboard.time_stretch(
        audio.astype(np.float32), plan.sample_rate, stretch_factor=rate, high_quality=False
    )
    return Deck(analysis=a, audio=stretched, rate=rate, gain=10 ** ((plan.target_lufs - a.lufs) / 20))


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


@dataclass
class TransitionWindow:
    start: float
    swap: float
    end: float
    out_deck: Deck
    in_deck: Deck


def place(plan: MixPlan, decks: list[Deck]) -> list[TransitionWindow]:
    """Position decks on the timeline and write their automation. Returns transition windows."""
    if len(plan.transitions) != len(decks) - 1:
        raise ValueError("A plan needs exactly one transition per adjacent pair of tracks")
    bar = 4 * 60 / plan.bpm
    first = decks[0]
    start_bar = plan.tracks[0].start_bar
    first.play_from = first.bar_time(start_bar) if start_bar else 0.0
    first.offset = -first.play_from

    windows = []
    for tr, a, b in zip(plan.transitions, decks, decks[1:]):
        _check_transition(tr, a, b)
        t0 = a.offset + a.bar_time(tr.out_bar)
        b.play_from = b.bar_time(tr.in_bar)
        b.offset = t0 - b.play_from + _nudge(tr, a, b)
        t1 = t0 + tr.overlap_bars * bar
        ts = t0 + (tr.swap_bar if tr.swap_bar is not None else tr.overlap_bars // 2) * bar
        a.play_to = t1 - a.offset
        _bass_swap_blend(a, b, t0, t1, ts)
        windows.append(TransitionWindow(t0, ts, t1, a, b))
    decks[-1].play_to = decks[-1].audio.shape[1] / plan.sample_rate
    return windows


def _nudge(tr: Transition, a: Deck, b: Deck) -> float:
    """Seconds to shift B so its kicks land on A's during the overlap, like nudging a platter.

    Corrects constant local offsets (e.g. after a splice) that a single track-wide grid can't.
    """
    off_a, _ = region_offset_ms(a.analysis, tr.out_bar, tr.out_bar + tr.overlap_bars)
    off_b, _ = region_offset_ms(b.analysis, tr.in_bar, tr.in_bar + tr.overlap_bars)
    return (off_a / a.rate - off_b / b.rate) / 1000


def _check_transition(tr: Transition, a: Deck, b: Deck) -> None:
    n_a, n_b = len(a.analysis.downbeats), len(b.analysis.downbeats)
    if not 0 <= tr.out_bar < n_a:
        raise ValueError(f"out_bar {tr.out_bar} outside {Path(a.analysis.path).name} ({n_a} bars)")
    if not 0 <= tr.in_bar < n_b or tr.in_bar + tr.overlap_bars > n_b:
        raise ValueError(f"in_bar {tr.in_bar} + overlap doesn't fit {Path(b.analysis.path).name} ({n_b} bars)")
    if tr.swap_bar is not None and not 0 < tr.swap_bar < tr.overlap_bars:
        raise ValueError("swap_bar must fall inside the overlap")


def _deck_signal(d: Deck, sr: int) -> tuple[int, np.ndarray]:
    """(mix start sample, automated audio) for the part of the deck that plays."""
    s0 = int(round(d.play_from * sr))
    s1 = min(int(round(d.play_to * sr)), d.audio.shape[1])
    seg = d.audio[:, s0:s1] * d.gain
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


def mix_alignment(plan: MixPlan, w: TransitionWindow) -> list[tuple[float, float]]:
    """(kick-vs-master-grid offset ms, confidence) for the outgoing and incoming deck during an overlap.

    The difference between the two offsets is what you'd hear as flamming.
    """
    sr = plan.sample_rate
    beat = 60 / plan.bpm
    result = []
    for d in (w.out_deck, w.in_deck):
        s0 = max(int(round((w.start - d.offset) * sr)), 0)
        s1 = int(round((w.end - d.offset) * sr))
        y = d.audio[:, s0:s1].mean(axis=0)
        result.append(onset_offset(y, sr, np.arange(0, (s1 - s0) / sr - beat, beat), beat))
    return result


def render(plan: MixPlan) -> tuple[np.ndarray, list[Deck], list[TransitionWindow]]:
    sr = plan.sample_rate
    decks = [_load_deck(t.path, plan) for t in plan.tracks]
    windows = place(plan, decks)
    end = max(d.offset + d.play_to for d in decks)
    mix = np.zeros((2, int(np.ceil(end * sr)) + 1), np.float32)
    for d in decks:
        m0, signal = _deck_signal(d, sr)
        mix[:, m0 : m0 + signal.shape[1]] += signal[:, : mix.shape[1] - m0]
    # JUCE's limiter treats the threshold as input gain and limits to 0 dBFS, so lower the
    # output afterwards to leave headroom for inter-sample peaks and lossy encoding.
    master = pedalboard.Pedalboard([
        pedalboard.Limiter(threshold_db=-3.0, release_ms=100),
        pedalboard.Gain(gain_db=CEILING_DBFS),
    ])
    return master(mix, sr), decks, windows
