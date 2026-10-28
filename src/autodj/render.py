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
from autodj.audio import load as load_audio
from autodj.analysis.beats import MIN_CONFIDENCE, onset_offset
from autodj.models import Analysis, MixPlan, PlanTrack, Transition
from autodj.planner import region_offset_ms

LOW_HZ = 200.0
HIGH_HZ = 2500.0
BANDS = ("low", "mid", "high")
SWAP_RAMP_S = 0.05  # bass swap completes exactly on the downbeat
CEILING_DBFS = -1.0
CURVE_POINTS = 32
CUT_FADE_S = 0.004  # hard cuts still get a few ms to avoid clicks
SWEEP_TOP_HZ = 2000.0  # outgoing high-pass ends here
SWEEP_BOTTOM_HZ = 250.0  # incoming low-pass starts here
ECHO_FLOOR = 0.02  # stop repeating below this level
ECHO_LOW_HZ, ECHO_HIGH_HZ = 300.0, 6000.0
CUT_ALIGN_BARS = 4  # bars either side of a cut used to line up the groove
TIME_MAP_STEP = 0.005  # seconds

Breakpoints = list[tuple[float, float]]  # (mix time s, gain)
TempoCurve = list[tuple[float, float]]  # (track-local input time s, stretch factor)


@dataclass
class Deck:
    """One track on the mix timeline. Times are 'input' (original audio) or 'output' (stretched)."""

    analysis: Analysis
    target_bpm: float  # tempo when playing solo
    gain: float  # loudness match to the plan's target LUFS
    semitones: int = 0  # pitch shift for key matching
    curve: TempoCurve = field(default_factory=list)  # empty: constant target tempo
    offset: float = 0.0  # mix time of the deck's output t=0
    play_from: float = 0.0  # output seconds
    play_to: float = 0.0
    automation: dict[str, Breakpoints] = field(default_factory=lambda: {b: [] for b in BANDS})
    filters: list[tuple[str, Breakpoints]] = field(default_factory=list)  # ("highpass"|"lowpass", (mix t, Hz))
    stem_gains: dict[str, Breakpoints] = field(default_factory=dict)  # drums / bass / other / vocals -> gains
    echo: "Echo | None" = None
    wash: "Wash | None" = None
    roll: float | None = None  # beat roll: mix time of the cut it rolls into
    tape_stop: float | None = None  # tape stop: mix time A comes to a halt
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

    def load_stretched(self, sr: int, source: str | None = None) -> np.ndarray:
        """The deck's audio (or one of its stems), time-stretched along its tempo curve."""
        audio, _ = load_audio(source or self.analysis.path, sr=sr, mono=False)
        audio = np.atleast_2d(audio).astype(np.float32)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        factor = self.factor_at(np.arange(audio.shape[1]) / sr)
        constant = np.allclose(factor, factor[0])
        # Rubber Band's R3 engine (high_quality=True) drifts up to ~50 ms within a track at 5%
        # stretch, which flams every kick. R2 holds beats within 0.2 ms at a constant factor and
        # ~4 ms through a ramp (see tests/test_render.py).
        return pedalboard.time_stretch(
            audio, sr, stretch_factor=float(factor[0]) if constant else factor.astype(np.float64), high_quality=False,
            pitch_shift_in_semitones=float(self.semitones),
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


@dataclass
class Echo:
    """Tempo-synced echo of the outgoing deck's last beat, engaged as the dry signal is cut."""

    cut: float  # mix time
    beat: float  # seconds
    delay_beats: float
    feedback: float

    @property
    def repeats(self) -> int:
        return int(np.ceil(np.log(ECHO_FLOOR) / np.log(self.feedback)))

    @property
    def tail(self) -> float:
        return self.repeats * self.delay_beats * self.beat


def _sweep(t0: float, t1: float, f0: float, f1: float) -> Breakpoints:
    """Exponential cutoff sweep (even steps in pitch)."""
    x = np.linspace(0, 1, CURVE_POINTS)
    return list(zip(t0 + x * (t1 - t0), f0 * (f1 / f0) ** x))


def _cut(out: Deck, inc: Deck, t: float) -> None:
    for band in BANDS:
        out.automation[band] += [(t - CUT_FADE_S, 1.0), (t, 0.0)]
        inc.automation[band] += [(t - CUT_FADE_S, 0.0), (t, 1.0)]


INSTRUMENTAL = ("drums", "bass", "other")
STEM_TRANSITIONS = ("acapella_overlay", "stem_blend", "melody_first", "double_drop")  # need separated stems
CUT_TYPES = ("echo_out", "drop_cut", "beat_roll", "tape_stop")
WASH_TAIL_BARS = 2
ROLL_PATTERN = (1, 1, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125)  # beats; sums to one bar
OVERLAY_BLEND_IN_BARS = 4  # B's beat fades in over the overlay's last bars
OVERLAY_BLEND_OUT_BARS = 2  # A's beat fades out after the bass swap


def _acapella_overlay(out: Deck, inc: Deck, t0: float, t1: float, beat: float) -> None:
    """A's vocals duck as B's acapella starts over A's instrumental. Over the last bars B's beat
    fades in (no bass yet), the basslines swap on the overlay's final downbeat, then A fades out."""
    bar = 4 * beat
    blend_in = max(t0, t1 - OVERLAY_BLEND_IN_BARS * bar)
    out.stem_gains.setdefault("vocals", []).extend([(t0 - beat, 1.0), (t0, 0.0)])
    for stem in INSTRUMENTAL:
        inc.stem_gains.setdefault(stem, []).extend(_fade(blend_in, t1, rising=True))
    inc.automation["low"] += [(t1 - SWAP_RAMP_S, 0.0), (t1, 1.0)]
    out.automation["low"] += [(t1 - SWAP_RAMP_S, 1.0), (t1, 0.0)]
    for band in ("mid", "high"):
        out.automation[band] += _fade(t1, t1 + OVERLAY_BLEND_OUT_BARS * bar, rising=False)


@dataclass
class Wash:
    """A reverb on the outgoing deck that rises as it fades, leaving a tail."""

    start: float
    peak: float  # wet level reaches its maximum here (the swap)
    end: float
    tail: float  # seconds of reverb tail after the end


def _melody_first(out: Deck, inc: Deck, t0: float, ts: float, beat: float) -> None:
    """B's melody and vocals arrive over A's groove (A's own melody and vocals step aside);
    the rhythm section swaps late, on ts."""
    bar = 4 * beat
    for stem in ("other", "vocals"):
        inc.stem_gains[stem] = _fade(t0, ts, rising=True)
    out.stem_gains["other"] = _fade(t0, ts, rising=False)
    out.stem_gains["vocals"] = _fade(t0, t0 + 2 * bar, rising=False)
    for stem in ("drums", "bass"):
        inc.stem_gains[stem] = [(ts - SWAP_RAMP_S, 0.0), (ts, 1.0)]
        out.stem_gains[stem] = [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]
    inc.automation["low"] += [(ts - SWAP_RAMP_S, 0.0), (ts, 1.0)]
    out.automation["low"] += [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]


def _double_drop(out: Deck, inc: Deck, t0: float, t1: float) -> None:
    """Both drops land together: A's drums and melody, B's bass, melody and vocals. A leaves at t1,
    when B's drums take over."""
    for stem in ("bass", "vocals"):
        out.stem_gains[stem] = [(t0 - SWAP_RAMP_S, 1.0), (t0, 0.0)]
    inc.stem_gains["drums"] = [(t1 - SWAP_RAMP_S, 0.0), (t1, 1.0)]
    for band in BANDS:
        out.automation[band] += [(t1 - CUT_FADE_S, 1.0), (t1, 0.0)]


def _wash_tail(seg: np.ndarray, times: np.ndarray, w: Wash, sr: int) -> tuple[int, np.ndarray]:
    """(start index, wet signal) for a reverb wash: full-wet reverb of the deck, rising to w.peak."""
    i0 = int(np.clip(np.searchsorted(times, w.start), 0, seg.shape[1]))
    i1 = int(np.clip(np.searchsorted(times, w.end), i0, seg.shape[1]))
    padded = np.concatenate([seg[:, i0:i1], np.zeros((2, int(w.tail * sr)), np.float32)], axis=1)
    reverb = pedalboard.Pedalboard([
        pedalboard.HighpassFilter(cutoff_frequency_hz=300.0),
        pedalboard.Reverb(room_size=0.92, damping=0.4, wet_level=1.0, dry_level=0.0, width=1.0),
    ])
    wet = reverb(padded.astype(np.float32), sr)
    t = w.start + np.arange(wet.shape[1]) / sr
    return i0, wet * np.interp(t, [w.start, w.peak], [0.0, 0.8])


def _beat_roll(out: np.ndarray, times: np.ndarray, cut: float, beat: float, sr: int) -> None:
    """Rebuild the bar before the cut as ever-shorter repeats of its first beat."""
    b = int(round(beat * sr))
    r0 = int(np.searchsorted(times, cut - 4 * beat))
    if r0 + 4 * b > out.shape[1] or r0 < 0:
        return
    src = out[:, r0 : r0 + b].copy()
    pieces = []
    for length in ROLL_PATTERN:
        piece = src[:, : max(8, int(round(length * b)))].copy()
        edge = min(int(0.002 * sr), piece.shape[1] // 4)
        if edge:
            piece[:, :edge] *= np.linspace(0, 1, edge)
            piece[:, -edge:] *= np.linspace(1, 0, edge)
        pieces.append(piece)
    roll = np.concatenate(pieces, axis=1)[:, : 4 * b]
    out[:, r0 : r0 + roll.shape[1]] = roll


def _tape_stop(out: np.ndarray, times: np.ndarray, stop: float, beat: float, sr: int) -> None:
    """The beat before `stop` slows from full speed to a halt (and drops in pitch, like a turntable)."""
    n = int(round(beat * sr))
    r0 = int(np.searchsorted(times, stop - beat))
    if r0 < 0 or r0 + n > out.shape[1]:
        return
    k = np.arange(n) / n
    read = r0 + (k - k**2 / 2) * n  # position read at speed 1 - k
    stopped = np.stack([np.interp(read, np.arange(out.shape[1]), ch) for ch in out])
    fade = n // 10
    stopped[:, -fade:] *= np.linspace(1, 0, fade)
    out[:, r0 : r0 + n] = stopped


def _stem_blend(out: Deck, inc: Deck, t0: float, t1: float, ts: float, beat: float) -> None:
    """Swap the parts one at a time instead of crossfading whole tracks:
    1st half: B's drums creep in (its kick and bass held back by the low band);
    swap downbeat: drums and bass change over;
    2nd half: melodies crossfade while A's vocals carry on over B's beat;
    end: A's vocals finish over a bar as B's come in."""
    bar = 4 * beat
    inc.automation["low"] += [(ts - SWAP_RAMP_S, 0.0), (ts, 1.0)]
    out.automation["low"] += [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]
    inc.stem_gains["drums"] = _fade(t0, ts, rising=True)
    out.stem_gains["drums"] = [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]
    inc.stem_gains["bass"] = [(ts - SWAP_RAMP_S, 0.0), (ts, 1.0)]
    out.stem_gains["bass"] = [(ts - SWAP_RAMP_S, 1.0), (ts, 0.0)]
    inc.stem_gains["other"] = _fade(ts, t1, rising=True)
    out.stem_gains["other"] = _fade(ts, t1, rising=False)
    inc.stem_gains["vocals"] = _fade(t1, t1 + bar, rising=True)
    out.stem_gains["vocals"] = _fade(t1, t1 + bar, rising=False)


def apply_transition(w: "TransitionWindow") -> None:
    """Write the automation for one transition onto its two decks."""
    tr, a, b = w.transition, w.out_deck, w.in_deck
    if tr.type == "bass_swap_blend":
        _bass_swap_blend(a, b, w.start, w.end, w.swap)
    elif tr.type == "filter_sweep":
        _bass_swap_blend(a, b, w.start, w.end, w.swap)
        a.filters.append(("highpass", _sweep(w.start, w.end, 20.0, SWEEP_TOP_HZ)))
        b.filters.append(("lowpass", _sweep(w.start, w.swap, SWEEP_BOTTOM_HZ, 20000.0)))
    elif tr.type == "echo_out":
        _cut(a, b, w.start)
        a.echo = Echo(cut=w.start, beat=w.beat, delay_beats=tr.echo_delay_beats, feedback=tr.echo_feedback)
    elif tr.type == "drop_cut":
        _cut(a, b, w.start)
    elif tr.type == "acapella_overlay":
        _acapella_overlay(a, b, w.start, w.end, w.beat)
    elif tr.type == "stem_blend":
        _stem_blend(a, b, w.start, w.end, w.swap, w.beat)
    elif tr.type == "melody_first":
        _melody_first(a, b, w.start, w.swap, w.beat)
    elif tr.type == "reverb_wash":
        _bass_swap_blend(a, b, w.start, w.end, w.swap)
        a.filters.append(("highpass", _sweep(w.swap, w.end, 20.0, 800.0)))
        a.wash = Wash(w.start, w.swap, w.end, WASH_TAIL_BARS * 4 * w.beat)
    elif tr.type == "double_drop":
        _double_drop(a, b, w.start, w.end)
    elif tr.type == "beat_roll":
        _cut(a, b, w.start)
        a.roll = w.start
        a.filters.append(("highpass", _sweep(w.start - 4 * w.beat, w.start, 20.0, 1200.0)))
    elif tr.type == "tape_stop":
        _cut(a, b, w.start)
        a.tape_stop = w.start
    else:
        raise ValueError(f"Unknown transition type: {tr.type}")


def _apply_filters(x: np.ndarray, times: np.ndarray, filters: list[tuple[str, Breakpoints]], sr: int) -> np.ndarray:
    """Time-varying 4th-order Butterworth-shaped filters, applied as an STFT magnitude mask."""
    n_fft, hop = 2048, 512
    spec = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    freqs = np.maximum(librosa.fft_frequencies(sr=sr, n_fft=n_fft), 1.0)[:, None]
    frame_t = times[0] + np.arange(spec.shape[-1]) * hop / sr
    mask = np.ones((len(freqs), spec.shape[-1]))
    for kind, points in filters:
        ts, fs = zip(*points)
        cutoff = np.exp(np.interp(frame_t, ts, np.log(fs)))[None, :]
        ratio = freqs / cutoff if kind == "lowpass" else cutoff / freqs
        mask *= 1 / np.sqrt(1 + ratio**8)
    return librosa.istft(spec * mask, hop_length=hop, length=x.shape[-1])


def _echo_tail(seg: np.ndarray, seg_t0: float, e: Echo, sr: int) -> tuple[int, np.ndarray]:
    """(start index in seg, tail audio): repeats of the beat before the cut, filtered thin."""
    src_end = int(round((e.cut - seg_t0) * sr))
    src_start = max(0, src_end - int(round(e.beat * sr)))
    src = seg[:, src_start:src_end].copy()
    edge = min(int(0.01 * sr), src.shape[1] // 2)
    if edge:
        ramp = np.linspace(0, 1, edge)
        src[:, :edge] *= ramp
        src[:, -edge:] *= ramp[::-1]
    step = int(round(e.delay_beats * e.beat * sr))
    tail = np.zeros((2, e.repeats * step + src.shape[1]))
    for k in range(1, e.repeats + 1):
        tail[:, k * step : k * step + src.shape[1]] += e.feedback**k * src
    tail = sosfiltfilt(butter(2, [ECHO_LOW_HZ, ECHO_HIGH_HZ], "bandpass", fs=sr, output="sos"), tail, axis=-1)
    return src_start, tail


def _split_bands(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Zero-phase 3-band split that sums back to the input exactly."""
    low = sosfiltfilt(butter(4, LOW_HZ, "lowpass", fs=sr, output="sos"), x, axis=-1)
    high = sosfiltfilt(butter(4, HIGH_HZ, "highpass", fs=sr, output="sos"), x, axis=-1)
    return {"low": low, "mid": x - low - high, "high": high}


def make_deck(track: PlanTrack, plan: MixPlan) -> Deck:
    a = analyze_file(Path(track.path), bpm_hint=track.bpm_hint)
    target = plan.bpm or track.bpm or a.bpm
    return Deck(analysis=a, target_bpm=target, gain=10 ** ((plan.target_lufs - a.lufs) / 20), semitones=track.semitones)


def _check_transition(tr: Transition, a: Deck, b: Deck) -> None:
    n_a, n_b = a.analysis.n_bars, b.analysis.n_bars
    if not 0 <= tr.out_bar < n_a:
        raise ValueError(f"out_bar {tr.out_bar} outside {Path(a.analysis.path).name} ({n_a} bars)")
    if not 0 <= tr.in_bar < n_b or tr.in_bar + tr.overlap_bars > n_b:
        raise ValueError(f"in_bar {tr.in_bar} + overlap doesn't fit {Path(b.analysis.path).name} ({n_b} bars)")
    is_cut = tr.type in CUT_TYPES
    if is_cut != (tr.overlap_bars == 0):
        raise ValueError(f"{tr.type} needs overlap_bars {'== 0' if is_cut else '> 0'}")
    if tr.swap_bar is not None and not is_cut and not 0 < tr.swap_bar < tr.overlap_bars:
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


def _measured_offset_ms(d: Deck, audio: np.ndarray, start_bar: int, end_bar: int, sr: int) -> tuple[float, bool]:
    """Measure a deck's kicks against its own mapped grid over some bars of the stretched audio.

    Returns (offset ms, reliable). Unreliable: too few kicks, or the best match sits at the edge
    of the +-1/4 beat search (the real alignment is probably outside it).
    """
    t0, t1 = d.bar_time(start_bar), d.bar_time(end_bar)
    y = audio[:, int(round(t0 * sr)) : int(round(t1 * sr))].mean(axis=0)
    beat = (t1 - t0) / (4 * (end_bar - start_bar))
    beats = np.arange(0, len(y) / sr - beat, beat)
    center = _analysis_offset_ms(d, start_bar, end_bar)
    off, confidence = onset_offset(y, sr, beats, beat, center_ms=center)
    at_edge = abs(off - center) >= 0.24 * beat * 1000
    return off, confidence >= MIN_CONFIDENCE and not at_edge


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
        bar = a.bar_time(tr.out_bar + 1) - a.bar_time(tr.out_bar)
        b.play_from = b.bar_time(tr.in_bar)
        b.offset = t0 - b.play_from
        ts = t0 + (tr.swap_bar if tr.swap_bar is not None else tr.overlap_bars // 2) * bar
        a.play_to = t1 - a.offset
        if tr.type == "echo_out":
            a.play_to += Echo(t0, bar / 4, tr.echo_delay_beats, tr.echo_feedback).tail + bar / 4
        elif tr.type in ("drop_cut", "beat_roll", "tape_stop", "double_drop"):
            a.play_to += CUT_FADE_S
        elif tr.type == "reverb_wash":
            a.play_to += WASH_TAIL_BARS * bar
        elif tr.type == "acapella_overlay":
            a.play_to += OVERLAY_BLEND_OUT_BARS * bar
        elif tr.type == "stem_blend":
            a.play_to += bar  # A's vocals finish their phrase over B
        windows.append(TransitionWindow(tr, t0, ts, t1, bar / 4, a, b))
    decks[-1].play_to = decks[-1].to_out(decks[-1].analysis.duration)
    return windows


def _segment(audio: np.ndarray, s0: int, n: int, gain: float) -> np.ndarray:
    seg = np.zeros((2, n), np.float32)
    part = audio[:, s0 : s0 + n]
    seg[:, : part.shape[1]] = part * gain  # zero-padded if an echo tail runs past the audio
    return seg


def _deck_signal(d: Deck, audio: np.ndarray, sr: int, stems: dict[str, np.ndarray] | None = None) -> tuple[int, np.ndarray]:
    """(mix start sample, automated audio) for the part of the deck that plays.

    Stem gains are applied as mix + sum((gain - 1) * stem), so untouched stems leave the
    original mix exactly as it was (no separation artefacts unless a stem is changed).
    """
    s0 = int(round(d.play_from * sr))
    n = int(round(d.play_to * sr)) - s0
    seg = _segment(audio, s0, n, d.gain)
    times = d.offset + np.arange(s0, s0 + n) / sr
    for stem, points in d.stem_gains.items():
        if stems is None or stem not in stems:
            raise ValueError(f"{Path(d.analysis.path).name}: stem {stem!r} needed but not loaded")
        seg += (np.interp(times, *zip(*points)) - 1.0) * _segment(stems[stem], s0, n, d.gain)
    out = np.zeros_like(seg)
    for band, x in _split_bands(seg, sr).items():
        points = d.automation[band]
        if points:
            ts, gs = zip(*points)
            out += x * np.interp(times, ts, gs)
        else:
            out += x
    beat = 60 / d.target_bpm
    if d.roll is not None:
        _beat_roll(out, times, d.roll, beat, sr)
    if d.tape_stop is not None:
        _tape_stop(out, times, d.tape_stop, beat, sr)
    if d.filters:
        out = _apply_filters(out, times, d.filters, sr)
    if d.wash:
        start, wet = _wash_tail(seg, times, d.wash, sr)
        end = min(start + wet.shape[1], n)
        out[:, start:end] += wet[:, : end - start]
    if d.echo:
        start, tail = _echo_tail(seg, times[0], d.echo, sr)
        end = min(start + tail.shape[1], n)
        out[:, start:end] += tail[:, : end - start]
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
    if tr.overlap_bars:
        a_bars = (tr.out_bar, tr.out_bar + tr.overlap_bars)
        b_bars = (tr.in_bar, tr.in_bar + tr.overlap_bars)
    else:  # a cut: keep the groove continuous across it
        a_bars = (max(0, tr.out_bar - CUT_ALIGN_BARS), tr.out_bar)
        b_bars = (tr.in_bar, min(w.in_deck.analysis.n_bars, tr.in_bar + CUT_ALIGN_BARS))
    off_a, ok_a = _measured_offset_ms(w.out_deck, out_audio, *a_bars, sr)
    off_b, ok_b = _measured_offset_ms(w.in_deck, in_audio, *b_bars, sr)
    if ok_a and ok_b:
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


MASHUP_LEVEL_DB = -2.0  # guest vocals sit just under the host


def _render_mashups(plan: MixPlan, decks: list[Deck], mix: np.ndarray, sr: int, log) -> None:
    """Lay each mashup's guest vocal over its host, tempo-matched, key-shifted and bar-aligned."""
    from autodj.stems import separate

    for m in plan.mashups:
        host, track = decks[m.host], plan.tracks[m.guest]
        g = analyze_file(Path(track.path), bpm_hint=track.bpm_hint)
        guest = Deck(analysis=g, target_bpm=host.tempo_at_bar(m.host_bar),
                     gain=10 ** ((plan.target_lufs - g.lufs + MASHUP_LEVEL_DB) / 20), semitones=m.semitones)
        guest.build_time_map()
        beat = 60 / guest.target_bpm
        start, end = guest.bar_time(m.guest_bar), guest.bar_time(m.guest_bar + m.bars)
        guest.play_from, guest.play_to = max(0.0, start - beat), end + beat  # room for pickups and tails
        off_host, _ = region_offset_ms(host.analysis, m.host_bar, m.host_bar + m.bars)
        off_guest, _ = region_offset_ms(g, m.guest_bar, m.guest_bar + m.bars)
        nudge = (off_host / float(host.factor_at(host.bar_input_time(m.host_bar)))
                 - off_guest / float(guest.factor_at(guest.bar_input_time(m.guest_bar)))) / 1000
        guest.offset = host.offset + host.bar_time(m.host_bar) - start + nudge
        t0, t1 = guest.offset + guest.play_from, guest.offset + guest.play_to
        for band in BANDS:
            guest.automation[band] = [(t0, 0.0), (t0 + beat, 1.0), (t1 - beat, 1.0), (t1, 0.0)]
        log(f"mashup: {Path(g.path).stem} vocal over {Path(host.analysis.path).stem}")
        vocals = guest.load_stretched(sr, source=str(separate(g.path, g.sha256)["vocals"]))
        _add(mix, *_deck_signal(guest, vocals, sr))


def load_stems(d: Deck, sr: int) -> dict[str, np.ndarray]:
    """All four stems, stretched and shifted exactly like the deck's mix."""
    from autodj.stems import STEM_NAMES, separate

    paths = separate(d.analysis.path, d.analysis.sha256)
    return {name: d.load_stretched(sr, source=str(paths[name])) for name in STEM_NAMES}


def _add(mix: np.ndarray, m0: int, signal: np.ndarray) -> None:
    if m0 < 0:
        signal, m0 = signal[:, -m0:], 0
    n = min(signal.shape[1], mix.shape[1] - m0)
    mix[:, m0 : m0 + n] += signal[:, :n]


def render(plan: MixPlan, log=lambda msg: None) -> tuple[np.ndarray, list[Deck], list[TransitionWindow]]:
    sr = plan.sample_rate
    decks = [make_deck(t, plan) for t in plan.tracks]
    windows = place(plan, decks)
    end = max(d.offset + d.play_to for d in decks)
    mix = np.zeros((2, int(np.ceil(end * sr)) + 1 + sr), np.float32)  # +1 s of room for nudges
    uses_stems = [t.type in STEM_TRANSITIONS for t in plan.transitions]
    needs_stems = [(i > 0 and uses_stems[i - 1]) or (i < len(uses_stems) and uses_stems[i]) for i in range(len(decks))]
    stems: list[dict[str, np.ndarray] | None] = [None] * len(decks)
    prev_audio = None
    for i, d in enumerate(decks):  # one deck (plus the previous one) in memory at a time
        log(f"[{i + 1}/{len(decks)}] rendering {Path(d.analysis.path).name}")
        audio = d.load_stretched(sr)
        if needs_stems[i]:
            stems[i] = load_stems(d, sr)
        if i > 0:
            w = windows[i - 1]
            _nudge(w, prev_audio, audio, sr)
            _shift_from(decks, windows, i, w.nudge_ms / 1000)
            if w.transition.overlap_bars:
                w.out_audio = _capture(decks[i - 1], prev_audio, w, sr)
                w.in_audio = _capture(d, audio, w, sr)
            apply_transition(w)
            _add(mix, *_deck_signal(decks[i - 1], prev_audio, sr, stems[i - 1]))  # its automation is complete
            stems[i - 1] = None
        prev_audio = audio
    _add(mix, *_deck_signal(decks[-1], prev_audio, sr, stems[-1]))
    del prev_audio, stems
    _render_mashups(plan, decks, mix, sr, log)
    mix = mix[:, : int(np.ceil(max(d.offset + d.play_to for d in decks) * sr)) + 1]
    # JUCE's limiter treats the threshold as input gain and limits to 0 dBFS, so lower the
    # output afterwards to leave headroom for inter-sample peaks and lossy encoding.
    master = pedalboard.Pedalboard([
        pedalboard.Limiter(threshold_db=-3.0, release_ms=100),
        pedalboard.Gain(gain_db=CEILING_DBFS),
    ])
    return master(mix, sr), decks, windows
