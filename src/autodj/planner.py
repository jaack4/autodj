"""Choose where and how two tracks meet: phrase-aligned out/in points and a transition
type, scored on structure, energy, key, tempo and grid reliability."""

from dataclasses import dataclass, field

import numpy as np

from autodj.analysis.beats import MIN_CONFIDENCE
from autodj.camelot import camelot_distance
from autodj.models import Analysis, Transition

# Overlap lengths to consider by Camelot distance, preferred first. Clashing keys get short blends.
OVERLAPS_BY_KEY_DISTANCE = {0: (16, 32, 8), 1: (16, 32, 8), 2: (8, 16), 3: (8, 4)}
SHORTEST_OVERLAPS = (4, 8)
MAX_INCOMING_SKIP = 0.4  # never start the incoming track past 40% of its bars
SPLICE_TOLERANCE_MS = 25.0


BLENDS = ("bass_swap_blend", "filter_sweep")
CUTS = ("echo_out", "drop_cut")
OVERLAYS = ("acapella_overlay",)
VOCAL_ON = 0.5  # bar_vocals above this counts as singing/rapping
ACAPELLA_LENGTHS = (8, 16)


@dataclass
class Candidate:
    out_bar: int
    in_bar: int
    overlap: int  # 0 for cuts
    swap_bar: int | None
    kind: str = "bass_swap_blend"
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def transition(self) -> Transition:
        return Transition(type=self.kind, out_bar=self.out_bar, in_bar=self.in_bar, overlap_bars=self.overlap,
                          swap_bar=self.swap_bar)

    def describe(self) -> str:
        if self.kind in CUTS:
            return f"{self.kind}: A bar {self.out_bar} -> B bar {self.in_bar}"
        if self.kind in OVERLAYS:
            return f"{self.kind}: B's vocal from bar {self.in_bar} over A bar {self.out_bar}, {self.overlap} bars"
        return f"{self.kind}: A bar {self.out_bar} = B bar {self.in_bar}, {self.overlap} bars, swap at +{self.swap_bar}"


def tempo_gap(a: float, b: float) -> float:
    """Relative tempo difference (symmetric, log ratio), allowing half/double time (87 vs 174 is 0)."""
    return float(min(abs(np.log(b * f / a)) for f in (0.5, 1.0, 2.0)))


STABLE_WINDOW_BARS = 12
MIN_STABLE_BARS = 8  # fewer confident bars than this nearby: trust the track-wide offset instead


def stable_offsets_ms(a: Analysis) -> np.ndarray:
    """Per-bar local grid offset, rolling-median smoothed over confident bars.

    A single off window (a syncopated kick pattern) disappears; a splice or tempo change
    persists as a step. Bars with too few confident neighbours fall back to the track median.
    """
    if "stable_offsets" in a._memo:
        return a._memo["stable_offsets"]
    offs = np.array(a.bar_offset_ms)
    good = np.array(a.bar_offset_confidence) >= MIN_CONFIDENCE
    half = STABLE_WINDOW_BARS // 2
    out = np.full(len(offs), a.onset_offset_ms)
    for k in range(len(offs)):
        lo, hi = max(0, k - half), k + half
        vals = offs[lo:hi][good[lo:hi]]
        if len(vals) >= MIN_STABLE_BARS:
            out[k] = np.median(vals)
    a._memo["stable_offsets"] = out
    return out


def region_offset_ms(a: Analysis, start: int, end: int) -> tuple[float, bool]:
    """(local grid offset over bars [start, end), whether it stays constant across them)."""
    stable = stable_offsets_ms(a)[start:end]
    return float(np.median(stable)), float(np.ptp(stable)) <= SPLICE_TOLERANCE_MS


def key_distance(a: Analysis, b: Analysis) -> int:
    d = camelot_distance(a.key, b.key)
    # An uncertain key estimate shouldn't force a short blend on its own.
    if min(a.key_confidence, b.key_confidence) < 0.05:
        d = min(d, 2)
    return d


def _overlaps(a: Analysis, b: Analysis) -> tuple[int, ...]:
    return OVERLAPS_BY_KEY_DISTANCE.get(key_distance(a, b), SHORTEST_OVERLAPS)


def _swap_bar(b: Analysis, in_bar: int, overlap: int) -> int:
    """Swap basslines halfway, or on an incoming section change near the middle (on a 4-bar line)."""
    half = overlap // 2
    inner = [s - in_bar for s in b.section_starts() if 0 < s - in_bar < overlap and (s - in_bar) % 4 == 0]
    if inner:
        return min(inner, key=lambda k: abs(k - half))
    return half - half % 4 if half >= 4 else half


def score(a: Analysis, b: Analysis, c: Candidate, preferred_overlap: int) -> Candidate:
    reasons: list[str] = []
    total = 0.0

    def add(value: float, why: str) -> None:
        nonlocal total
        total += value
        if abs(value) >= 0.05:
            reasons.append(f"{value:+.2f} {why}")

    a_end = c.out_bar + c.overlap
    b_end = c.in_bar + c.overlap
    clash = key_distance(a, b) >= 3
    far_tempo = tempo_gap(a.bpm, b.bpm) > 0.06
    # A drop cut trades A's remaining track for B's drop by design, so coverage counts less.
    coverage = 1.0 if c.kind != "drop_cut" else 0.5
    add(coverage * (2.0 * a_end / a.n_bars - 2.0), f"plays {a_end / a.n_bars:.0%} of A")
    add(-coverage * 2.0 * c.in_bar / b.n_bars, f"skips {c.in_bar / b.n_bars:.0%} of B")
    later_drops = [s for s in a.sections if s.label == "drop" and s.start_bar >= a_end]
    if c.kind == "drop_cut":  # the drop at the cut point is replaced by B's, not lost
        later_drops = [s for s in later_drops if s.start_bar != c.out_bar]
    if later_drops:
        add(-0.5, "cuts A before one of its drops")
    if a_end in a.section_starts():
        add(0.5, "A leaves at a section boundary")

    if c.kind in BLENDS:
        a_labels = [a.section_at(k).label for k in range(c.out_bar, a_end)]
        b_labels = [b.section_at(k).label for k in range(c.in_bar, b_end)]
        add(-1.0 * a_labels.count("drop") / c.overlap, "A is dropping during the blend")
        add(-1.0 * b_labels.count("drop") / c.overlap, "B's drop is buried under the blend")
        ea = np.array(a.bar_energy[c.out_bar : a_end])
        eb = np.array(b.bar_energy[c.in_bar : b_end])
        mud = 2.0 * float(np.clip(ea + eb - 1.2, 0, None).mean())
        add(-mud * (0.5 if c.kind == "filter_sweep" else 1.0), "both tracks loud at once (mud)")
        if b_end in b.section_starts():
            add(0.5, "blend ends on a B section change")
            if b.section_at(b_end).label == "drop":
                add(0.3, "blend lands on B's drop")
        if a.bar_vocals is not None and b.bar_vocals is not None:
            va = np.array(a.bar_vocals[c.out_bar : a_end]) > VOCAL_ON
            vb = np.array(b.bar_vocals[c.in_bar : b_end]) > VOCAL_ON
            both = float((va & vb).mean())
            add(-1.5 * both * (0.6 if c.kind == "filter_sweep" else 1.0), "both tracks have vocals at once")
        for name, x, start, end in (("A", a, c.out_bar, a_end), ("B", b, c.in_bar, b_end)):
            if not region_offset_ms(x, start, end)[1]:
                add(-2.0, f"{name}'s grid jumps inside the blend (splice/tempo change)")
        d = key_distance(a, b)
        if d >= 2:
            clash_cost = 0.15 * (d - 1) * c.overlap / 8 * (0.5 if c.kind == "filter_sweep" else 1.0)
            add(-clash_cost, f"keys {d} steps apart for {c.overlap} bars")
        elif c.overlap == preferred_overlap:
            add(0.3, f"{c.overlap}-bar blend suits the keys")
        if far_tempo:
            add(-0.3, "big tempo stretch while both tracks play")
        if c.kind == "filter_sweep":
            add(-0.1, "filter sweep (plain blend preferred when equal)")
            if key_distance(a, b) >= 2:
                add(0.3, "filters hide the key clash")
    elif c.kind == "acapella_overlay":
        add(0.2, "B's vocal over A's instrumental")
        vb = np.array(b.bar_vocals[c.in_bar : b_end])
        add(0.5 * float((vb > VOCAL_ON).mean()) - 0.5, f"B sings in {(vb > VOCAL_ON).mean():.0%} of the overlay")
        if b_end in b.section_starts():
            add(0.4, "B's beat comes in on a section change")
            if b.section_at(b_end).label == "drop":
                add(0.4, "B's beat comes in on its drop")
        a_labels = [a.section_at(k).label for k in range(c.out_bar, a_end)]
        add(-0.5 * a_labels.count("break") / c.overlap, "A's instrumental is a breakdown (little beat to sing over)")
        if not region_offset_ms(a, c.out_bar, a_end)[1]:
            add(-2.0, "A's grid jumps under the overlay (splice/tempo change)")
    elif c.kind == "echo_out":
        add(-0.4, "echo out (a blend is smoother when it works)")
        if clash or far_tempo:
            add(0.8, "no overlap, so the key/tempo gap doesn't matter")
        if b.section_at(c.in_bar).label == "drop":
            add(0.2, "echoes out into B's drop")
    elif c.kind == "drop_cut":
        add(-0.2, "hard cut")
        add(0.6, "cuts from A's build straight into B's drop")
        if clash or far_tempo:
            add(0.4, "no overlap, so the key/tempo gap doesn't matter")

    c.score = round(total, 3)
    c.reasons = reasons
    return c


def _blend_candidates(a: Analysis, b: Analysis, min_out_bar: int) -> list[Candidate]:
    overlaps = _overlaps(a, b)
    found = []
    for overlap in overlaps:
        outs = {s.start_bar + 4 * k for s in a.sections for k in range(max(1, (s.end_bar - s.start_bar) // 4))}
        outs = {o for o in outs if o + overlap <= a.n_bars and o >= max(a.n_bars // 3, min_out_bar)}
        ins = {s.start_bar for s in b.sections if s.start_bar <= MAX_INCOMING_SKIP * b.n_bars} | {0}
        ins = {i for i in ins if i + overlap <= b.n_bars}
        for kind in BLENDS:
            if kind == "filter_sweep" and overlap not in (8, 16):
                continue
            for o in outs:
                for i in ins:
                    c = Candidate(out_bar=o, in_bar=i, overlap=overlap, swap_bar=_swap_bar(b, i, overlap), kind=kind)
                    found.append(score(a, b, c, preferred_overlap=overlaps[0]))
    return found


def _cut_candidates(a: Analysis, b: Analysis, min_out_bar: int) -> list[Candidate]:
    found = []
    lo = max(a.n_bars // 3, min_out_bar, 1)
    # echo_out: leave A on a 4-bar phrase line where it still has something to echo.
    lines = {s.start_bar + 4 * k for s in a.sections for k in range(max(1, (s.end_bar - s.start_bar) // 4))}
    outs = {o for o in lines if lo <= o < a.n_bars and a.bar_energy[o - 1] > 0.3}
    ins = {s.start_bar for s in b.sections if s.start_bar <= MAX_INCOMING_SKIP * b.n_bars} | {0}
    for o in outs:
        for i in ins:
            found.append(score(a, b, Candidate(o, i, 0, None, kind="echo_out"), 0))
    # drop_cut: where A would drop after a build-up, cut to one of B's drops instead.
    outs = {s.start_bar for s in a.sections if s.label == "drop" and lo <= s.start_bar
            and a.section_at(s.start_bar - 1).label in ("build", "break", "body")}
    ins = {s.start_bar for s in b.sections if s.label == "drop" and s.start_bar <= 0.6 * b.n_bars}
    for o in outs:
        for i in ins:
            found.append(score(a, b, Candidate(o, i, 0, None, kind="drop_cut"), 0))
    return found


def _overlay_candidates(a: Analysis, b: Analysis, min_out_bar: int) -> list[Candidate]:
    """B's acapella over A's instrumental, then B's beat comes in. Needs stems and compatible keys."""
    if a.bar_vocals is None or b.bar_vocals is None or key_distance(a, b) > 1:
        return []
    found = []
    lines = {s.start_bar + 4 * k for s in a.sections for k in range(max(1, (s.end_bar - s.start_bar) // 4))}
    for length in ACAPELLA_LENGTHS:
        outs = {o for o in lines if o + length <= a.n_bars and o >= max(a.n_bars // 3, min_out_bar)}
        # Vocal phrases of B that end where B's beat can take over (a section change).
        ins = {e - length for e in b.section_starts() if 0 <= e - length and e <= 0.7 * b.n_bars}
        ins = {i for i in ins if np.mean(np.array(b.bar_vocals[i : i + length]) > VOCAL_ON) >= 0.5}
        for o in outs:
            for i in ins:
                found.append(score(a, b, Candidate(o, i, length, None, kind="acapella_overlay"), 0))
    return found


def candidates(a: Analysis, b: Analysis, min_out_bar: int = 0, kinds: tuple[str, ...] | None = None) -> list[Candidate]:
    """All phrase-aligned candidates, best first. min_out_bar keeps A's previous blend clear."""
    found = (_blend_candidates(a, b, min_out_bar) + _cut_candidates(a, b, min_out_bar)
             + _overlay_candidates(a, b, min_out_bar))
    if kinds:
        found = [c for c in found if c.kind in kinds]
    return sorted(found, key=lambda c: c.score, reverse=True)


def best_transition(a: Analysis, b: Analysis, min_out_bar: int = 0, kinds: tuple[str, ...] | None = None) -> Candidate:
    found = candidates(a, b, min_out_bar, kinds)
    if found:
        return found[0]
    if kinds and "echo_out" not in kinds:
        raise ValueError(f"No {'/'.join(kinds)} transition fits between {a.path} and {b.path}")
    # Nothing scored (very short tracks, or a late previous blend): echo out of A's last full bar.
    out_bar = a.n_bars - 1
    if out_bar < max(min_out_bar, 1):
        raise ValueError(f"{a.path} is too short to leave room for its transitions")
    return Candidate(out_bar, 0, 0, None, kind="echo_out", score=-1.0, reasons=["fallback: nothing else fits"])
