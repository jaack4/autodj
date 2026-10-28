"""Choose where and how two tracks blend: phrase-aligned out/in points scored on
structure, energy, key and grid reliability."""

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


@dataclass
class Candidate:
    out_bar: int
    in_bar: int
    overlap: int
    swap_bar: int
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def transition(self) -> Transition:
        return Transition(out_bar=self.out_bar, in_bar=self.in_bar, overlap_bars=self.overlap, swap_bar=self.swap_bar)

    def describe(self) -> str:
        return f"A bar {self.out_bar} = B bar {self.in_bar}, {self.overlap} bars, swap at +{self.swap_bar}"


STABLE_WINDOW_BARS = 12
MIN_STABLE_BARS = 8  # fewer confident bars than this nearby: trust the track-wide offset instead


def stable_offsets_ms(a: Analysis) -> np.ndarray:
    """Per-bar local grid offset, rolling-median smoothed over confident bars.

    A single off window (a syncopated kick pattern) disappears; a splice or tempo change
    persists as a step. Bars with too few confident neighbours fall back to the track median.
    """
    offs = np.array(a.bar_offset_ms)
    good = np.array(a.bar_offset_confidence) >= MIN_CONFIDENCE
    half = STABLE_WINDOW_BARS // 2
    out = np.full(len(offs), a.onset_offset_ms)
    for k in range(len(offs)):
        lo, hi = max(0, k - half), k + half
        vals = offs[lo:hi][good[lo:hi]]
        if len(vals) >= MIN_STABLE_BARS:
            out[k] = np.median(vals)
    return out


def region_offset_ms(a: Analysis, start: int, end: int) -> tuple[float, bool]:
    """(local grid offset over bars [start, end), whether it stays constant across them)."""
    stable = stable_offsets_ms(a)[start:end]
    return float(np.median(stable)), float(np.ptp(stable)) <= SPLICE_TOLERANCE_MS


def _overlaps(a: Analysis, b: Analysis) -> tuple[int, ...]:
    d = camelot_distance(a.key, b.key)
    # An uncertain key estimate shouldn't force a short blend on its own.
    if min(a.key_confidence, b.key_confidence) < 0.05:
        d = min(d, 2)
    return OVERLAPS_BY_KEY_DISTANCE.get(d, SHORTEST_OVERLAPS)


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
    add(2.0 * a_end / a.n_bars - 2.0, f"plays {a_end / a.n_bars:.0%} of A")
    add(-2.0 * c.in_bar / b.n_bars, f"skips {c.in_bar / b.n_bars:.0%} of B")

    a_labels = [a.section_at(k).label for k in range(c.out_bar, a_end)]
    b_labels = [b.section_at(k).label for k in range(c.in_bar, b_end)]
    add(-1.0 * a_labels.count("drop") / c.overlap, "A is dropping during the blend")
    add(-1.0 * b_labels.count("drop") / c.overlap, "B's drop is buried under the blend")
    if any(s.label == "drop" and s.start_bar >= a_end for s in a.sections):
        add(-0.5, "cuts A before one of its drops")

    ea = np.array(a.bar_energy[c.out_bar : a_end])
    eb = np.array(b.bar_energy[c.in_bar : b_end])
    add(-2.0 * float(np.clip(ea + eb - 1.2, 0, None).mean()), "both tracks loud at once (mud)")

    if a_end in a.section_starts():
        add(0.5, "A leaves at a section boundary")
    if b_end in b.section_starts():
        add(0.5, "blend ends on a B section change")
        if b.section_at(b_end).label == "drop":
            add(0.3, "blend lands on B's drop")

    for name, x, start, end in (("A", a, c.out_bar, a_end), ("B", b, c.in_bar, b_end)):
        if not region_offset_ms(x, start, end)[1]:
            add(-2.0, f"{name}'s grid jumps inside the blend (splice/tempo change)")

    if c.overlap == preferred_overlap:
        add(0.3, f"{c.overlap}-bar blend suits the keys")

    c.score = round(total, 3)
    c.reasons = reasons
    return c


def candidates(a: Analysis, b: Analysis) -> list[Candidate]:
    """All phrase-aligned candidates, best first."""
    overlaps = _overlaps(a, b)
    out: list[Candidate] = []
    for overlap in overlaps:
        outs = {s.start_bar + 4 * k for s in a.sections for k in range(max(1, (s.end_bar - s.start_bar) // 4))}
        outs = {o for o in outs if o + overlap <= a.n_bars and o >= a.n_bars // 3}
        ins = {s.start_bar for s in b.sections if s.start_bar <= MAX_INCOMING_SKIP * b.n_bars} | {0}
        ins = {i for i in ins if i + overlap <= b.n_bars}
        for o in outs:
            for i in ins:
                c = Candidate(out_bar=o, in_bar=i, overlap=overlap, swap_bar=_swap_bar(b, i, overlap))
                out.append(score(a, b, c, preferred_overlap=overlaps[0]))
    return sorted(out, key=lambda c: c.score, reverse=True)


def best_transition(a: Analysis, b: Analysis) -> Candidate:
    found = candidates(a, b)
    if not found:
        raise ValueError(f"No transition fits between {a.path} and {b.path}")
    return found[0]
