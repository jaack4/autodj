"""Choose where and how two tracks meet: phrase-aligned out/in points and a transition
type, scored on structure, energy, key, tempo and grid reliability."""

from dataclasses import dataclass, field

import numpy as np

from autodj.analysis.beats import MIN_CONFIDENCE
from autodj.camelot import camelot_distance
from autodj.models import Analysis, Loop, Transition

# Overlap lengths to consider by Camelot distance, preferred first. Clashing keys get short blends.
OVERLAPS_BY_KEY_DISTANCE = {0: (16, 32, 8), 1: (16, 32, 8), 2: (8, 16), 3: (8, 4)}
SHORTEST_OVERLAPS = (4, 8)
MAX_INCOMING_SKIP = 0.4  # never start the incoming track past 40% of its bars
SPLICE_TOLERANCE_MS = 25.0
UNCERTAIN_KEY = 0.03  # key confidence below this: don't trust the key to rule anything out


BLENDS = ("bass_swap_blend", "filter_sweep", "stem_blend", "melody_first", "reverb_wash")
NEEDS_STEMS = ("stem_blend", "melody_first", "double_drop", "acapella_overlay")
STYLES = {  # transition types each style may use
    "smooth": BLENDS,  # pure mixing: overlaps, nothing abrupt
    "mixed": None,  # everything, including cuts, echo outs and acapella overlays
}
FULL_PLAY = 0.85  # playing this share of a track counts as playing all of it (outros fade anyway)
CUTS = ("echo_out", "drop_cut", "beat_roll", "tape_stop")
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
    loop: Loop | None = None  # A is loop-extended first (out_bar counts the looped bars)

    def transition(self) -> Transition:
        return Transition(type=self.kind, out_bar=self.out_bar, in_bar=self.in_bar, overlap_bars=self.overlap,
                          swap_bar=self.swap_bar, loop_out=self.loop)

    def describe(self) -> str:
        if self.kind in CUTS:
            return f"{self.kind}: A bar {self.out_bar} -> B bar {self.in_bar}"
        if self.kind == "double_drop":
            return f"double_drop: A's drop at bar {self.out_bar} with B's drop at bar {self.in_bar}, {self.overlap} bars"
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
    if min(a.key_confidence, b.key_confidence) < UNCERTAIN_KEY:
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


def vocal_carry_end_bar(a: Analysis, swap: int, a_end: int) -> int:
    """Where A's vocals stop being carried in a stem blend: they continue a phrase that's already going
    at the swap, but never run into a new section of A (e.g. a slow outro vocal over B's beat)."""
    if a.bar_vocals is None or swap >= a.n_bars or max(a.bar_vocals[max(0, swap - 1) : swap + 1]) <= VOCAL_ON:
        return swap
    return min(a_end, a.section_at(swap).end_bar)


def _share(values: list[float] | None, start: int, end: int, on: float = VOCAL_ON) -> float:
    """Share of bars in [start, end) above `on` (0 if there's no data)."""
    if values is None or end <= start:
        return 0.0
    return float(np.mean(np.array(values[start:end]) > on))


def character(x: Analysis, start: int, end: int) -> dict | None:
    """Median loudness (dB vs the track's LUFS), grit and drums level over bars [start, end)."""
    lo, hi = max(0, start), min(x.n_bars, end)
    if not x.bar_loudness or hi <= lo:
        return None
    c = {"loud": float(np.median(x.bar_loudness[lo:hi])), "grit": float(np.median(x.bar_grit[lo:hi]))}
    if x.bar_drums:
        c["drums"] = float(np.median(x.bar_drums[lo:hi]))
    return c


def character_jump(ca: dict | None, cb: dict | None) -> float:
    """How much harsher/punchier B is than A (negative: calmer). ~1 is clearly noticeable."""
    if ca is None or cb is None:
        return 0.0
    jump = (cb["grit"] - ca["grit"]) / 0.1 + (cb["loud"] - ca["loud"]) / 6
    if "drums" in ca and "drums" in cb:
        jump += (cb["drums"] - ca["drums"]) / 15
    return jump


def drop_bars(x: Analysis) -> list[tuple[int, int]]:
    return [(s.start_bar, s.end_bar) for s in x.sections if s.label == "drop"]


def drop_character(x: Analysis) -> dict | None:
    """Character of the track's drops together (its 'vibe' at full power); whole track if it has none."""
    bars = [k for a, b in drop_bars(x) for k in range(a, b)] or list(range(x.n_bars))
    if not x.bar_loudness:
        return None
    c = {"loud": float(np.median([x.bar_loudness[k] for k in bars])),
         "grit": float(np.median([x.bar_grit[k] for k in bars]))}
    if x.bar_drums:
        c["drums"] = float(np.median([x.bar_drums[k] for k in bars]))
    return c


def biggest_drop(x: Analysis) -> tuple[int, int] | None:
    """The drop with the most power; short bursts (< 8 bars) only count if there's nothing longer."""
    drops = drop_bars(x)
    drops = [d for d in drops if d[1] - d[0] >= 8] or drops
    best, best_power = None, -np.inf
    for a, b in drops:
        c = character(x, a, b)
        power = (c["loud"] + 10 * c["grit"]) if c else float(np.mean(x.bar_energy[a:b]))
        if power > best_power:
            best, best_power = (a, b), power
    return best


def _mean_energy(x: Analysis, start: int, end: int) -> float:
    start, end = max(0, start), min(x.n_bars, end)
    return float(np.mean(x.bar_energy[start:end])) if end > start else 0.0


def score(a: Analysis, b: Analysis, c: Candidate, preferred_overlap: int, style: str = "mixed") -> Candidate:
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
    coverage = 0.5 if c.kind in ("drop_cut", "beat_roll", "double_drop") else 1.0
    shortfall = max(0.0, FULL_PLAY - a_end / a.n_bars)
    add(-coverage * 2.5 * shortfall, f"plays only {a_end / a.n_bars:.0%} of A")
    add(-coverage * 2.0 * c.in_bar / b.n_bars, f"skips {c.in_bar / b.n_bars:.0%} of B")
    later_drops = [s for s in a.sections if s.label == "drop" and s.start_bar >= a_end]
    if c.kind in ("drop_cut", "beat_roll", "double_drop"):  # the drop at the cut point is replaced by B's
        later_drops = [s for s in later_drops if s.start_bar != c.out_bar]
    if later_drops:
        add(-0.5, "cuts A before one of its drops")
    if a_end in a.section_starts():
        add(0.5, "A leaves at a section boundary")

    # Energy match: what B brings in should sit near what A is doing at that moment.
    if c.overlap:
        ea, eb = _mean_energy(a, c.out_bar, a_end), _mean_energy(b, c.in_bar, b_end)
    else:
        ea, eb = _mean_energy(a, c.out_bar - 4, c.out_bar), _mean_energy(b, c.in_bar, c.in_bar + 4)
    jump = eb - ea
    if jump > 0.15 and c.kind not in ("drop_cut", "beat_roll", "double_drop"):  # these jumps are the point
        add(-2.0 * (jump - 0.15), f"B comes in much more intense than A ({eb:.2f} vs {ea:.2f})")
    elif jump < -0.3:
        add(-1.0 * (-jump - 0.3), f"B comes in much quieter than A ({eb:.2f} vs {ea:.2f})")
    if c.overlap and ea < 0.25:
        add(-0.5, "A has already faded out")

    # Character match (comparable across tracks): what B brings in vs what A is doing there.
    if c.kind not in ("drop_cut", "beat_roll", "double_drop"):
        if c.overlap:
            ca, cb = character(a, c.out_bar, a_end), character(b, c.in_bar, b_end)
        else:
            ca, cb = character(a, c.out_bar - 4, c.out_bar), character(b, c.in_bar, c.in_bar + 4)
        jump = character_jump(ca, cb)
        if jump > 1.0:
            add(-0.5 * min(jump - 1.0, 4.0), f"B is harsher/punchier than A's section here ({jump:+.1f})")
        elif jump < -1.5:
            add(-0.3 * min(-jump - 1.5, 4.0), f"B is much softer than A's section here ({jump:+.1f})")

    # Respect A's arc: don't leave in the middle of a build, don't cut into its biggest drop.
    if c.kind not in ("drop_cut", "beat_roll", "double_drop"):
        if any(a.section_at(k).label == "build" for k in range(max(0, a_end - 4), a_end)):
            add(-0.8, "A leaves mid-build (its tension never pays off)")
        elif any(s.label == "drop" and a_end - 2 <= s.start_bar <= a_end + 4 for s in a.sections):
            add(-0.8, "A leaves just as its own drop arrives (the build-up never pays off)")
    peak = biggest_drop(a)
    if peak and c.out_bar < (peak[0] + peak[1]) // 2 and a_end > peak[0] and c.kind != "drop_cut":
        add(-1.0, "interrupts A's biggest drop")  # blending out of its second half is fine

    # Letdown: the drop B arrives at shouldn't be a big step down from A's drops.
    landing = next((s for s in b.sections if s.label == "drop" and s.start_bar >= b_end - 1), None)
    if landing and drop_bars(a):
        let = character_jump(drop_character(a), character(b, landing.start_bar, landing.end_bar))
        if let < -1.0:
            add(-0.4 * min(-let - 1.0, 4.0), f"B's drop is a step down from A's drops ({let:+.1f})")

    if c.kind in BLENDS:
        # A's drop bars under the blend, not counting each drop's last 4 (mixing out of a drop that
        # has played out is normal).
        a_dropping = sum(1 for k in range(c.out_bar, a_end) if a.section_at(k).label == "drop"
                         and k < (a.section_at(k).start_bar + a.section_at(k).end_bar) // 2)
        b_labels = [b.section_at(k).label for k in range(c.in_bar, b_end)]
        add(-1.0 * a_dropping / c.overlap, "A is dropping during the blend")
        add(-1.0 * b_labels.count("drop") / c.overlap, "B's drop is buried under the blend")
        ea = np.array(a.bar_energy[c.out_bar : a_end])
        eb = np.array(b.bar_energy[c.in_bar : b_end])
        mud = 2.0 * float(np.clip(ea + eb - 1.2, 0, None).mean())
        add(-mud * (1.0 if c.kind == "bass_swap_blend" else 0.5), "both tracks loud at once (mud)")
        if c.kind == "melody_first":
            add(0.15, "melody first: B's melody and vocals arrive over A's groove")
            va = _share(a.bar_vocals, c.out_bar, a_end)
            vb = _share(b.bar_vocals, c.in_bar, b_end)
            if vb >= 0.5 and va <= 0.25:
                add(0.3, "B's hook teased over A's beat")
            if _mean_energy(a, c.out_bar, a_end) < 0.45:
                add(-0.4, "A's groove is too thin to carry B's melody")
        elif c.kind == "reverb_wash":
            add(-0.05, "reverb wash: A dissolves into reverb as B rises (an effect: needs a reason)")
            if _share(a.bar_vocals, c.out_bar, a_end) >= 0.5:
                add(0.15, "A's vocals wash out nicely")
        if b_end in b.section_starts():
            add(0.5, "blend ends on a B section change")
            if b.section_at(b_end).label == "drop":
                add(0.3, "blend lands on B's drop")
        if c.kind == "stem_blend":
            add(0.2, "stem blend: drums, bass and melody swap separately")
            swap = c.out_bar + (c.swap_bar or c.overlap // 2)
            carry_end = vocal_carry_end_bar(a, swap, a_end)
            carried = (_share(a.bar_vocals, swap, carry_end) * (carry_end - swap) / (a_end - swap)
                       if a_end > swap else 0.0)
            if carried >= 0.5:
                add(0.4, f"A's vocals carry across B's beat ({carried:.0%} of the second half)")
        elif c.kind == "melody_first" or c.kind == "reverb_wash":
            pass  # A's vocals duck early / dissolve, so no clash
        elif a.bar_vocals is not None and b.bar_vocals is not None:  # stem blends hold B's vocals back
            va = np.array(a.bar_vocals[c.out_bar : a_end]) > VOCAL_ON
            vb = np.array(b.bar_vocals[c.in_bar : b_end]) > VOCAL_ON
            both = float((va & vb).mean())
            add(-1.5 * both * (0.6 if c.kind == "filter_sweep" else 1.0), "both tracks have vocals at once")
        for name, x, start, end in (("A", a, c.out_bar, a_end), ("B", b, c.in_bar, b_end)):
            if not region_offset_ms(x, start, end)[1]:
                add(-2.0, f"{name}'s grid jumps inside the blend (splice/tempo change)")
        d = key_distance(a, b)
        if d >= 2:
            soften = {"filter_sweep": 0.5, "reverb_wash": 0.3}.get(c.kind, 1.0)
            clash_cost = 0.15 * (d - 1) * c.overlap / 8 * soften
            add(-clash_cost, f"keys {d} steps apart for {c.overlap} bars")
        elif c.overlap == preferred_overlap:
            add(0.3, f"{c.overlap}-bar blend suits the keys")
        if style == "smooth":
            add(0.15 * c.overlap / 8, f"{c.overlap}-bar blend (smooth style likes long blends)")
        if far_tempo:
            add(-0.3, "big tempo stretch while both tracks play")
        if c.kind == "filter_sweep":
            add(-0.1, "filter sweep (plain blend preferred when equal)")
            if key_distance(a, b) >= 2:
                add(0.3, "filters hide the key clash")
    elif c.kind == "acapella_overlay":
        add(-0.2, "acapella overlay (a special move: best used sparingly)")
        vb = np.array(b.bar_vocals[c.in_bar : b_end])
        add(0.5 * float((vb > VOCAL_ON).mean()) - 0.5, f"B sings in {(vb > VOCAL_ON).mean():.0%} of the overlay")
        if b_end in b.section_starts():
            add(0.3, "B's beat comes in on a section change")
            if b.section_at(b_end).label == "drop":
                add(0.3, "B's beat comes in on its drop")
        a_labels = [a.section_at(k).label for k in range(c.out_bar, a_end)]
        add(-0.5 * a_labels.count("break") / c.overlap, "A's instrumental is a breakdown (little beat to sing over)")
        if not region_offset_ms(a, c.out_bar, a_end)[1]:
            add(-2.0, "A's grid jumps under the overlay (splice/tempo change)")
    elif c.kind == "double_drop":
        add(0.6, "double drop: both drops hit together")
        if not any(s.label == "drop" and s.start_bar > c.out_bar for s in a.sections):
            add(0.3, "it's A's final drop, so nothing of A is lost")
        if tempo_gap(a.bpm, b.bpm) > 0.04:
            add(-0.5, "drops at different tempos stretch badly together")
    elif c.kind == "beat_roll":
        add(-0.1, "beat roll: A stutters into the cut")
        add(0.6, "rolls A's build straight into B's drop")
        if clash or far_tempo:
            add(0.3, "no overlap, so the key/tempo gap doesn't matter")
    elif c.kind == "tape_stop":
        add(-0.3, "tape stop: a dramatic full stop (use rarely)")
        if b.section_at(c.in_bar).label == "drop":
            add(0.3, "restarts straight into B's drop")
        if clash or far_tempo:
            add(0.5, "no overlap, so the key/tempo gap doesn't matter")
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


def _blend_candidates(a: Analysis, b: Analysis, min_out_bar: int, style: str = "mixed") -> list[Candidate]:
    overlaps = _overlaps(a, b)
    found = []
    for overlap in overlaps:
        outs = {s.start_bar + 4 * k for s in a.sections for k in range(max(1, (s.end_bar - s.start_bar) // 4))}
        outs = {o for o in outs if o + overlap <= a.n_bars and o >= max(a.n_bars // 3, min_out_bar)}
        ins = {s.start_bar for s in b.sections if s.start_bar <= MAX_INCOMING_SKIP * b.n_bars} | {0}
        ins = {i for i in ins if i + overlap <= b.n_bars}
        for kind in BLENDS:
            if kind == "filter_sweep" and overlap not in (8, 16, 32):
                continue
            if kind in NEEDS_STEMS and (a.bar_vocals is None or b.bar_vocals is None
                                        or key_distance(a, b) > 1 or overlap < 8):
                continue  # needs stems, and A's parts must sit in key over B's
            if kind == "reverb_wash" and overlap > 8:
                continue  # a wash is an accent, not a long transition
            for o in outs:
                for i in ins:
                    swap = _swap_bar(b, i, overlap)
                    if kind == "melody_first":  # the rhythm swaps late: B's top rides A's groove first
                        swap = max(4, (overlap * 3 // 4) - (overlap * 3 // 4) % 4)
                    c = Candidate(out_bar=o, in_bar=i, overlap=overlap, swap_bar=swap, kind=kind)
                    found.append(score(a, b, c, preferred_overlap=overlaps[0], style=style))
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
            found.append(score(a, b, Candidate(o, i, 0, None, kind="beat_roll"), 0))
    # tape_stop: stop A dead at the end of one of its drops, then B starts.
    outs = {s.end_bar for s in a.sections if s.label == "drop" and lo <= s.end_bar < a.n_bars}
    ins = {s.start_bar for s in b.sections if s.start_bar <= MAX_INCOMING_SKIP * b.n_bars} | {0}
    for o in outs:
        for i in ins:
            found.append(score(a, b, Candidate(o, i, 0, None, kind="tape_stop"), 0))
    return found


def _double_drop_candidates(a: Analysis, b: Analysis, min_out_bar: int) -> list[Candidate]:
    """A's drop and B's drop start on the same downbeat and play together for a few bars."""
    if a.bar_vocals is None or b.bar_vocals is None or key_distance(a, b) > 1:
        return []
    found = []
    lo = max(a.n_bars // 3, min_out_bar)
    a_drops = [s for s in a.sections if s.label == "drop" and s.start_bar >= lo]
    b_drops = [s for s in b.sections if s.label == "drop" and s.start_bar <= 0.6 * b.n_bars]
    for sa in a_drops:
        for sb in b_drops:
            for length in (16, 8):
                if sa.start_bar + length <= sa.end_bar and sb.start_bar + length <= sb.end_bar:
                    found.append(score(a, b, Candidate(sa.start_bar, sb.start_bar, length, None, kind="double_drop"), 0))
                    break
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


def candidates(a: Analysis, b: Analysis, min_out_bar: int = 0, kinds: tuple[str, ...] | None = None,
               style: str = "mixed", loops: bool = True) -> list[Candidate]:
    """All phrase-aligned candidates, best first. min_out_bar keeps A's previous blend clear.

    style 'smooth' only offers real blends (kinds, if given, narrows it further).
    """
    found = _blend_candidates(a, b, min_out_bar, style)
    if style != "smooth":
        found += (_double_drop_candidates(a, b, min_out_bar) + _cut_candidates(a, b, min_out_bar)
                  + _overlay_candidates(a, b, min_out_bar))
    allowed = STYLES.get(style)
    if allowed:
        found = [c for c in found if c.kind in allowed]
    if kinds:
        found = [c for c in found if c.kind in kinds]
    if loops:
        found += _looped_candidates(a, b, min_out_bar, kinds, style)
    return sorted(found, key=lambda c: c.score, reverse=True)


LOOP_COST = 0.2  # per repeat


def _looped_candidates(a: Analysis, b: Analysis, min_out_bar: int, kinds, style: str) -> list[Candidate]:
    """Candidates that first loop A's last bars, so a full blend fits after its last drop."""
    from autodj.loops import extend_analysis, loop_options

    found = []
    for loop in loop_options(a):
        va = extend_analysis(a, loop)
        for c in candidates(va, b, min_out_bar, kinds, style, loops=False):
            if c.out_bar + c.overlap <= loop.at_bar:
                continue  # doesn't reach the loop: same as an unlooped candidate
            c.loop = loop
            c.score = round(c.score - LOOP_COST * loop.repeats, 3)
            c.reasons = c.reasons + [f"{-LOOP_COST * loop.repeats:+.2f} loops A's last {loop.bars} bars "
                                     f"x{loop.repeats} to make room for the blend"]
            found.append(c)
    return found


MIN_SOLO_BARS = 8  # a track plays alone at least this long between its two transitions (if long enough)


def min_out_after(c: Candidate, b: Analysis) -> int:
    """Earliest bar where track B may start its own outgoing transition after entering via c."""
    blend_end = c.in_bar + c.overlap
    return blend_end + min(MIN_SOLO_BARS, max(0, b.n_bars - blend_end - 4))


def best_transition(a: Analysis, b: Analysis, min_out_bar: int = 0, kinds: tuple[str, ...] | None = None,
                    style: str = "mixed") -> Candidate:
    found = candidates(a, b, min_out_bar, kinds, style)
    if found:
        return found[0]
    if kinds and "echo_out" not in kinds:
        raise ValueError(f"No {'/'.join(kinds)} transition fits between {a.path} and {b.path}")
    # Nothing scored (very short tracks, or a late previous blend): echo out of A's last full bar.
    out_bar = a.n_bars - 1
    if out_bar < max(min_out_bar, 1):
        raise ValueError(f"{a.path} is too short to leave room for its transitions")
    return Candidate(out_bar, 0, 0, None, kind="echo_out", score=-1.0, reasons=["fallback: nothing else fits"])
