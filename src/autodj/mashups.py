"""Mid-song mashups: while one track plays an instrumental stretch, lay a sung phrase from
another track in the set over it (tempo-matched and key-shifted)."""

from dataclasses import dataclass

import numpy as np

from autodj.camelot import camelot_distance, shift_key
from autodj.models import Analysis, Mashup, MixPlan
from autodj.planner import UNCERTAIN_KEY, VOCAL_ON, tempo_gap

LENGTHS = (16, 8)
MAX_TEMPO_GAP = 0.08
CLEARANCE_BARS = 4  # keep this far from the host's own transitions
MIN_SCORE = 1.2  # only mashups that clearly fit
HOST_VOCALS_MAX = 0.25  # share of bars with vocals that still counts as an instrumental stretch
GUEST_VOCALS_MIN = 0.6


@dataclass
class _Option:
    mashup: Mashup
    score: float


def _vocal_share(a: Analysis, start: int, end: int) -> float:
    return float(np.mean(np.array(a.bar_vocals[start:end]) > VOCAL_ON)) if end > start else 0.0


def _solo_window(plan: MixPlan, i: int, a: Analysis) -> tuple[int, int]:
    """Bars of track i that play with no transition going on (with some clearance)."""
    start = 0
    if i > 0:
        t = plan.transitions[i - 1]
        start = t.in_bar + t.overlap_bars + CLEARANCE_BARS
    end = a.n_bars
    if i < len(plan.transitions):
        end = plan.transitions[i].out_bar - CLEARANCE_BARS
    return start, end


def _key(plan: MixPlan, i: int, a: Analysis) -> str:
    t = plan.tracks[i]
    return shift_key(t.key or a.key, t.semitones)


def _tempo(plan: MixPlan, i: int, a: Analysis) -> float:
    return plan.bpm or plan.tracks[i].bpm or a.bpm


def find_mashups(plan: MixPlan, analyses: list[Analysis], max_count: int = 2) -> list[Mashup]:
    """Best non-overlapping mashups: at most one per host, each guest used once."""
    if max_count <= 0 or any(a.bar_vocals is None for a in analyses):
        return []
    options: list[_Option] = []
    for i, host in enumerate(analyses):
        lo, hi = _solo_window(plan, i, host)
        host_key, host_bpm = _key(plan, i, host), _tempo(plan, i, host)
        for j, guest in enumerate(analyses):
            if j == i or min(host.key_confidence, guest.key_confidence) < UNCERTAIN_KEY:
                continue
            gap = tempo_gap(host_bpm, guest.bpm)
            if gap > MAX_TEMPO_GAP:
                continue
            shifts = [s for s in sorted(range(-2, 3), key=abs)
                      if camelot_distance(host_key, shift_key(guest.key, s)) <= 1]
            if not shifts:
                continue
            shift = shifts[0]
            for length in LENGTHS:
                hosts = [s.start_bar for s in host.sections
                         if lo <= s.start_bar and s.start_bar + length <= hi and s.label in ("drop", "body", "build")]
                guests = [s.start_bar for s in guest.sections if s.start_bar + length <= guest.n_bars]
                for hb in hosts:
                    if _vocal_share(host, hb, hb + length) > HOST_VOCALS_MAX:
                        continue
                    energy = float(np.mean(host.bar_energy[hb : hb + length]))
                    if energy < 0.5:
                        continue
                    for gb in guests:
                        sung = _vocal_share(guest, gb, gb + length)
                        if sung < GUEST_VOCALS_MIN:
                            continue
                        notes = []
                        score = 1.0 + 0.5 * sung
                        notes.append(f"{guest_name(plan, j)} sings in {sung:.0%} of the phrase")
                        if host.section_at(hb).label == "drop":
                            score += 0.3
                            notes.append("over the host's drop")
                        if abs(j - i) == 1:
                            score -= 0.5
                            notes.append("guest is a neighbour in the set (less of a surprise)")
                        elif j > i:
                            score += 0.2
                            notes.append("a teaser of a track still to come")
                        else:
                            notes.append("a callback to an earlier track")
                        score -= 0.3 * abs(shift) + 0.3 * (gap / 0.04) ** 2
                        if length == 16:
                            score += 0.1
                        options.append(_Option(Mashup(host=i, host_bar=hb, guest=j, guest_bar=gb, bars=length,
                                                      semitones=shift, notes=notes), round(score, 3)))
    chosen: list[Mashup] = []
    for opt in sorted(options, key=lambda o: o.score, reverse=True):
        m = opt.mashup
        if opt.score < MIN_SCORE or len(chosen) >= max_count:
            break
        if any(c.host == m.host or c.guest == m.guest for c in chosen):
            continue
        m.notes = [f"score {opt.score:+.2f}"] + m.notes
        chosen.append(m)
    return sorted(chosen, key=lambda m: m.host)


def guest_name(plan: MixPlan, j: int) -> str:
    from pathlib import Path

    t = plan.tracks[j]
    return " - ".join(x for x in (t.artist, t.title) if x) or Path(t.path).stem
