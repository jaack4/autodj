"""Loop extend: repeat a track's last bars to make room for a full blend, like a DJ looping an
outro. A looped track is treated as a longer, virtual track: its analysis gets the repeated bars
inserted, and its audio gets them spliced in before time-stretching."""

import numpy as np

from autodj.models import Analysis, Loop, Section

SPLICE_FADE_S = 0.005
LOOP_LABEL = "loop"
MIN_TAIL_BARS = 16  # tracks with less than this after their last drop get loop options
MAX_LOOP_VOCALS = 0.25  # a looped stretch may be at most this vocal (repeated lines sound broken)


def _repeat(values: list | None, at: int, length: int, repeats: int) -> list | None:
    if values is None:
        return None
    if not values:
        return values
    block = list(values[at - length : at])
    return list(values[:at]) + block * repeats + list(values[at:])


def extend_analysis(a: Analysis, loop: Loop) -> Analysis:
    """The analysis of the track as it plays with `loop` applied (bars [at-length, at) repeated)."""
    at, length, repeats = loop.at_bar, loop.bars, loop.repeats
    db = list(a.downbeats)
    bar_len = float(np.diff(db).mean())
    t_at = db[at] if at < len(db) else db[-1] + bar_len
    span = length * bar_len
    inserted = repeats * span
    downbeats = (db[:at] + [db[at - length + k] + (r + 1) * span for r in range(repeats) for k in range(length)]
                 + [d + inserted for d in db[at:]])
    t0 = db[at - length]
    block = [b for b in a.beats if t0 <= b < t_at]
    beats = ([b for b in a.beats if b < t_at] + [b + (r + 1) * span for r in range(repeats) for b in block]
             + [b + inserted for b in a.beats if b >= t_at])

    sections: list[Section] = []
    added = repeats * length
    for s in a.sections:
        if s.end_bar <= at:
            sections.append(s)
        elif s.start_bar >= at:
            sections.append(s.model_copy(update={"start_bar": s.start_bar + added, "end_bar": s.end_bar + added,
                                                 "start": s.start + inserted, "end": s.end + inserted}))
        else:  # the loop point falls inside this section: split it around the loop
            sections.append(s.model_copy(update={"end_bar": at, "end": t_at}))
            sections.append(s.model_copy(update={"start_bar": at + added, "end_bar": s.end_bar + added,
                                                 "start": t_at + inserted, "end": s.end + inserted}))
    energy = float(np.mean(a.bar_energy[at - length : at]))
    sections.append(Section(label=LOOP_LABEL, start_bar=at, end_bar=at + added, start=t_at, end=t_at + inserted,
                            energy=round(energy, 3)))
    sections.sort(key=lambda s: s.start_bar)

    update = {"downbeats": downbeats, "beats": beats, "duration": a.duration + inserted, "sections": sections}
    for name in ("bar_energy", "bar_novelty", "bar_offset_ms", "bar_offset_confidence", "bar_vocals",
                 "bar_loudness", "bar_grit", "bar_drums"):
        update[name] = _repeat(getattr(a, name), at, length, repeats)
    out = a.model_copy(update=update)
    out._memo = {}  # model_copy shares private attributes; derived values must be recomputed
    return out


def splice_audio(audio: np.ndarray, sr: int, a: Analysis, loop: Loop) -> np.ndarray:
    """Insert the loop's repeats into (channels, samples) audio at the original sample rate/time."""
    db = list(a.downbeats)
    bar_len = float(np.diff(db).mean())
    t_at = db[loop.at_bar] if loop.at_bar < len(db) else db[-1] + bar_len
    s1 = int(round(t_at * sr))
    s0 = int(round(db[loop.at_bar - loop.bars] * sr))
    block = audio[:, s0:s1].copy()
    fade = min(int(SPLICE_FADE_S * sr), block.shape[1] // 4)
    if fade:  # soften the joins (they fall on downbeats, so a few ms is inaudible)
        block[:, :fade] *= np.linspace(0, 1, fade)
        block[:, -fade:] *= np.linspace(1, 0, fade)
    return np.concatenate([audio[:, :s1]] + [block] * loop.repeats + [audio[:, s1:]], axis=1)


def loop_options(a: Analysis) -> list[Loop]:
    """Ways to extend a track whose last drop leaves too little room for a blend after it."""
    drops = [s for s in a.sections if s.label == "drop"]
    if not drops:
        return []
    last = drops[-1]
    tail = a.n_bars - last.end_bar
    if tail >= MIN_TAIL_BARS:
        return []
    options = []
    for length in (8, 4):
        if last.end_bar - length < last.start_bar:
            continue
        if a.bar_vocals is not None:
            sung = np.mean(np.array(a.bar_vocals[last.end_bar - length : last.end_bar]) > 0.5)
            if sung > MAX_LOOP_VOCALS:
                continue
        need = MIN_TAIL_BARS - tail
        repeats = int(np.ceil(need / length))
        options.append(Loop(at_bar=last.end_bar, bars=length, repeats=repeats))
    return options
