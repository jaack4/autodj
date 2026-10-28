"""Turn a playlist into a MixPlan: analyze tracks, order them, plan every transition."""

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from autodj.analysis import analyze_file
from autodj.camelot import camelot_distance, parse_key
from autodj.models import Analysis, MixPlan, PlanTrack
from autodj.planner import Candidate, best_transition, min_out_after, tempo_gap

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
MIN_TRACK_BARS = 16  # shorter files (jingles, snippets) are skipped
EXACT_ORDER_MAX_TRACKS = 12  # Held-Karp above this gets slow; fall back to greedy + 2-opt
ARCS = ("warmup_peak", "flat")


@dataclass
class Entry:
    track: PlanTrack
    analysis: Analysis


def load_playlist(source: Path) -> tuple[str, list[PlanTrack], dict]:
    """A playlist YAML, a folder of audio files, or a single audio file -> (name, tracks, options)."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        return source.name, [PlanTrack(path=str(p)) for p in files], {}
    if source.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(source.read_text())
        tracks = []
        for t in data.get("tracks", []):
            path = Path(t["path"]).expanduser()
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            tracks.append(PlanTrack(
                path=str(path), title=t.get("title"), artist=t.get("artist"),
                bpm_hint=t.get("bpm"), key=parse_key(str(t["key"])) if t.get("key") else None,
            ))
        options = {k: data[k] for k in ("keep_order", "energy_arc") if k in data}
        return data.get("name", source.stem), tracks, options
    raise ValueError(f"Not a playlist or folder: {source}")


def analyze_entries(tracks: list[PlanTrack], stems: bool = False, log=print) -> list[Entry]:
    """Analyze every track (cached). Unreadable files are skipped with a warning."""
    entries = []
    for k, t in enumerate(tracks):
        log(f"[{k + 1}/{len(tracks)}] analyzing {Path(t.path).name}")
        try:
            a = analyze_file(Path(t.path), bpm_hint=t.bpm_hint, stems=stems)
        except Exception as e:  # one bad file shouldn't sink the set
            log(f"  skipped: {e}")
            continue
        if a.n_bars < MIN_TRACK_BARS:
            log(f"  skipped: only {a.n_bars} bars long (need {MIN_TRACK_BARS} to mix)")
            continue
        if not a.has_steady_beat:
            log(f"  warning: no steady beat found; its tempo ({a.bpm:.1f} BPM) is a guess")
        if t.key:  # the user's key beats detection
            a = a.model_copy(update={"key": t.key, "key_confidence": 1.0})
        entries.append(Entry(track=t.model_copy(update={"path": a.path}), analysis=a))
    return entries


def pair_cost(a: Analysis, b: Analysis, transition: Candidate, arc: str) -> float:
    d = camelot_distance(a.key, b.key)
    key_cost = {0: 0.0, 1: 0.3, 2: 1.0}.get(d, 2.0 + 0.3 * (d - 3))
    key_cost *= min(1.0, 0.3 + min(a.key_confidence, b.key_confidence) / 0.15)  # trust uncertain keys less
    tempo_cost = (tempo_gap(a.bpm, b.bpm) / 0.04) ** 2  # 2% -> 0.25, 4% -> 1, 8% -> 4
    arc_cost = 0.5 if arc == "warmup_peak" and b.bpm < a.bpm - 1 else 0.0
    return key_cost + tempo_cost + arc_cost - 0.5 * transition.score


def _held_karp(cost: np.ndarray) -> list[int]:
    """Cheapest open path through every node (exact)."""
    n = len(cost)
    best = {(1 << i, i): (0.0, -1) for i in range(n)}
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = sum(1 << i for i in subset)
            for j in subset:
                prev_mask = mask & ~(1 << j)
                best[(mask, j)] = min((best[(prev_mask, i)][0] + cost[i, j], i) for i in subset if i != j)
    full = (1 << n) - 1
    j = min(range(n), key=lambda k: best[(full, k)][0])
    order, mask = [], full
    while j != -1:
        order.append(j)
        mask, j = mask & ~(1 << j), best[(mask, j)][1]
    return order[::-1]


def _path_cost(order: list[int], cost: np.ndarray) -> float:
    return float(sum(cost[i, j] for i, j in zip(order, order[1:])))


def _greedy_2opt(cost: np.ndarray) -> list[int]:
    n = len(cost)
    best: list[int] = []
    for start in range(n):
        order, left = [start], set(range(n)) - {start}
        while left:
            nxt = min(left, key=lambda j: cost[order[-1], j])
            order.append(nxt)
            left.remove(nxt)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    candidate = order[:i] + order[i : k + 1][::-1] + order[k + 1 :]
                    if _path_cost(candidate, cost) < _path_cost(order, cost) - 1e-9:
                        order, improved = candidate, True
        if not best or _path_cost(order, cost) < _path_cost(best, cost):
            best = order
    return best


def order_tracks(entries: list[Entry], arc: str) -> tuple[list[int], dict[tuple[int, int], Candidate]]:
    n = len(entries)
    transitions: dict[tuple[int, int], Candidate] = {}
    cost = np.zeros((n, n))
    for i, j in itertools.permutations(range(n), 2):
        transitions[i, j] = best_transition(entries[i].analysis, entries[j].analysis)
        cost[i, j] = pair_cost(entries[i].analysis, entries[j].analysis, transitions[i, j], arc)
    if n <= 2:
        order = min(itertools.permutations(range(n)), key=lambda o: _path_cost(list(o), cost))
        return list(order), transitions
    order = _held_karp(cost) if n <= EXACT_ORDER_MAX_TRACKS else _greedy_2opt(cost)
    return order, transitions


def build_plan(
    source: Path, keep_order: bool | None = None, arc: str | None = None, bpm: float | None = None,
    stems: bool = False, log=print,
) -> MixPlan:
    name, tracks, options = load_playlist(source)
    keep_order = options.get("keep_order", False) if keep_order is None else keep_order
    arc = arc or options.get("energy_arc", "warmup_peak")
    if arc not in ARCS:
        arc = "flat"
    entries = analyze_entries(tracks, stems=stems, log=log)
    if len(entries) < 2:
        raise ValueError("A set needs at least two readable tracks")
    log(f"planning {len(entries)} tracks")
    if keep_order:
        order = list(range(len(entries)))
        transitions = {(i, i + 1): best_transition(entries[i].analysis, entries[i + 1].analysis)
                       for i in range(len(entries) - 1)}
    else:
        order, transitions = order_tracks(entries, arc)
    plan_transitions = []
    min_out = 0
    for i, j in zip(order, order[1:]):
        c = transitions[i, j]
        if c.out_bar < min_out:  # would collide with this track's incoming blend
            c = best_transition(entries[i].analysis, entries[j].analysis, min_out_bar=min_out)
        min_out = min_out_after(c, entries[j].analysis)
        t = c.transition()
        t.notes = [f"score {c.score:+.2f}"] + c.reasons
        plan_transitions.append(t)
    return MixPlan(name=name, bpm=bpm, tracks=[entries[i].track for i in order], transitions=plan_transitions)
