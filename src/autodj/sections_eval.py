"""Score detected sections against hand labels from `autodj label`."""

from dataclasses import dataclass

import numpy as np

from autodj.models import Analysis


@dataclass
class SectionScore:
    name: str
    hits: int  # boundaries matched within the tolerance
    predicted: int
    truth: int
    label_accuracy: float  # share of bars whose label matches
    confusion: dict[tuple[str, str], int]  # (truth, predicted) -> bars

    @property
    def precision(self) -> float:
        return self.hits / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_boundaries(predicted: list[float], truth: list[float], tolerance: float) -> int:
    """One-to-one matches within tolerance, closest pairs first."""
    pairs = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth) if abs(p - t) <= tolerance)
    used_p, used_t, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            hits += 1
    return hits


def _label_at(sections: list[tuple[float, str]], t: float) -> str:
    label = sections[0][1]
    for start, name in sections:
        if start <= t:
            label = name
    return label


def score_track(a: Analysis, truth: list[dict], name: str = "", tolerance_bars: float = 1.0) -> SectionScore:
    """Boundaries: every section start except the track's first. Labels: compared at each bar's midpoint."""
    bar = float(np.diff(a.downbeats).mean())
    first = a.downbeats[0] + bar / 2
    pred_starts = [s.start for s in a.sections if s.start > first]
    true_starts = [s["start"] for s in truth if s["start"] > first]
    hits = match_boundaries(pred_starts, true_starts, tolerance_bars * bar)

    pred = [(s.start, s.label) for s in a.sections]
    true = sorted((s["start"], s["label"]) for s in truth)
    confusion: dict[tuple[str, str], int] = {}
    agree = 0
    for d in a.downbeats:
        t = d + bar / 2
        key = (_label_at(true, t), _label_at(pred, t))
        confusion[key] = confusion.get(key, 0) + 1
        agree += key[0] == key[1]
    return SectionScore(name, hits, len(pred_starts), len(true_starts), agree / len(a.downbeats), confusion)


def summarize(scores: list[SectionScore]) -> SectionScore:
    """Micro-averaged totals across tracks."""
    confusion: dict[tuple[str, str], int] = {}
    for s in scores:
        for k, v in s.confusion.items():
            confusion[k] = confusion.get(k, 0) + v
    bars = sum(confusion.values())
    agree = sum(v for (t, p), v in confusion.items() if t == p)
    return SectionScore("all", sum(s.hits for s in scores), sum(s.predicted for s in scores),
                        sum(s.truth for s in scores), agree / bars if bars else 0.0, confusion)
