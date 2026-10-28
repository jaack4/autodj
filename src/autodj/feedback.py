"""Remember which transitions you liked or disliked, so later plans keep the good ones.

Stored in labels/feedback.json (git-ignored). A like keeps the pair together when ordering and
strongly prefers the exact transition; a dislike penalizes the exact transition."""

import json
from dataclasses import dataclass
from pathlib import Path

from autodj.models import Loop, MixPlan, Transition

FEEDBACK_FILE = Path("labels/feedback.json")
LIKED_TRANSITION_BONUS = 1.0  # added to the matching candidate's score
DISLIKED_TRANSITION_PENALTY = 2.0
LIKED_PAIR_BONUS = 1.0  # subtracted from the pair's ordering cost
DISLIKED_PAIR_PENALTY = 0.5


@dataclass
class Verdict:
    from_track: str  # file stems
    to_track: str
    liked: bool
    type: str
    out_bar: int
    in_bar: int
    overlap: int
    loop: Loop | None
    note: str = ""

    def matches(self, kind: str, out_bar: int, in_bar: int, overlap: int, loop: Loop | None) -> bool:
        return (self.type, self.out_bar, self.in_bar, self.overlap, self.loop) == (kind, out_bar, in_bar, overlap, loop)


def load(path: Path = FEEDBACK_FILE) -> list[Verdict]:
    if not path.exists():
        return []
    out = []
    for v in json.loads(path.read_text()):
        loop = Loop(**v["loop"]) if v.get("loop") else None
        out.append(Verdict(v["from"], v["to"], v["verdict"] == "like", v["type"], v["out_bar"], v["in_bar"],
                           v["overlap"], loop, v.get("note", "")))
    return out


def record(plan: MixPlan, number: int, liked: bool, note: str = "", path: Path = FEEDBACK_FILE) -> Verdict:
    """Store a verdict on transition `number` (1-based) of a plan; replaces an earlier verdict on the same one."""
    t: Transition = plan.transitions[number - 1]
    v = Verdict(Path(plan.tracks[number - 1].path).stem, Path(plan.tracks[number].path).stem, liked, t.type,
                t.out_bar, t.in_bar, t.overlap_bars, t.loop_out, note)
    existing = [x for x in load(path) if not (x.from_track == v.from_track and x.to_track == v.to_track
                                              and x.matches(v.type, v.out_bar, v.in_bar, v.overlap, v.loop))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_to_json(x) for x in existing + [v]], indent=1))
    return v


def _to_json(v: Verdict) -> dict:
    return {"from": v.from_track, "to": v.to_track, "verdict": "like" if v.liked else "dislike", "type": v.type,
            "out_bar": v.out_bar, "in_bar": v.in_bar, "overlap": v.overlap,
            "loop": v.loop.model_dump() if v.loop else None, "note": v.note}


def for_pair(verdicts: list[Verdict], a_path: str, b_path: str) -> list[Verdict]:
    a, b = Path(a_path).stem, Path(b_path).stem
    return [v for v in verdicts if v.from_track == a and v.to_track == b]
