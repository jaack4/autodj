"""The creative director: Claude picks among the planner's validated transition options
and writes the set's story. It never touches audio or timing, only chooses."""

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

from autodj.camelot import shift_key
from autodj.models import Analysis, MixPlan
from autodj.planner import Candidate, best_transition, candidates, min_out_after

MODEL = "claude-opus-5-5"
OPTIONS_PER_TRANSITION = 6
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0)}  # (input, output) USD, for the cost estimate

SYSTEM = """You are the creative director for an automatic DJ mix. The tracks and their order are fixed.

For each transition, a planner has already generated options that are technically sound (beat-matched,
phrase-aligned, tempo handled) and scored them with simple rules. Your job is the musical judgement the
rules lack: choose one option per transition so the whole set flows and builds, and give the reason in
one sentence a DJ would agree with.

Things worth weighing: the energy arc across the whole set (where the peaks land, saving the biggest
moments), variety in transition types (a set of identical blends is dull, but don't force novelty),
vocals clashing, key compatibility, and how much of each track the crowd actually gets to hear. Planner
scores are a useful prior, not a rule.

One hard constraint: on a track that is both entered and left, the option you pick for leaving it must
have out_bar at least 8 bars after the incoming option's busy_until_bar. A pick that breaks this gets
replaced automatically, so avoid it.

Then give the set a short title and a 2-4 sentence narrative of its arc."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "narrative": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "transition": {"type": "integer"},
                    "option_id": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["transition", "option_id", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "narrative", "choices"],
    "additionalProperties": False,
}


class Choice(BaseModel):
    transition: int
    option_id: str
    why: str


class Direction(BaseModel):
    title: str
    narrative: str
    choices: list[Choice]


class DirectorError(RuntimeError):
    pass


@dataclass
class Options:
    """The menu for one transition, best planner score first."""

    index: int  # 1-based transition number
    a: Analysis
    b: Analysis
    items: dict[str, Candidate]


def build_options(analyses: list[Analysis], plan: MixPlan, per_transition: int = OPTIONS_PER_TRANSITION) -> list[Options]:
    """The planner's current pick, the best of each transition type, then the next best overall."""
    menus = []
    for k, (a, b, current) in enumerate(zip(analyses, analyses[1:], plan.transitions)):
        ranked = candidates(a, b, style=plan.style)
        picked: list[Candidate] = []

        def take(c: Candidate) -> None:
            if len(picked) < per_transition and not any(
                (p.kind, p.out_bar, p.in_bar, p.overlap) == (c.kind, c.out_bar, c.in_bar, c.overlap) for p in picked
            ):
                picked.append(c)

        for c in ranked:  # the plan's current choice, if the planner still offers it
            if (c.kind, c.out_bar, c.in_bar, c.overlap) == (current.type, current.out_bar, current.in_bar,
                                                            current.overlap_bars):
                take(c)
                break
        for kind in dict.fromkeys(c.kind for c in ranked):
            take(next(c for c in ranked if c.kind == kind))
        for c in ranked:
            take(c)
        picked.sort(key=lambda c: c.score, reverse=True)
        menus.append(Options(k + 1, a, b, {f"T{k + 1}{'abcdefghij'[i]}": c for i, c in enumerate(picked)}))
    return menus


def _section_summary(a: Analysis) -> str:
    parts = []
    for s in a.sections:
        vocals = ""
        if a.bar_vocals is not None:
            share = sum(v > 0.5 for v in a.bar_vocals[s.start_bar : s.end_bar]) / (s.end_bar - s.start_bar)
            vocals = f", vocals {share:.0%}"
        repeat = f", repeat group {s.group}" if s.group else ""
        parts.append(f"{s.label} {s.start_bar}-{s.end_bar} (energy {s.energy:.2f}{vocals}{repeat})")
    return " | ".join(parts)


def _where(a: Analysis, bar: int) -> str:
    if bar >= a.n_bars:
        return "end of track"
    return f"{a.section_at(bar).label} (energy {a.section_at(bar).energy:.2f})"


def build_prompt(plan: MixPlan, analyses: list[Analysis], menus: list[Options]) -> str:
    tracks = [{
        "position": i + 1,
        "name": " - ".join(x for x in (t.artist, t.title) if x) or Path(a.path).stem,
        "bpm": round(a.bpm, 1), "key": shift_key(t.key or a.key, t.semitones),
        "pitch_shift_semitones": t.semitones, "bars": a.n_bars, "seconds": round(a.duration),
        "sections": _section_summary(a),
    } for i, (t, a) in enumerate(zip(plan.tracks, analyses))]
    transitions = []
    for m in menus:
        options = []
        for oid, c in m.items.items():
            options.append({
                "id": oid, "type": c.kind, "out_bar": c.out_bar, "in_bar": c.in_bar, "overlap_bars": c.overlap,
                "a_leaves_from": _where(m.a, c.out_bar), "b_enters_at": _where(m.b, c.in_bar),
                "b_after_transition": _where(m.b, c.in_bar + c.overlap),
                "busy_until_bar": c.in_bar + c.overlap,
                "planner_score": c.score, "planner_notes": c.reasons,
            })
        transitions.append({"transition": m.index, "from_track": m.index, "to_track": m.index + 1, "options": options})
    return json.dumps({"set_name": plan.name, "tracks": tracks, "transitions": transitions}, indent=1)


def ask_claude(prompt: str, model: str = MODEL, client: anthropic.Anthropic | None = None) -> tuple[Direction, dict]:
    """One structured-output request. Returns the direction and usage info."""
    client = client or anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a safety classifier declines, Anthropic re-runs on its recommended model
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise DirectorError("No valid Anthropic credentials: set ANTHROPIC_API_KEY") from e
    except anthropic.RateLimitError as e:
        raise DirectorError("Rate limited by the Anthropic API; try again in a minute") from e
    except anthropic.APIStatusError as e:
        raise DirectorError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise DirectorError("Couldn't reach the Anthropic API (network)") from e

    if response.stop_reason == "refusal":
        raise DirectorError("The model declined this request")
    if response.stop_reason == "max_tokens":
        raise DirectorError("The response was cut off (max_tokens)")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise DirectorError("The response had no text")
    try:
        direction = Direction.model_validate_json(text)
    except ValidationError as e:
        raise DirectorError(f"The response didn't match the expected format: {e}") from e

    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.model in PRICE_PER_MTOK:
        price_in, price_out = PRICE_PER_MTOK[response.model]
        usage["cost_usd"] = round(usage["input_tokens"] * price_in / 1e6 + usage["output_tokens"] * price_out / 1e6, 4)
    return direction, usage


def apply_direction(plan: MixPlan, menus: list[Options], direction: Direction, usage: dict) -> tuple[MixPlan, list[str]]:
    """Swap in the director's picks, keeping the plan valid. Returns the new plan and any repairs made."""
    picks = {c.transition: c for c in direction.choices}
    repairs: list[str] = []
    transitions = []
    min_out = 0
    for m, current in zip(menus, plan.transitions):
        choice = picks.get(m.index)
        if choice is None or choice.option_id not in m.items:
            label = "no pick" if choice is None else f"unknown option {choice.option_id!r}"
            repairs.append(f"transition {m.index}: {label}; kept the planner's choice")
            c, why = next(iter(m.items.values())), "planner's choice (director gave none)"
        else:
            c, why = m.items[choice.option_id], choice.why
        if c.out_bar < min_out:  # would start before this track's incoming transition has finished
            valid = [x for x in m.items.values() if x.out_bar >= min_out]
            replacement = valid[0] if valid else best_transition(m.a, m.b, min_out_bar=min_out)
            repairs.append(f"transition {m.index}: {c.describe()} overlaps the previous transition; "
                           f"used {replacement.describe()}")
            c, why = replacement, f"(replaced: the director's pick overlapped the previous transition) {why}"
        t = c.transition()
        t.notes = [f"director: {why}", f"planner score {c.score:+.2f}"] + c.reasons
        transitions.append(t)
        min_out = min_out_after(c, m.b)
    directed = plan.model_copy(update={
        "transitions": transitions,
        "director": {"title": direction.title, "narrative": direction.narrative, **usage, "repairs": repairs},
    })
    return directed, repairs


def direct(plan: MixPlan, analyses: list[Analysis], model: str = MODEL,
           client: anthropic.Anthropic | None = None) -> tuple[MixPlan, list[str]]:
    menus = build_options(analyses, plan)
    direction, usage = ask_claude(build_prompt(plan, analyses, menus), model, client)
    return apply_direction(plan, menus, direction, usage)
