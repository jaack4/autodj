import json
from types import SimpleNamespace

import pytest

from autodj.director import DirectorError, build_options, build_prompt, direct
from autodj.models import MixPlan, PlanTrack
from autodj.planner import best_transition


class FakeClient:
    """Stands in for anthropic.Anthropic: returns a canned response, records the request."""

    def __init__(self, payload=None, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.request = payload, stop_reason, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        text = json.dumps(self.payload) if self.payload is not None else ""
        return SimpleNamespace(
            stop_reason=self.stop_reason, model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=12000, output_tokens=900),
        )


def _set(club_track, keys=("8A", "9A", "9A")):
    analyses = [club_track(key=k, path=f"/t{i}.mp3") for i, k in enumerate(keys)]
    transitions = []
    for a, b in zip(analyses, analyses[1:]):
        c = best_transition(a, b)
        transitions.append(c.transition())
    plan = MixPlan(name="test", tracks=[PlanTrack(path=a.path) for a in analyses], transitions=transitions)
    return plan, analyses


def test_options_include_the_current_pick_and_each_type(club_track):
    plan, analyses = _set(club_track)
    menus = build_options(analyses, plan)
    for m, current in zip(menus, plan.transitions):
        assert 1 < len(m.items) <= 6
        assert any((c.kind, c.out_bar, c.in_bar) == (current.type, current.out_bar, current.in_bar)
                   for c in m.items.values())
        assert {"bass_swap_blend", "filter_sweep"} <= {c.kind for c in m.items.values()}
    prompt = json.loads(build_prompt(plan, analyses, menus))
    assert len(prompt["tracks"]) == 3 and len(prompt["transitions"]) == 2


def test_director_picks_are_applied(club_track):
    plan, analyses = _set(club_track)
    menus = build_options(analyses, plan)
    pick = {m.index: next(oid for oid, c in m.items.items() if c.kind == "filter_sweep") for m in menus}
    client = FakeClient({"title": "Night Drive", "narrative": "Starts slow.", "choices": [
        {"transition": k, "option_id": oid, "why": f"reason {k}"} for k, oid in pick.items()]})
    directed, repairs = direct(plan, analyses, client=client)
    assert not repairs
    assert [t.type for t in directed.transitions] == ["filter_sweep", "filter_sweep"]
    assert directed.transitions[0].notes[0] == "director: reason 1"
    assert directed.director["title"] == "Night Drive"
    assert directed.director["cost_usd"] == pytest.approx(12000 * 4 / 1e6 + 900 * 20 / 1e6)
    req = client.request
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["output_config"]["format"]["type"] == "json_schema"


def test_unknown_option_keeps_the_planner_choice(club_track):
    plan, analyses = _set(club_track)
    client = FakeClient({"title": "t", "narrative": "n", "choices": [{"transition": 1, "option_id": "T9z", "why": "?"}]})
    directed, repairs = direct(plan, analyses, client=client)
    assert len(repairs) == 2  # one unknown id, one missing pick
    assert (directed.transitions[0].type, directed.transitions[0].out_bar) == (plan.transitions[0].type,
                                                                              plan.transitions[0].out_bar)


def test_overlapping_pick_is_replaced(club_track):
    plan, analyses = _set(club_track)
    menus = build_options(analyses, plan)
    # Enter track 2 late (a 16-bar blend from bar 16) then leave it early: impossible.
    first = next((oid for oid, c in menus[0].items.items() if c.in_bar > 0), next(iter(menus[0].items)))
    early = min(menus[1].items, key=lambda oid: menus[1].items[oid].out_bar)
    client = FakeClient({"title": "t", "narrative": "n", "choices": [
        {"transition": 1, "option_id": first, "why": "a"}, {"transition": 2, "option_id": early, "why": "b"}]})
    directed, repairs = direct(plan, analyses, client=client)
    t1, t2 = directed.transitions
    assert t2.out_bar >= t1.in_bar + t1.overlap_bars


def test_refusal_raises(club_track):
    plan, analyses = _set(club_track)
    with pytest.raises(DirectorError, match="declined"):
        direct(plan, analyses, client=FakeClient(stop_reason="refusal"))
