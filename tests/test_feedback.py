import dataclasses

from autodj import feedback
from autodj.models import MixPlan, PlanTrack, Transition
from autodj.planner import candidates
from autodj.setlist import _with_feedback, pair_cost


def test_record_and_load_round_trip(tmp_path):
    plan = MixPlan(tracks=[PlanTrack(path="/m/A.mp3"), PlanTrack(path="/m/B.mp3")],
                   transitions=[Transition(type="stem_blend", out_bar=96, in_bar=0, overlap_bars=16)])
    path = tmp_path / "feedback.json"
    feedback.record(plan, 1, liked=True, note="great", path=path)
    feedback.record(plan, 1, liked=False, path=path)  # replaces the earlier verdict on the same transition
    (v,) = feedback.load(path)
    assert (v.from_track, v.to_track, v.liked, v.out_bar, v.overlap) == ("A", "B", False, 96, 16)


def test_liked_transition_wins_and_pair_is_cheaper(club_track):
    a, b = club_track(path="/m/A.mp3"), club_track(path="/m/B.mp3")
    ranked = candidates(a, b, style="smooth")
    second = ranked[1]
    liked = feedback.Verdict("A", "B", True, second.kind, second.out_bar, second.in_bar, second.overlap, second.loop)
    boosted = _with_feedback(ranked, a, b, [liked])
    assert (boosted[0].kind, boosted[0].out_bar, boosted[0].overlap) == (second.kind, second.out_bar, second.overlap)
    assert any("you liked" in r for r in boosted[0].reasons)
    assert pair_cost(a, b, ranked[0], "flat", verdicts=[liked]) < pair_cost(a, b, ranked[0], "flat")
    disliked = dataclasses.replace(liked, liked=False)
    assert _with_feedback(ranked, a, b, [disliked])[0].out_bar != second.out_bar or \
        _with_feedback(ranked, a, b, [disliked])[0].kind != second.kind
