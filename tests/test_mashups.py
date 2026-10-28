from autodj.mashups import find_mashups
from autodj.models import MixPlan, PlanTrack, Transition

from conftest import make_analysis


def _track(key, vocals_on, bpm=126.0, path="/t.mp3"):
    sections = [("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("break", 48, 64, 0.5),
                ("drop", 64, 96, 0.9), ("outro", 96, 112, 0.4)]
    v = [0.0] * 112
    for s, e in vocals_on:
        v[s:e] = [1.0] * (e - s)
    return make_analysis(sections, key=key, bpm=bpm, path=path, vocals=v)


def _plan(analyses):
    transitions = [Transition(type="stem_blend", out_bar=96, in_bar=0, overlap_bars=16) for _ in analyses[1:]]
    return MixPlan(tracks=[PlanTrack(path=a.path) for a in analyses], transitions=transitions)


def test_finds_a_vocal_from_another_track_over_an_instrumental_drop():
    host = _track("8A", vocals_on=[], path="/host.mp3")  # instrumental drops
    mid = _track("8A", vocals_on=[(0, 112)], path="/mid.mp3")  # sings everywhere: never a host
    guest = _track("8A", vocals_on=[(48, 64)], path="/guest.mp3")  # a 16-bar sung break
    analyses = [host, mid, guest]
    found = find_mashups(_plan(analyses), analyses)
    assert found
    m = found[0]
    assert (m.host, m.guest, m.guest_bar, m.bars) == (0, 2, 48, 16)
    assert analyses[0].section_at(m.host_bar).label == "drop"
    assert any("teaser" in n for n in m.notes)


def test_no_mashups_without_stems_or_compatible_tempo():
    host, guest = _track("8A", []), _track("8A", [(48, 64)])
    no_stems = [a.model_copy(update={"bar_vocals": None}) for a in (host, guest, host)]
    assert find_mashups(_plan(no_stems), no_stems) == []
    fast = [host, host, _track("8A", [(48, 64)], bpm=150.0)]
    assert find_mashups(_plan(fast), fast) == []


def test_guest_gets_shifted_into_key():
    host = _track("8A", [], path="/host.mp3")
    guest = _track("1A", [(48, 64)], path="/guest.mp3")  # +1 semitone -> 8A
    analyses = [host, _track("8A", [(0, 112)], path="/mid.mp3"), guest]
    m = find_mashups(_plan(analyses), analyses)[0]
    assert m.semitones == 1
