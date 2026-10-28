import itertools

import numpy as np
import pytest

from autodj.models import PlanTrack
from autodj.planner import tempo_gap
from autodj.setlist import Entry, _held_karp, _path_cost, load_playlist, order_tracks


def test_tempo_gap_allows_half_and_double_time():
    assert tempo_gap(87, 174) == 0
    assert tempo_gap(126, 128) == pytest.approx(tempo_gap(128, 126))
    assert tempo_gap(126, 128) == pytest.approx(0.0158, abs=1e-3)


def test_held_karp_finds_the_cheapest_open_path():
    rng = np.random.default_rng(1)
    cost = rng.uniform(0, 1, (7, 7))
    best = min(itertools.permutations(range(7)), key=lambda o: _path_cost(list(o), cost))
    assert _path_cost(_held_karp(cost), cost) == pytest.approx(_path_cost(list(best), cost))


def test_order_follows_keys_and_tempo(club_track):
    # A shuffled 8A -> 9A -> 10A -> 11A walk with rising tempo; the planner should recover it.
    specs = [("10A", 126.0), ("8A", 122.0), ("11A", 128.0), ("9A", 124.0)]
    entries = [Entry(PlanTrack(path=f"/{k}.mp3"), club_track(key=k, bpm=b, path=f"/{k}.mp3")) for k, b in specs]
    order, _ = order_tracks(entries, arc="warmup_peak")
    assert [specs[i][0] for i in order] == ["8A", "9A", "10A", "11A"]


def test_load_playlist_yaml_resolves_paths_and_keys(tmp_path):
    (tmp_path / "a.mp3").touch()
    playlist = tmp_path / "set.yaml"
    playlist.write_text("name: Test\nkeep_order: true\ntracks:\n  - path: a.mp3\n    bpm: 124\n    key: A minor\n")
    name, tracks, options = load_playlist(playlist)
    assert name == "Test" and options == {"keep_order": True}
    assert tracks[0].path == str(tmp_path / "a.mp3")
    assert tracks[0].bpm_hint == 124 and tracks[0].key == "8A"


def test_load_playlist_folder(tmp_path):
    for n in ("b.flac", "a.mp3", "notes.txt"):
        (tmp_path / n).touch()
    _, tracks, _ = load_playlist(tmp_path)
    assert [t.path.split("/")[-1] for t in tracks] == ["a.mp3", "b.flac"]


def test_unreadable_files_are_skipped(tmp_path):
    from autodj.setlist import analyze_entries

    (tmp_path / "broken.mp3").write_text("not audio")
    logged = []
    entries = analyze_entries([PlanTrack(path=str(tmp_path / "broken.mp3"))], log=logged.append)
    assert entries == []
    assert any("skipped" in m for m in logged)


def test_best_shift_fixes_clashes_cheaply():
    from autodj.setlist import best_shift

    assert best_shift("8A", "8A") == (0, 0.0)
    assert best_shift("8A", "9A")[0] == 0  # already mixable: don't touch it
    s, cost = best_shift("6B", "11B")  # 5 steps apart: one semitone up turns 11B into 6B
    assert s == 1 and cost == pytest.approx(0.2)
    s, cost = best_shift("8A", "10A")  # 2 steps: two semitones
    assert s == -2 and cost == pytest.approx(0.5)
    assert best_shift("8A", "3A", max_shift=0)[0] == 0


def test_shifts_chain_from_the_previous_shifted_key(club_track):
    from autodj.camelot import shift_key
    from autodj.setlist import assign_shifts

    entries = [Entry(PlanTrack(path=f"/{k}.mp3"), club_track(key=k, path=f"/{k}.mp3")) for k in ("8A", "1A", "1A")]
    shifted = assign_shifts(entries)
    semis = [e.track.semitones for e, _ in shifted]
    assert semis == [0, 1, 1]  # 1A +1 semitone = 8A; the third then matches the shifted second
    assert [a.key for _, a in shifted] == ["8A", "8A", "8A"]
    assert shift_key("1A", 1) == "8A"


def test_uncertain_keys_are_not_shifted(club_track):
    from autodj.setlist import assign_shifts

    entries = [Entry(PlanTrack(path="/a.mp3"), club_track(key="8A")),
               Entry(PlanTrack(path="/b.mp3"), club_track(key="1A", key_confidence=0.0))]
    assert [e.track.semitones for e, _ in assign_shifts(entries)] == [0, 0]


def test_set_varies_transition_types(club_track):
    from autodj.setlist import _varied_transition

    a, b = club_track(key="8A"), club_track(key="8A")
    first = _varied_transition(a, b, 0, [])
    second = _varied_transition(a, b, 0, [first.kind, first.kind])
    assert second.kind != first.kind


def test_refine_order_never_makes_the_set_worse(club_track):
    from autodj.setlist import order_cost, plan_order, refine_order

    specs = [("10A", 126.0), ("8A", 122.0), ("11A", 128.0), ("9A", 124.0), ("3B", 125.0)]
    entries = [Entry(PlanTrack(path=f"/{k}.mp3"), club_track(key=k, bpm=b, path=f"/{k}.mp3")) for k, b in specs]
    start = [0, 1, 2, 3, 4]
    cache: dict = {}
    refined = refine_order(entries, start, "warmup_peak", 2, "smooth", cache, log=lambda m: None)
    cost = lambda o: order_cost(*plan_order([entries[i] for i in o], 2, "smooth", cache), "warmup_peak")
    assert sorted(refined) == start and cost(refined) <= cost(start)


def test_journey_target_shape():
    from autodj.setlist import journey_target

    curve = [journey_target(k, 11) for k in range(11)]
    assert curve[0] == pytest.approx(0.2) and max(curve) == pytest.approx(1.0)
    assert curve.index(max(curve)) == 7 and curve[-1] == pytest.approx(0.5)


def test_variety_penalty_keeps_the_loop():
    from autodj.setlist import _varied_transition
    from conftest import make_analysis

    a = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 32, 0.9), ("break", 32, 48, 0.5),
                       ("drop", 48, 64, 0.95), ("outro", 64, 68, 0.3)], vocals=[0.0] * 68)
    b = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("outro", 48, 64, 0.4)], vocals=[0.0] * 64)
    first = _varied_transition(a, b, 0, [], "smooth")
    again = _varied_transition(a, b, 0, [first.kind, first.kind], "smooth")
    for c in (first, again):
        assert c.out_bar + c.overlap <= a.n_bars or c.loop is not None  # looped bars always come with their loop
