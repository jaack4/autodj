import itertools

import numpy as np
import pytest

from autodj.models import PlanTrack
from autodj.setlist import Entry, _held_karp, _path_cost, load_playlist, order_tracks, tempo_gap


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
