import pytest

from autodj.sections_eval import match_boundaries, score_track, summarize

from conftest import make_analysis


def test_match_boundaries_is_one_to_one():
    assert match_boundaries([10.0, 10.5, 30.0], [10.2, 29.0], tolerance=1.0) == 2
    assert match_boundaries([10.0], [10.2, 10.4], tolerance=1.0) == 1


def test_perfect_labels_score_100(club_track):
    a = club_track()
    truth = [{"start": s.start, "label": s.label} for s in a.sections]
    s = score_track(a, truth)
    assert (s.precision, s.recall, s.label_accuracy) == (1.0, 1.0, 1.0)


def test_shifted_and_mislabelled_sections(club_track):
    a = club_track()  # intro 0-16, drop 16-48, break 48-64, drop 64-96, outro 96-112
    bar = a.downbeats[1] - a.downbeats[0]
    truth = [{"start": a.downbeats[0], "label": "intro"},
             {"start": a.downbeats[18], "label": "drop"},  # 2 bars late: a miss at 1-bar tolerance
             {"start": a.downbeats[48], "label": "break"},
             {"start": a.downbeats[64], "label": "build"},  # we say drop
             {"start": a.downbeats[96], "label": "outro"}]
    s = score_track(a, truth, tolerance_bars=1.0)
    assert s.hits == 3 and s.truth == 4 and s.predicted == 4
    assert s.label_accuracy == pytest.approx((112 - 2 - 32) / 112)
    assert s.confusion[("build", "drop")] == 32
    assert summarize([s, s]).hits == 6
