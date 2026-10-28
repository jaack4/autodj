from autodj.planner import best_transition, candidates, region_offset_ms

from conftest import make_analysis


def test_blends_outro_into_intro(club_track):
    a, b = club_track(path="/a.mp3"), club_track(path="/b.mp3")
    c = best_transition(a, b)
    assert (c.out_bar, c.in_bar, c.overlap) == (96, 0, 16)
    assert c.swap_bar == 8


def test_clashing_keys_get_short_blends(club_track):
    a, b = club_track(key="8A"), club_track(key="2B")
    assert all(c.overlap <= 8 for c in candidates(a, b))


def test_uncertain_keys_dont_force_short_blends(club_track):
    a, b = club_track(key="8A", key_confidence=0.02), club_track(key="2B")
    assert max(c.overlap for c in candidates(a, b)) == 16


def test_avoids_blending_through_a_splice(club_track):
    offsets = [0.0] * 100 + [110.0] * 12  # outro is spliced 110 ms late from bar 100
    a = club_track(offsets_ms=offsets)
    c = best_transition(a, club_track())
    assert c.out_bar + c.overlap <= 100 or c.out_bar >= 100
    assert not any("splice" in r for r in c.reasons)


def test_region_offset_ignores_single_outliers(club_track):
    offsets = [0.0] * 112
    offsets[40] = 45.0  # one syncopated window
    a = club_track(offsets_ms=offsets)
    median, consistent = region_offset_ms(a, 32, 48)
    assert median == 0.0 and consistent


def test_swap_lands_on_incoming_section_change():
    a = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("outro", 48, 64, 0.4)])
    b = make_analysis([("intro", 0, 4, 0.3), ("build", 4, 16, 0.5), ("drop", 16, 48, 0.9), ("outro", 48, 64, 0.4)])
    c = best_transition(a, b)
    assert (c.out_bar, c.in_bar, c.overlap) == (48, 0, 16)
    assert c.swap_bar == 4  # the build starts 4 bars in; closest 4-bar-aligned section change to halfway is 4
