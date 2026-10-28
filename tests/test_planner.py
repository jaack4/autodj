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


def test_compatible_keys_prefer_a_blend(club_track):
    c = best_transition(club_track(key="8A"), club_track(key="9A"))
    assert c.kind == "bass_swap_blend"


def test_clashing_keys_prefer_a_cut(club_track):
    c = best_transition(club_track(key="8A"), club_track(key="2B"))
    assert c.kind in ("echo_out", "drop_cut")
    assert c.overlap == 0


def test_drop_cut_goes_from_a_build_into_a_drop():
    a = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("build", 48, 56, 0.6),
                       ("drop", 56, 88, 0.95), ("outro", 88, 96, 0.4)], key="8A")
    b = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("outro", 48, 64, 0.4)], key="3B")
    cuts = [c for c in candidates(a, b) if c.kind == "drop_cut"]
    assert cuts and all(a.section_at(c.out_bar).label == "drop" for c in cuts)
    assert all(b.section_at(c.in_bar).label == "drop" for c in cuts)


def _vocals(n, on):
    v = [0.0] * n
    for s, e in on:
        v[s:e] = [1.0] * (e - s)
    return v


def test_blends_avoid_two_vocals_at_once(club_track):
    a = club_track(vocals=_vocals(112, [(96, 112)]))  # A sings through its outro
    b_quiet = club_track(vocals=_vocals(112, [(16, 48)]))
    b_singing = club_track(vocals=_vocals(112, [(0, 16)]))  # B sings over its intro
    quiet = best_transition(a, b_quiet, kinds=("bass_swap_blend",))
    singing = best_transition(a, b_singing, kinds=("bass_swap_blend",))
    assert singing.score < quiet.score
    assert not any("vocals at once" in r for r in quiet.reasons)


def test_acapella_overlay_needs_stems_and_compatible_keys(club_track):
    v = _vocals(112, [(0, 16)])
    assert not candidates(club_track(), club_track(), kinds=("acapella_overlay",))
    assert not candidates(club_track(vocals=v), club_track(key="3B", vocals=v), kinds=("acapella_overlay",))
    found = candidates(club_track(vocals=v), club_track(vocals=v), kinds=("acapella_overlay",))
    assert found
    best = found[0]
    assert best.in_bar + best.overlap == 16  # B's vocal phrase ends where its drop comes in
    assert any("drop" in r for r in best.reasons)


def test_short_tracks_fall_back_to_an_echo_out():
    tiny = make_analysis([("intro", 0, 1, 0.3), ("outro", 1, 5, 0.9)])
    c = best_transition(tiny, tiny)
    assert c.kind == "echo_out" and c.out_bar < tiny.n_bars


def test_smooth_style_only_blends(club_track):
    found = candidates(club_track(key="8A"), club_track(key="2B"), style="smooth")
    assert found and {c.kind for c in found} <= {"bass_swap_blend", "filter_sweep"}


def test_no_loud_entry_over_a_quiet_outro():
    # A ends with a long quiet outro; B's intro is loud. The planner should leave A before its fade.
    a = make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("body", 48, 80, 0.7),
                       ("outro", 80, 112, 0.1)])
    b = make_analysis([("intro", 0, 16, 0.7), ("drop", 16, 48, 0.95), ("outro", 48, 64, 0.3)])
    c = best_transition(a, b, style="smooth")
    assert c.out_bar + c.overlap <= 80  # blends out of A's body, not its faded outro
    from_outro = [x for x in candidates(a, b, style="smooth") if x.out_bar >= 80]
    assert from_outro and all(any("faded" in r or "intense" in r for r in x.reasons) for x in from_outro)


def test_stem_blend_needs_stems_and_rewards_carried_vocals(club_track):
    assert not candidates(club_track(), club_track(), kinds=("stem_blend",))  # no stems
    v_outro = _vocals(112, [(96, 112)])  # A sings through its outro
    a = club_track(vocals=v_outro)
    b = club_track(vocals=_vocals(112, []))
    best = best_transition(a, b, style="smooth")
    assert best.kind == "stem_blend"
    assert any("carry across" in r for r in best.reasons)
    assert not candidates(a, club_track(key="3B", vocals=v_outro), kinds=("stem_blend",))  # clashing keys
