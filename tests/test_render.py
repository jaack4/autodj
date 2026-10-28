import numpy as np
import pedalboard
import pytest

from autodj.render import _fade, _split_bands

SR = 44100


def test_band_split_reconstructs_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, SR))
    bands = _split_bands(x, SR)
    assert np.allclose(sum(bands.values()), x)


def test_equal_power_fades():
    up, down = _fade(0, 1, rising=True), _fade(0, 1, rising=False)
    power = np.array([g for _, g in up]) ** 2 + np.array([g for _, g in down]) ** 2
    assert np.allclose(power, 1)


@pytest.mark.parametrize("rate", [0.95, 1.055])
def test_time_stretch_keeps_beats_in_place(rate):
    """Guards the high_quality=False choice: R3 drifts ~50 ms over a track."""
    times = np.arange(0.5, 40, 60 / 126)
    burst = (np.sin(2 * np.pi * 60 * np.arange(800) / SR) * np.exp(-np.arange(800) / 150)).astype(np.float32)
    y = np.zeros(int(41 * SR), np.float32)
    for t in times:
        y[int(t * SR) : int(t * SR) + 800] += burst
    s = pedalboard.time_stretch(np.stack([y, y]), SR, stretch_factor=rate, high_quality=False)[0]
    errors = []
    for t in times[2:-2]:
        lo = int((t / rate - 0.06) * SR)
        cc = np.correlate(s[lo : lo + int(0.12 * SR) + 800], burst, mode="valid")
        errors.append((lo + int(np.argmax(cc))) / SR - t / rate)
    errors = np.array(errors) * 1000
    assert np.abs(errors).max() < 3.0
    assert np.ptp(errors) < 1.0


def _decks(plan, analyses):
    from autodj.render import Deck

    return [Deck(analysis=a, target_bpm=plan.bpm or a.bpm, gain=1.0) for a in analyses]


def _plan(transitions, bpm=None):
    from autodj.models import MixPlan, PlanTrack

    return MixPlan(bpm=bpm, tracks=[PlanTrack(path=f"/t{i}.mp3") for i in range(len(transitions) + 1)],
                   transitions=transitions)


def test_incoming_track_matches_tempo_during_blend_then_ramps(club_track):
    from autodj.models import Transition
    from autodj.render import place

    a, b = club_track(bpm=126.0), club_track(bpm=130.0)
    plan = _plan([Transition(out_bar=96, in_bar=0, overlap_bars=16, ramp_bars=16)])
    da, db = _decks(plan, [a, b])
    place(plan, [da, db])
    for k in range(17):  # every bar line in the blend coincides
        assert da.offset + da.bar_time(96 + k) == pytest.approx(db.offset + db.bar_time(k), abs=0.002)
    blend_bar = db.bar_time(8) - db.bar_time(7)
    settled_bar = db.bar_time(60) - db.bar_time(59)
    assert blend_bar == pytest.approx(240 / 126, abs=1e-3)
    assert settled_bar == pytest.approx(240 / 130, abs=1e-3)
    assert da.curve == []  # the outgoing track is never stretched


def test_ramp_stops_before_the_next_blend(club_track):
    from autodj.models import Transition
    from autodj.render import place

    a, b, c = club_track(bpm=126.0), club_track(bpm=130.0), club_track(bpm=130.0)
    plan = _plan([Transition(out_bar=96, in_bar=0, overlap_bars=16, ramp_bars=64),
                  Transition(out_bar=40, in_bar=0, overlap_bars=16)])
    da, db, dc = _decks(plan, [a, b, c])
    place(plan, [da, db, dc])
    assert db.tempo_at_bar(40) == pytest.approx(130.0, abs=0.01)
    for k in range(17):
        assert db.offset + db.bar_time(40 + k) == pytest.approx(dc.offset + dc.bar_time(k), abs=0.002)


def test_locked_tempo_plan_has_no_ramps(club_track):
    from autodj.models import Transition
    from autodj.render import place

    plan = _plan([Transition(out_bar=96, in_bar=0, overlap_bars=16)], bpm=128.0)
    da, db = _decks(plan, [club_track(bpm=126.0), club_track(bpm=130.0)])
    place(plan, [da, db])
    assert da.curve == [] and db.curve == []
    assert db.bar_time(50) - db.bar_time(49) == pytest.approx(240 / 128, abs=1e-6)


def test_time_stretch_follows_a_tempo_ramp():
    """The time map the renderer computes must match what Rubber Band actually does with a ramp."""
    times = np.arange(0.5, 40, 60 / 126)
    burst = (np.sin(2 * np.pi * 60 * np.arange(800) / SR) * np.exp(-np.arange(800) / 150)).astype(np.float32)
    y = np.zeros(int(41 * SR), np.float32)
    for t in times:
        y[int(t * SR) : int(t * SR) + 800] += burst
    t_in = np.arange(len(y)) / SR
    f = np.interp(t_in, [0, 10, 25, 41], [1.05, 1.05, 1.0, 1.0])
    s = pedalboard.time_stretch(np.stack([y, y]), SR, stretch_factor=f, high_quality=False)[0]
    out_t = np.concatenate([[0], np.cumsum(1 / f)]) / SR
    expected = np.interp(times * SR, np.arange(len(y) + 1), out_t)
    errors = []
    for t in expected[1:-2]:
        lo = int((t - 0.06) * SR)
        cc = np.correlate(s[lo : lo + int(0.12 * SR) + 800], burst, mode="valid")
        errors.append((lo + int(np.argmax(cc))) / SR - t)
    assert np.abs(errors).max() * 1000 < 6.0
