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


def test_lowpass_mask_attenuates_above_cutoff():
    from autodj.render import _apply_filters

    t = np.arange(SR * 2) / SR
    x = np.stack([np.sin(2 * np.pi * 5000 * t)] * 2)
    closed = _apply_filters(x, t, [("lowpass", [(0.0, 1000.0), (2.0, 1000.0)])], SR)
    opened = _apply_filters(x, t, [("lowpass", [(0.0, 20000.0), (2.0, 20000.0)])], SR)
    rms = lambda y: np.sqrt(np.mean(y[:, SR // 4 : -SR // 4] ** 2))
    assert rms(closed) < 0.01 * rms(x)
    assert rms(opened) == pytest.approx(rms(x), rel=0.02)


def test_echo_tail_repeats_on_the_delay_grid():
    from autodj.render import Echo, _echo_tail

    beat = 0.5
    seg = np.zeros((2, SR * 4))
    seg[:, int(1.6 * SR)] = 1.0  # a click inside the beat before the cut at 2.0 s
    e = Echo(cut=2.0, beat=beat, delay_beats=0.75, feedback=0.5)
    start, tail = _echo_tail(seg, 0.0, e, SR)
    assert start == int(1.5 * SR)
    repeat = lambda k: int((0.1 + k * 0.375) * SR)
    peaks = [k for k in range(1, 4) if np.abs(tail[0, repeat(k) - 200 : repeat(k) + 200]).max() > 0.01]
    assert peaks == [1, 2, 3]
    assert np.abs(tail[0, : int(0.4 * SR)]).max() < 0.01  # nothing before the first repeat


@pytest.mark.parametrize("kind", ["echo_out", "drop_cut"])
def test_cuts_start_the_incoming_track_on_the_cut(club_track, kind):
    from autodj.models import Transition
    from autodj.render import Echo, apply_transition, place

    plan = _plan([Transition(type=kind, out_bar=64, in_bar=16, overlap_bars=0)])
    da, db = _decks(plan, [club_track(bpm=126.0), club_track(bpm=126.0)])
    (w,) = place(plan, [da, db])
    apply_transition(w)
    cut = da.offset + da.bar_time(64)
    assert db.offset + db.play_from == pytest.approx(cut)
    if kind == "echo_out":
        assert da.play_to + da.offset > cut + Echo(cut, w.beat, 0.75, 0.55).tail
        assert da.echo is not None
    gains = lambda d, t: np.interp(t, *zip(*d.automation["mid"]))
    assert gains(da, cut + 0.01) == 0 and gains(db, cut + 0.01) == 1
    assert gains(da, cut - 0.01) == 1 and gains(db, cut - 0.01) == 0


def test_overlap_must_match_transition_type(club_track):
    from autodj.models import Transition
    from autodj.render import place

    plan = _plan([Transition(type="echo_out", out_bar=64, in_bar=16, overlap_bars=16)])
    with pytest.raises(ValueError, match="overlap_bars"):
        place(plan, _decks(plan, [club_track(), club_track()]))


def test_acapella_overlay_mixes_stems(club_track):
    from autodj.models import Transition
    from autodj.render import OVERLAY_BLEND_OUT_BARS, _deck_signal, apply_transition, place

    plan = _plan([Transition(type="acapella_overlay", out_bar=64, in_bar=0, overlap_bars=16)])
    da, db = _decks(plan, [club_track(bpm=120.0), club_track(bpm=120.0)])
    (w,) = place(plan, [da, db])
    apply_transition(w)
    sr = 8000
    t = np.arange(int(300 * sr)) / sr
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)  # mid band, untouched by bass swaps
    full = np.stack([tone, tone])
    stems = {name: full * 0.25 for name in ("drums", "bass", "other", "vocals")}  # sums to the mix
    ma, sa = _deck_signal(da, full, sr, stems)
    mb, sb = _deck_signal(db, full, sr, stems)

    def level(m0, sig, when):
        i = int(when * sr) - m0
        return float(np.sqrt(2 * np.mean(sig[0, i : i + sr // 10] ** 2)))

    mid = (w.start + w.end) / 2
    assert level(ma, sa, mid) == pytest.approx(0.75, abs=0.03)  # A: instrumental only
    assert level(mb, sb, mid) == pytest.approx(0.25, abs=0.03)  # B: acapella only
    assert level(mb, sb, w.end - 2 * w.beat) > 0.4  # B's beat fading in before the end
    assert level(mb, sb, w.end + 1) == pytest.approx(1.0, abs=0.03)  # then B in full
    assert da.offset + da.play_to == pytest.approx(w.end + OVERLAY_BLEND_OUT_BARS * 4 * w.beat)  # A fades out


def test_stem_blend_swaps_parts_and_carries_vocals(club_track):
    from autodj.models import Transition
    from autodj.render import _deck_signal, apply_transition, place

    plan = _plan([Transition(type="stem_blend", out_bar=80, in_bar=0, overlap_bars=16, swap_bar=8)])
    da, db = _decks(plan, [club_track(bpm=120.0), club_track(bpm=120.0)])
    (w,) = place(plan, [da, db])
    apply_transition(w)
    sr = 8000
    t = np.arange(int(300 * sr)) / sr
    tone = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)  # mid band only

    def part(name):  # a deck where only one stem is present, to read that stem's gain
        stems = {n: tone * (1.0 if n == name else 0.0) for n in ("drums", "bass", "other", "vocals")}
        return stems

    def level(d, name, when):
        m0, sig = _deck_signal(d, part(name)["drums"] * 0 + sum(part(name).values()), sr, part(name))
        i = int(when * sr) - m0
        return float(np.sqrt(2 * np.mean(sig[0, i : i + sr // 20] ** 2))) if 0 <= i < sig.shape[1] else 0.0

    bar = 4 * w.beat
    early, late = w.start + 2 * bar, w.end - 2 * bar
    assert level(db, "drums", early) > 0.1 and level(db, "other", early) < 0.05  # B's drums creep in first
    assert level(da, "drums", late) < 0.05 and level(db, "drums", late) > 0.95  # drums swapped
    assert level(da, "vocals", w.end - 0.5 * bar) > 0.95  # A's vocals carry across the whole blend
    assert level(db, "vocals", late) < 0.05  # B's vocals wait
    assert level(da, "vocals", w.end + 1.5 * bar) == 0.0  # and A has finished a bar later
