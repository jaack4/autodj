import numpy as np
import pytest

from autodj.analysis.beats import comb_grid


def _click_track(bpm, seconds, sr=22050):
    import librosa

    times = np.arange(0.25, seconds, 60 / bpm)
    return librosa.clicks(times=times, sr=sr, click_duration=0.02, length=int(seconds * sr)), times


def test_grid_alignment_flags_wrong_tempo():
    from autodj.analysis.beats import grid_alignment, local_alignment

    def drift(beats, period):
        return grid_alignment(*local_alignment(y, 22050, beats, period))[1]

    y, _ = _click_track(123.3, 140)
    good = 0.25 + (60 / 123.3) * np.arange(280)
    bad = 0.25 + (60 / 122.4) * np.arange(280)  # 0.7% slow: slides ~2 beats over the track
    assert abs(drift(good, 60 / 123.3)) < 10
    assert abs(drift(bad, 60 / 122.4)) > 300


def test_local_alignment_locates_a_splice():
    from autodj.analysis.beats import local_alignment

    sr = 22050
    period = 60 / 126
    # 64 beats on grid, then everything 100 ms late (an edit that isn't a whole number of beats)
    times = np.r_[0.3 + period * np.arange(64), 0.4 + period * np.arange(64, 128)]
    y = np.zeros(int((times[-1] + 1) * sr))
    burst = np.sin(2 * np.pi * 55 * np.arange(1000) / sr) * np.exp(-np.arange(1000) / 200)
    for t in times:
        y[int(t * sr) : int(t * sr) + 1000] += burst
    offsets, confidence = local_alignment(y, sr, 0.3 + period * np.arange(128), period)
    assert (confidence > 1.5).all()
    assert np.abs(offsets[:4]).max() < 0.01
    assert np.abs(offsets[4:] - 0.1).max() < 0.01


def _kick_track(bpm, seconds, start=0.31, sr=22050, seed=0):
    """Low sine-burst 'kicks' with timing jitter, dropouts and noise."""
    rng = np.random.default_rng(seed)
    y = 0.01 * rng.standard_normal(int(seconds * sr))
    t = np.arange(int(0.08 * sr)) / sr
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 40)
    for i, time in enumerate(np.arange(start, seconds - 0.1, 60 / bpm)):
        if i % 11 == 5:
            continue
        k = int((time + rng.normal(0, 0.003)) * sr)
        y[k : k + len(kick)] += kick[: len(y) - k]
    return y


def test_comb_grid_recovers_tempo_and_phase():
    y = _kick_track(126.0, 120)
    period, first = comb_grid(y, 22050, 120)
    assert 60 / period == pytest.approx(126.0, abs=0.02)
    assert first == pytest.approx(0.31, abs=0.015)


def test_comb_grid_prefers_double_time_on_ties():
    y = _kick_track(174.0, 120)  # an 87 BPM grid hits every other kick and scores the same
    period, _ = comb_grid(y, 22050, 120)
    assert 60 / period == pytest.approx(174.0, abs=0.05)


def test_comb_grid_keeps_half_time_when_offbeats_are_empty():
    y = _kick_track(87.0, 120)
    period, _ = comb_grid(y, 22050, 120)
    assert 60 / period == pytest.approx(87.0, abs=0.05)


def test_comb_grid_respects_hint():
    y = _kick_track(87.0, 120)
    period, _ = comb_grid(y, 22050, 120, bpm_hint=87)
    assert 60 / period == pytest.approx(87.0, abs=0.05)
