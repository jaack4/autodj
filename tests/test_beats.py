import numpy as np
import pytest

from autodj.analysis.beats import fit_grid


def _jittered_beats(bpm, start, n, jitter_ms, seed=0):
    rng = np.random.default_rng(seed)
    period = 60 / bpm
    return start + period * np.arange(n) + rng.normal(0, jitter_ms / 1000, n)


def test_fit_grid_recovers_tempo_and_phase():
    beats = _jittered_beats(124.0, 0.37, 400, jitter_ms=8)
    period, first, resid = fit_grid(beats, duration=200)
    assert 60 / period == pytest.approx(124.0, abs=0.01)
    assert first == pytest.approx(0.37, abs=0.005)
    assert np.median(np.abs(resid)) < 0.010


def test_fit_grid_ignores_missing_and_spurious_beats():
    beats = _jittered_beats(128.0, 0.1, 300, jitter_ms=5)
    beats = np.delete(beats, np.arange(20, 300, 7))  # dropouts
    beats = np.sort(np.r_[beats, [10.13, 50.71, 90.3]])  # off-grid hits
    period, _, _ = fit_grid(beats, duration=150)
    assert 60 / period == pytest.approx(128.0, abs=0.02)


def test_fit_grid_folds_half_time_with_hint():
    beats = _jittered_beats(70.0, 0.2, 200, jitter_ms=5)
    period, _, _ = fit_grid(beats, duration=180, bpm_hint=140)
    assert 60 / period == pytest.approx(140.0, abs=0.05)
