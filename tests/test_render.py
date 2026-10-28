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
