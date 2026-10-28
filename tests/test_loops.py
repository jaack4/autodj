import numpy as np
import pytest

from autodj.loops import extend_analysis, loop_options, splice_audio
from autodj.models import Loop
from autodj.planner import candidates

from conftest import make_analysis


def _short_tail():
    """intro 16 / drop 16-32 / break 32-48 / drop 48-64 / outro 64-68: only 4 bars after the last drop."""
    return make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 32, 0.9), ("break", 32, 48, 0.5),
                          ("drop", 48, 64, 0.95), ("outro", 64, 68, 0.3)], vocals=[0.0] * 68)


def test_loop_options_only_for_short_tails(club_track):
    assert loop_options(club_track()) == []  # 16-bar outro: room enough
    opts = loop_options(_short_tail())
    assert Loop(at_bar=64, bars=8, repeats=2) in opts and Loop(at_bar=64, bars=4, repeats=3) in opts


def test_extended_analysis_inserts_the_loop():
    a = _short_tail()
    v = extend_analysis(a, Loop(at_bar=64, bars=8, repeats=2))
    bar = a.downbeats[1] - a.downbeats[0]
    assert v.n_bars == 68 + 16
    assert v.downbeats[64] == pytest.approx(a.downbeats[56] + 8 * bar)  # first repeat replays bar 56
    assert v.downbeats[80] == pytest.approx(a.downbeats[64] + 16 * bar)  # the outro, shifted
    assert v.duration == pytest.approx(a.duration + 16 * bar)
    labels = [(s.label, s.start_bar, s.end_bar) for s in v.sections]
    assert ("drop", 48, 64) in labels and ("loop", 64, 80) in labels and ("outro", 80, 84) in labels
    assert v.bar_energy[64:72] == a.bar_energy[56:64]
    assert a.n_bars == 68  # the original is untouched


def test_splice_repeats_the_audio():
    a = _short_tail()
    sr = 1000
    audio = np.arange(int(a.duration * sr) + 10, dtype=np.float32)[None].repeat(2, 0)
    loop = Loop(at_bar=64, bars=4, repeats=2)
    out = splice_audio(audio, sr, a, loop)
    s0, s1 = int(round(a.downbeats[60] * sr)), int(round(a.downbeats[64] * sr))
    assert out.shape[1] == audio.shape[1] + 2 * (s1 - s0)
    mid = s1 + (s1 - s0) // 2
    assert out[0, mid] == audio[0, s0 + (s1 - s0) // 2]  # inside the first repeat: replays the block


def test_planner_loops_a_short_tail_to_blend_after_the_last_drop():
    a, b = _short_tail(), make_analysis([("intro", 0, 16, 0.4), ("drop", 16, 48, 0.9), ("outro", 48, 64, 0.4)],
                                        vocals=[0.0] * 64)
    looped = [c for c in candidates(a, b, style="smooth") if c.loop]
    assert looped
    best = max(looped, key=lambda c: c.score)
    assert best.out_bar >= 64 - 4 and best.overlap >= 8  # blends from the loop, after the drop has played
    assert any("loops A's last" in r for r in best.reasons)
