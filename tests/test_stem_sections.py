import numpy as np

from autodj.analysis.stem_sections import _merge_repeats, label_sections, repeat_groups


def _features(n, kick_on, brightness_ramp=()):
    f = {k: np.zeros(n) for k in ("kick", "hats", "bass", "other", "density", "brightness")}
    for a, b in kick_on:
        f["kick"][a:b] = 1.0
    f["brightness"][:] = 1000.0
    for a, b in brightness_ramp:
        f["brightness"][a:b] = np.linspace(1000, 3000, b - a)
    f["timbre"], f["chroma"] = np.zeros((n, 13)), np.zeros((n, 12))
    return f


def test_layers_decide_labels():
    edges = [0, 16, 24, 56, 72, 104, 112]
    f = _features(112, kick_on=[(24, 56), (72, 104), (104, 112)], brightness_ramp=[(16, 24)])
    energy = np.r_[np.full(16, 0.3), np.full(8, 0.5), np.full(32, 0.95), np.full(16, 0.3), np.full(32, 0.95), np.full(8, 0.4)]
    labels = label_sections(edges, f, energy)
    # intro, build (brightening before the drop), drop, break (no kick), drop, outro (kick, low energy, after last drop)
    assert labels == ["intro", "build", "drop", "break", "drop", "outro"]


def test_drops_do_not_need_the_bass_stem():
    f = _features(32, kick_on=[(16, 32)])  # bass stays 0 (808s filed under drums)
    energy = np.r_[np.full(16, 0.3), np.full(16, 0.95)]
    assert label_sections([0, 16, 32], f, energy)[1] == "drop"


def test_repeats_get_a_shared_group_and_merge():
    f = _features(48, kick_on=[])
    f["chroma"][:16, 0] = 1.0   # A
    f["chroma"][16:32, 7] = 1.0  # B
    f["chroma"][32:48, 0] = 1.0  # A again
    edges, labels = [0, 16, 32, 48], ["drop", "break", "drop"]
    groups = repeat_groups(edges, labels, f)
    assert groups[0] == groups[2] and groups[0] is not None and groups[1] is None
    e2, l2, g2 = _merge_repeats([0, 8, 16, 48], ["drop", "drop", "break"], ["A", "A", None])
    assert (e2, l2, g2) == ([0, 16, 48], ["drop", "break"], ["A", None])
