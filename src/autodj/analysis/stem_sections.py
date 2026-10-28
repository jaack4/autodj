"""Section detection from separated stems: in dance music, section types are defined by
which layers are playing (kick, risers, vocals), not just by overall loudness.

The bass stem isn't trusted for labels: Demucs often files 808s and distorted bass-music
basslines under drums or other, so drops can show almost no "bass"."""

import librosa
import numpy as np

from autodj.analysis.structure import find_boundaries, novelty
from autodj.audio import load as load_audio
from autodj.models import Analysis, Section

SR = 22050
HOP = 512
LEVEL_RANGE_DB = 24.0  # a layer this far below its loudest bars counts as off
FULL = 0.6  # kick/bass level that counts as "playing"
OFF = 0.35
REPEAT_SIMILARITY = 0.92


def _bar_index(n_frames: int, downbeats: list[float], bar_len: float) -> np.ndarray:
    times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP)
    idx = np.searchsorted(downbeats, times, side="right") - 1
    idx[(times >= downbeats[-1] + bar_len)] = -1
    return idx


def _per_bar(values: np.ndarray, idx: np.ndarray, n_bars: int) -> np.ndarray:
    """Mean of (dims, frames) per bar -> (n_bars, dims)."""
    ok = idx >= 0
    counts = np.maximum(np.bincount(idx[ok], minlength=n_bars), 1)
    return np.stack([np.bincount(idx[ok], weights=v[ok], minlength=n_bars) for v in np.atleast_2d(values)], 1) / counts[:, None]


def _level(power: np.ndarray) -> np.ndarray:
    """Per-bar power -> 0..1 'how much this layer is playing', relative to its loudest bars."""
    db = 10 * np.log10(power + 1e-12)
    top = np.percentile(db, 95)
    return np.clip((db - (top - LEVEL_RANGE_DB)) / LEVEL_RANGE_DB, 0, 1)


def stem_bar_features(stems: dict[str, str], mix_path: str, downbeats: list[float]) -> dict[str, np.ndarray]:
    """Per-bar layer levels and texture from the stems and the mix."""
    bar_len = float(np.diff(downbeats).mean())
    n = len(downbeats)
    drums, _ = load_audio(stems["drums"], sr=SR, mono=True)
    bass, _ = load_audio(stems["bass"], sr=SR, mono=True)
    other, _ = load_audio(stems["other"], sr=SR, mono=True)
    mix, _ = load_audio(mix_path, sr=SR, mono=True)

    def spec(y: np.ndarray) -> np.ndarray:
        return np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2

    d_spec = spec(drums)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    idx = _bar_index(d_spec.shape[1], downbeats, bar_len)
    kick = _per_bar(d_spec[freqs < 150].sum(0), idx, n)[:, 0]
    hats = _per_bar(d_spec[freqs > 4000].sum(0), idx, n)[:, 0]
    onsets = librosa.onset.onset_detect(y=drums, sr=SR, hop_length=HOP, units="frames")
    density = np.bincount(np.clip(idx[np.clip(onsets, 0, len(idx) - 1)], 0, None), minlength=n)[:n].astype(float)

    def power(y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0] ** 2
        return _per_bar(rms, _bar_index(len(rms), downbeats, bar_len), n)[:, 0]

    centroid = librosa.feature.spectral_centroid(y=other, sr=SR, hop_length=HOP)[0]
    m_spec = librosa.feature.melspectrogram(y=mix, sr=SR, hop_length=HOP, n_mels=64)
    m_idx = _bar_index(m_spec.shape[1], downbeats, bar_len)
    timbre = _per_bar(librosa.feature.mfcc(S=librosa.power_to_db(m_spec), n_mfcc=13), m_idx, n)
    chroma = _per_bar(librosa.feature.chroma_stft(S=np.abs(librosa.stft(mix, n_fft=2048, hop_length=HOP)) ** 2,
                                                  sr=SR, hop_length=HOP), m_idx, n)
    return {
        "kick": _level(kick), "hats": _level(hats), "bass": _level(power(bass)), "other": _level(power(other)),
        "density": density / (np.percentile(density, 95) + 1e-9),
        "brightness": _per_bar(centroid, _bar_index(len(centroid), downbeats, bar_len), n)[:, 0],
        "timbre": timbre, "chroma": chroma,
    }


def _jump(x: np.ndarray, window: int = 4) -> np.ndarray:
    out = np.zeros(len(x))
    for i in range(1, len(x)):
        out[i] = abs(x[i : i + window].mean() - x[max(0, i - window) : i].mean())
    return out


def stem_boundaries(f: dict[str, np.ndarray], energy: np.ndarray, vocals: np.ndarray) -> list[int]:
    """Section edges: timbre novelty plus layers switching on or off (kick and bass changes are decisive)."""
    layers = np.stack([f["kick"], f["bass"], f["hats"], f["other"], vocals], 1)
    feats = np.hstack([layers * 3, (f["timbre"] - f["timbre"].mean(0)) / (f["timbre"].std(0) + 1e-9)])
    nov = novelty((feats - feats.mean(0)) / (feats.std(0) + 1e-9))
    layer_jump = np.max([_jump(f["kick"]), _jump(f["bass"]), _jump(vocals)], axis=0)
    strength = 0.4 * nov + 0.25 * np.clip(_jump(energy) / 0.5, 0, 1) + 0.35 * np.clip(layer_jump / 0.6, 0, 1)
    return find_boundaries(strength)


def label_sections(edges: list[int], f: dict[str, np.ndarray], energy: np.ndarray) -> list[str]:
    """intro / build / drop / break / body / outro from which layers play in each section."""
    spans = list(zip(edges, edges[1:]))

    def mean(key: str, a: int, b: int) -> float:
        return float(f[key][a:b].mean())

    high = np.quantile(energy, 0.6)
    labels = []
    for a, b in spans:
        kick, e = mean("kick", a, b), float(energy[a:b].mean())
        if kick >= FULL and e >= high:
            labels.append("drop")
        elif kick < OFF:
            labels.append("break")
        else:
            labels.append("body")

    # Builds: the section right before a drop that ramps up (busier drums, brighter, or rising energy).
    for k in range(len(spans) - 1):
        if labels[k + 1] != "drop" or labels[k] == "drop":
            continue
        a, b = spans[k]
        h = (a + b) // 2
        if b - a < 2:
            continue
        rising = (f["density"][h:b].mean() > 1.2 * f["density"][a:h].mean() + 0.05
                  or f["brightness"][h:b].mean() > 1.1 * f["brightness"][a:h].mean()
                  or energy[h:b].mean() > energy[a:h].mean() + 0.1)
        if rising:
            labels[k] = "build"

    # Intro and outro: the non-drop stretch before the first full section and after the last one.
    full = [k for k, lab in enumerate(labels) if lab == "drop"]
    first, last = (full[0], full[-1]) if full else (len(labels), -1)
    for k in range(len(labels)):
        if k < first and labels[k] in ("body", "break") and all(labels[j] != "build" for j in range(k)):
            labels[k] = "intro"
        if k > last and labels[k] != "drop":
            labels[k] = "outro"
    if not full:  # no drop anywhere: still call the ends intro/outro
        labels[0], labels[-1] = "intro", "outro"
    return labels


def repeat_groups(edges: list[int], labels: list[str], f: dict[str, np.ndarray]) -> list[str | None]:
    """Group letters for sections that repeat an earlier one (same label, similar harmony and timbre)."""
    vecs = []
    for a, b in zip(edges, edges[1:]):
        v = np.concatenate([f["chroma"][a:b].mean(0), f["timbre"][a:b].mean(0) / 50])
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    groups: list[str | None] = [None] * len(vecs)
    next_letter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(len(vecs)):
        for j in range(i):
            if labels[j] == labels[i] and float(vecs[i] @ vecs[j]) >= REPEAT_SIMILARITY:
                if groups[j] is None:
                    groups[j] = next(next_letter, None)
                groups[i] = groups[j]
                break
    return groups


def _merge_repeats(edges: list[int], labels: list[str], groups: list[str | None]):
    """Back-to-back sections with the same label and group are one section (e.g. a 24-bar drop)."""
    keep = [0] + [k for k in range(1, len(labels))
                  if not (labels[k] == labels[k - 1] and groups[k] is not None and groups[k] == groups[k - 1])]
    return [edges[k] for k in keep] + [edges[-1]], [labels[k] for k in keep], [groups[k] for k in keep]


def sections_from_stems(a: Analysis, f: dict[str, np.ndarray]) -> list[Section]:
    energy = np.array(a.bar_energy)
    vocals = np.array(a.bar_vocals if a.bar_vocals is not None else np.zeros(a.n_bars))
    edges = stem_boundaries(f, energy, vocals)
    labels = label_sections(edges, f, energy)
    groups = repeat_groups(edges, labels, f)
    edges, labels, groups = _merge_repeats(edges, labels, groups)
    # A group letter only means something if the group appears more than once.
    groups = [g if g is not None and groups.count(g) > 1 else None for g in groups]
    bar_len = float(np.diff(a.downbeats).mean())
    out = []
    for (s, e), label, group in zip(zip(edges, edges[1:]), labels, groups):
        out.append(Section(label=label, start_bar=s, end_bar=e, start=a.downbeats[s],
                           end=a.downbeats[e] if e < a.n_bars else a.downbeats[-1] + bar_len,
                           energy=round(float(energy[s:e].mean()), 3), group=group))
    return out
