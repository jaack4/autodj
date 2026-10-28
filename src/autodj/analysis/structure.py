"""Bar-level features and phrase-aligned section segmentation."""

import librosa
import numpy as np

from autodj.models import Section

HOP = 512


def _per_bar(values: np.ndarray, frame_times: np.ndarray, bar_starts: np.ndarray, end: float) -> np.ndarray:
    """Average (dims, frames) values into (bars, dims)."""
    idx = np.searchsorted(bar_starts, frame_times, side="right") - 1
    valid = (idx >= 0) & (frame_times < end)
    counts = np.maximum(np.bincount(idx[valid], minlength=len(bar_starts)), 1)
    out = np.stack(
        [np.bincount(idx[valid], weights=row[valid], minlength=len(bar_starts)) for row in values], axis=1
    )
    return out / counts[:, None]


def _norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, [5, 95])
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)


def bar_features(y: np.ndarray, sr: int, bar_starts: np.ndarray, bar_len: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (energy 0..1 per bar, standardized timbre/harmony features per bar)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64)
    S_db = librosa.power_to_db(S)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    mel_freqs = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)

    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=HOP))
    low_db = librosa.power_to_db(S[mel_freqs < 150].sum(axis=0, keepdims=True))
    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    n = min(S.shape[1], rms_db.shape[1], chroma.shape[1])

    end = bar_starts[-1] + bar_len
    stacked = np.vstack([rms_db[:, :n], low_db[:, :n], mfcc[:, :n], chroma[:, :n]])
    per_bar = _per_bar(stacked, times[:n], bar_starts, end)

    energy = 0.5 * _norm01(per_bar[:, 0]) + 0.5 * _norm01(per_bar[:, 1])
    feats = (per_bar - per_bar.mean(axis=0)) / (per_bar.std(axis=0) + 1e-9)
    return energy, feats


def novelty(feats: np.ndarray, half: int = 4) -> np.ndarray:
    """Foote checkerboard novelty; value at bar i means 'a new section starts at bar i'."""
    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    S = X @ X.T
    coords = np.arange(-half, half) + 0.5
    taper = np.exp(-((coords / half) ** 2) * 2)
    w = np.sign(coords) * taper
    K = np.outer(w, w)
    Sp = np.pad(S, half, mode="edge")
    nov = np.array([np.sum(K * Sp[i : i + 2 * half, i : i + 2 * half]) for i in range(len(S))])
    nov = np.clip(nov, 0, None)
    return nov / (nov.max() + 1e-9)


SECTION_COST = 0.3  # a boundary must be at least this strong to be worth adding
MAX_SECTION_BARS = 32


def _length_penalty(bars: int) -> float:
    """Dance music phrases come in 8s; allow 4s, and odd lengths only for strong boundaries."""
    if bars % 8 == 0:
        return 0.0
    if bars % 4 == 0:
        return 0.15
    if bars < 4:
        return 1.0  # fills and gaps belong to the neighboring section
    return 0.5


def boundary_strength(energy: np.ndarray, nov: np.ndarray, window: int = 4) -> np.ndarray:
    n = len(energy)
    jump = np.zeros(n)
    for i in range(1, n):
        jump[i] = abs(energy[i : i + window].mean() - energy[max(0, i - window) : i].mean())
    return 0.6 * nov + 0.4 * np.clip(jump / 0.5, 0, 1)


def find_boundaries(strength: np.ndarray) -> list[int]:
    """Dynamic programming over section boundaries, preferring phrase-length sections."""
    n = len(strength)
    best = np.full(n + 1, -np.inf)
    prev = np.zeros(n + 1, int)
    best[0] = 0.0
    for j in range(1, n + 1):
        gain = strength[j] if j < n else 0.0
        for i in range(max(0, j - MAX_SECTION_BARS), j):
            score = best[i] + gain - SECTION_COST - _length_penalty(j - i)
            if j == n or i == 0:
                score += _length_penalty(j - i) * 0.5  # partial intro/outro phrases are common
            if score > best[j]:
                best[j], prev[j] = score, i
    edges = [n]
    while edges[-1] > 0:
        edges.append(prev[edges[-1]])
    return edges[::-1]


def segment(energy: np.ndarray, nov: np.ndarray, bar_starts: np.ndarray, bar_len: float) -> list[Section]:
    """Split a track into labeled sections with phrase-aligned boundaries."""
    n = len(energy)
    edges = find_boundaries(boundary_strength(energy, nov))

    hi, lo = np.quantile(energy, [0.66, 0.33])
    sections: list[Section] = []
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        e = float(energy[a:b].mean())
        if k == 0:
            label = "intro"
        elif k == len(edges) - 2:
            label = "outro"
        elif e >= hi:
            label = "drop"
        elif e <= lo:
            label = "break"
        else:
            label = "body"
        sections.append(
            Section(
                label=label,
                start_bar=a,
                end_bar=b,
                start=float(bar_starts[a]),
                end=float(bar_starts[b] if b < n else bar_starts[-1] + bar_len),
                energy=round(e, 3),
            )
        )

    # A non-drop section that ramps up straight into a drop is a build.
    for prev, nxt in zip(sections, sections[1:]):
        if nxt.label == "drop" and prev.label in ("body", "break"):
            half = (prev.start_bar + prev.end_bar) // 2
            if energy[half : prev.end_bar].mean() > energy[prev.start_bar : half].mean():
                prev.label = "build"
    return sections
