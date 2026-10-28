"""Per-bar sonic character on scales that compare *across* tracks (unlike bar_energy, which is
relative within a track): loudness relative to the track's integrated LUFS, grit (spectral
flatness at 2-10 kHz: smooth/tonal ~0.05, distorted/noisy ~0.4), and drums level from the stem."""

import librosa
import numpy as np

HOP = 512


def _per_bar_median(values: np.ndarray, times: np.ndarray, downbeats: list[float]) -> np.ndarray:
    bar_len = float(np.diff(downbeats).mean())
    edges = np.r_[downbeats, downbeats[-1] + bar_len]
    idx = np.searchsorted(edges, times, side="right") - 1
    return np.array([float(np.median(values[idx == k])) if np.any(idx == k) else 0.0 for k in range(len(downbeats))])


def _bar_level_db(y: np.ndarray, sr: int, downbeats: list[float], lufs: float) -> np.ndarray:
    bar_len = float(np.diff(downbeats).mean())
    edges = np.r_[downbeats, downbeats[-1] + bar_len]
    power = np.array([np.mean(y[int(a * sr) : int(b * sr)] ** 2) + 1e-12 for a, b in zip(edges, edges[1:])])
    return 10 * np.log10(power) - lufs


def bar_character(y: np.ndarray, sr: int, downbeats: list[float], lufs: float) -> tuple[list[float], list[float]]:
    """(loudness dB relative to the track's LUFS, grit 0..1) per bar."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = S[(freqs > 2000) & (freqs < 10000)]
    flatness = np.exp(np.mean(np.log(band + 1e-12), axis=0)) / (np.mean(band, axis=0) + 1e-12)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    grit = _per_bar_median(flatness, times, downbeats)
    loud = _bar_level_db(y, sr, downbeats, lufs)
    return np.round(loud, 1).tolist(), np.round(grit, 3).tolist()


def bar_drums_db(drums: np.ndarray, sr: int, downbeats: list[float], lufs: float) -> list[float]:
    return np.round(_bar_level_db(drums, sr, downbeats, lufs), 1).tolist()
