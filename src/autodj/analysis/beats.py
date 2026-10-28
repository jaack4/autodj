"""Beat and downbeat detection, snapped to a constant-tempo grid."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import librosa
import numpy as np

BPM_RANGE = (75.0, 180.0)  # fold half/double-time detections into this range
GRID_MAX_DRIFT_MS = 25.0  # kick-vs-grid drift between first and last third


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class BeatGrid:
    period: float  # seconds per beat
    beats: np.ndarray
    downbeats: np.ndarray
    kick_coverage: float  # share of beats with a kick onset within 60 ms
    kick_offset_ms: float  # median (onset - beat); constant offsets are harmless
    drift_ms: float  # change in that offset from first to last third of the track

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    @property
    def ok(self) -> bool:
        return abs(self.drift_ms) <= GRID_MAX_DRIFT_MS


BEAT_THIS_CHECKPOINT = "final0"


def beat_this_checkpoint_cached() -> bool:
    import torch

    return (Path(torch.hub.get_dir()) / "checkpoints" / f"beat_this-{BEAT_THIS_CHECKPOINT}.ckpt").exists()


@cache
def _beat_this_model():
    from beat_this.inference import Audio2Beats

    try:
        return Audio2Beats(checkpoint_path=BEAT_THIS_CHECKPOINT, device="cpu", dbn=False)
    except ValueError as e:
        raise BackendUnavailable(f"beat_this checkpoint could not be loaded: {e}") from e


def detect_raw(y: np.ndarray, sr: int, backend: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Raw beat times, plus downbeat times if the backend predicts them."""
    if backend == "beat_this":
        beats, downbeats = _beat_this_model()(y, sr)
        return np.asarray(beats), np.asarray(downbeats)
    if backend == "librosa":
        hop = 512
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
        _, frames = librosa.beat.beat_track(
            onset_envelope=onset, sr=sr, hop_length=hop, units="frames", tightness=400
        )
        return librosa.frames_to_time(frames, sr=sr, hop_length=hop), None
    raise ValueError(f"Unknown beat backend: {backend}")


def fit_grid(raw_beats: np.ndarray, duration: float, bpm_hint: float | None = None) -> tuple[float, float, np.ndarray]:
    """Least-squares constant-tempo fit. Returns (period, first_beat, residuals_s)."""
    if len(raw_beats) < 8:
        raise ValueError("Too few beats detected to fit a grid")
    period = float(np.median(np.diff(raw_beats)))
    if bpm_hint:
        period = min((period * f for f in (0.5, 1.0, 2.0)), key=lambda p: abs(60 / p - bpm_hint))
    else:
        while 60 / period < BPM_RANGE[0]:
            period /= 2
        while 60 / period >= BPM_RANGE[1]:
            period *= 2

    # Fit on a short window first and widen it, so a slightly-off initial period
    # can't make beat indices slip far from the start of the track.
    offset = float(raw_beats[0])
    span = 16 * period
    while True:
        window = raw_beats <= raw_beats[0] + span
        for _ in range(2):
            idx = np.round((raw_beats - offset) / period)
            resid = raw_beats - (offset + idx * period)
            keep = window & (np.abs(resid) < 0.25 * period)
            period, offset = np.polyfit(idx[keep], raw_beats[keep], 1)
        if window.all():
            break
        span *= 2

    idx = np.round((raw_beats - offset) / period)
    resid = raw_beats - (offset + idx * period)
    first = offset % period
    return float(period), float(first), resid[keep]


def downbeat_phase_from_model(grid_beats: np.ndarray, raw_downbeats: np.ndarray, period: float) -> int:
    idx = np.round((raw_downbeats - grid_beats[0]) / period).astype(int)
    return int(np.bincount(idx % 4, minlength=4).argmax())


def downbeat_phase_heuristic(y: np.ndarray, sr: int, grid_beats: np.ndarray) -> int:
    """Guess which beat (0..3) starts the bar: timbre changes and bass hits cluster on downbeats."""
    hop = 512
    frames = librosa.time_to_frames(grid_beats, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    beat_mfcc = librosa.util.sync(mfcc, frames, aggregate=np.mean)[:, 1 : len(frames)]
    change = np.r_[0, np.linalg.norm(np.diff(beat_mfcc, axis=1), axis=0)]
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=150, n_mels=32)
    low_at = low[np.clip(frames, 0, len(low) - 1)][: len(change)]

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    score = z(change) + 0.5 * z(low_at)
    return int(np.argmax([score[p::4].mean() for p in range(4)]))


def kick_alignment(y: np.ndarray, sr: int, beats: np.ndarray) -> tuple[float, float, float]:
    """Compare a beat grid to low-frequency onsets: (coverage, median offset ms, drift ms)."""
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=32)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="time")
    if len(peaks) < 2:
        return 0.0, 0.0, 0.0
    j = np.searchsorted(peaks, beats).clip(1, len(peaks) - 1)
    nearest = np.where(np.abs(peaks[j] - beats) < np.abs(peaks[j - 1] - beats), peaks[j], peaks[j - 1])
    offset_ms = (nearest - beats) * 1000
    hit = np.abs(offset_ms) < 60
    hits = offset_ms[hit]
    if len(hits) < 9:
        return float(hit.mean()), 0.0, 0.0
    third = len(hits) // 3
    drift = np.median(hits[-third:]) - np.median(hits[:third])
    return float(hit.mean()), float(np.median(hits)), float(drift)


def detect_grid(y: np.ndarray, sr: int, duration: float, backend: str, bpm_hint: float | None = None) -> BeatGrid:
    raw_beats, raw_downbeats = detect_raw(y, sr, backend)
    period, first, _ = fit_grid(raw_beats, duration, bpm_hint)
    beats = first + period * np.arange(int((duration - first) / period) + 1)
    if raw_downbeats is not None and len(raw_downbeats) >= 4:
        phase = downbeat_phase_from_model(beats, raw_downbeats, period)
    else:
        phase = downbeat_phase_heuristic(y, sr, beats)
    coverage, offset_ms, drift_ms = kick_alignment(y, sr, beats)
    return BeatGrid(
        period=period,
        beats=beats,
        downbeats=beats[phase::4],
        kick_coverage=coverage,
        kick_offset_ms=offset_ms,
        drift_ms=drift_ms,
    )
