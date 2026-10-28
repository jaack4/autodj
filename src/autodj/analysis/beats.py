"""Constant-tempo beat grid and downbeats.

Tempo and beat phase come from a comb search over the whole track's kick-band onsets,
which is more reliable for dance music than any beat tracker's local decisions (both
librosa and beat_this got a 126 BPM track wrong by 2-3%). The tracker backend is only
used to decide which beat starts the bar.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import librosa
import numpy as np

BPM_RANGE = (75.0, 180.0)
DOUBLE_TIME_RATIO = 0.8  # prefer 2x tempo if its comb score is at least this close
GRID_MAX_DRIFT_MS = 25.0  # onset-vs-grid drift between first and last third
ALIGN_WINDOW = 16  # beats per local alignment measurement
MIN_CONFIDENCE = 1.5  # windows below this have too few kicks to measure


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class BeatGrid:
    period: float  # seconds per beat
    beats: np.ndarray
    downbeats: np.ndarray
    phase: int  # index of the first downbeat in `beats`
    onset_offset_ms: float  # median (onsets - grid); constant offsets are harmless
    drift_ms: float  # change in that offset from first to last third of the track
    window_offsets_ms: np.ndarray  # local offset per ALIGN_WINDOW beats
    window_confidence: np.ndarray  # peak/mean of each window's alignment score (~1 = no kicks)

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


def _onset_env(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=32)
    return env - env.mean()


def _comb_scores(env: np.ndarray, fps: float, duration: float, bpms: np.ndarray, phase_step: float):
    """For each tempo: best mean onset strength over grid phases, and that phase."""
    scores, phases = np.empty(len(bpms)), np.empty(len(bpms))
    for i, bpm in enumerate(bpms):
        period = 60 / bpm
        ph = np.arange(0, period, phase_step)
        t = ph[:, None] + period * np.arange(int(duration / period) - 1)[None, :]
        s = env[np.clip((t * fps).astype(int), 0, len(env) - 1)].mean(axis=1)
        j = int(s.argmax())
        scores[i], phases[i] = s[j], ph[j]
    return scores, phases


def comb_grid(y: np.ndarray, sr: int, duration: float, bpm_hint: float | None = None) -> tuple[float, float]:
    """Constant (period, first_beat) that best lines up with kick-band onsets over the whole track."""
    hop = 64
    fps = sr / hop
    env = _onset_env(y, sr, hop)
    lo, hi = (bpm_hint * 0.97, bpm_hint * 1.03) if bpm_hint else BPM_RANGE

    # Coarse pass on a smoothed envelope (wider peaks), then refine around the winner.
    smooth = np.convolve(env, np.hanning(9) / np.hanning(9).sum(), mode="same")
    coarse = np.arange(lo, hi, 0.05)
    scores, _ = _comb_scores(smooth, fps, duration, coarse, 0.008)
    bpm = coarse[scores.argmax()]

    def refine(center: float) -> tuple[float, float, float]:
        fine = np.arange(center - 0.1, center + 0.1, 0.002)
        sc, ph = _comb_scores(env, fps, duration, fine, 0.001)
        k = int(sc.argmax())
        return float(fine[k]), float(ph[k]), float(sc[k])

    bpm, phase, score = refine(bpm)
    if not bpm_hint and bpm * 2 < hi:
        bpm2, phase2, score2 = refine(bpm * 2)
        if score2 >= DOUBLE_TIME_RATIO * score:
            bpm, phase = bpm2, phase2
    return 60 / bpm, phase


def beat_this_downbeats(y: np.ndarray, sr: int) -> np.ndarray:
    _, downbeats = _beat_this_model()(y, sr)
    return np.asarray(downbeats)


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


def _kick_env(y: np.ndarray, sr: int, hop: int = 128) -> tuple[np.ndarray, float]:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=32), sr / hop


def _best_shift(env: np.ndarray, fps: float, beats: np.ndarray, period: float, center: float = 0.0) -> tuple[float, float]:
    """Shift within +-1/4 beat of `center` that best lines `beats` up with onsets: (shift s, peak/mean score).

    +-1/4 beat keeps off-beat hi-hats (exactly 1/2 beat away) from capturing the alignment.
    """
    shifts = np.linspace(-0.25, 0.25, 51) * period
    frames = np.round((beats[None, :] + center + shifts[:, None]) * fps).astype(int)
    score = env[np.clip(frames, 0, len(env) - 1)].sum(axis=1)
    k = int(score.argmax())
    return center + float(shifts[k]), float(score[k] / (score.mean() + 1e-9))


def local_alignment(
    y: np.ndarray, sr: int, beats: np.ndarray, period: float, window: int = ALIGN_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Kick-band offset (s) and confidence of each `window`-beat stretch of the grid.

    Each window searches near the last confident window's offset, so a grid that slides by
    whole beats accumulates offset instead of silently matching the wrong onsets, and
    sparse sections (no kicks) can't drag the tracking off course.
    """
    env, fps = _kick_env(y, sr)
    offsets, confidence = [], []
    center = 0.0
    for start in range(0, len(beats) - window + 1, window):
        shift, conf = _best_shift(env, fps, beats[start : start + window], period, center)
        offsets.append(shift)
        confidence.append(conf)
        if conf >= MIN_CONFIDENCE:
            center = shift
    return np.array(offsets), np.array(confidence)


def grid_alignment(offsets: np.ndarray, confidence: np.ndarray) -> tuple[float, float]:
    """(median offset ms, drift ms between first and last third) over confident windows."""
    good = offsets[confidence >= MIN_CONFIDENCE]
    if len(good) < 3:
        return 0.0, 0.0
    third = max(1, len(good) // 3)
    drift = np.median(good[-third:]) - np.median(good[:third])
    return float(np.median(good) * 1000), float(drift * 1000)


def onset_offset(
    y: np.ndarray, sr: int, beats: np.ndarray, period: float, center_ms: float = 0.0
) -> tuple[float, float]:
    """Kick-band offset (ms) of a stretch of audio vs a grid, searched within +-1/4 beat of
    center_ms, and its confidence (peak/mean, ~1 = no kicks)."""
    env, fps = _kick_env(y, sr)
    shift, confidence = _best_shift(env, fps, beats, period, center_ms / 1000)
    return shift * 1000, confidence


def detect_grid(y: np.ndarray, sr: int, duration: float, backend: str, bpm_hint: float | None = None) -> BeatGrid:
    """backend picks the downbeat method: 'beat_this' (model) or 'heuristic'."""
    period, first = comb_grid(y, sr, duration, bpm_hint)
    beats = first + period * np.arange(int((duration - first) / period) + 1)
    if backend == "beat_this":
        phase = downbeat_phase_from_model(beats, beat_this_downbeats(y, sr), period)
    elif backend == "heuristic":
        phase = downbeat_phase_heuristic(y, sr, beats)
    else:
        raise ValueError(f"Unknown downbeat backend: {backend}")
    offsets, confidence = local_alignment(y, sr, beats, period)
    offset_ms, drift_ms = grid_alignment(offsets, confidence)
    return BeatGrid(
        period=period,
        beats=beats,
        downbeats=beats[phase::4],
        phase=phase,
        onset_offset_ms=offset_ms,
        drift_ms=drift_ms,
        window_offsets_ms=offsets * 1000,
        window_confidence=confidence,
    )
