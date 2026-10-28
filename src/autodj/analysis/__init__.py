"""Per-track analysis: beat grid, key, loudness and phrase-aligned sections, cached by file hash."""

import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm

from autodj.analysis.beats import ALIGN_WINDOW, BackendUnavailable, beat_this_checkpoint_cached, detect_grid
from autodj.analysis.key import estimate_key
from autodj.audio import load as load_audio
from autodj.analysis.structure import bar_features, novelty, segment
from autodj.camelot import key_name, to_camelot
from autodj.models import Analysis

ANALYSIS_VERSION = 7
ANALYSIS_SR = 22050
CACHE_DIR = Path(".cache/analysis")
BACKENDS = ("beat_this", "heuristic")  # downbeat methods


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _cache_path(sha: str, backend: str, bpm_hint: float | None) -> Path:
    hint = f".bpm{bpm_hint:g}" if bpm_hint else ""
    return CACHE_DIR / f"{sha[:16]}.{backend}{hint}.v{ANALYSIS_VERSION}.json"


def _analyze(path: Path, sha: str, backend: str, bpm_hint: float | None) -> Analysis:
    stereo, native_sr = load_audio(path, sr=None, mono=False)
    stereo = np.atleast_2d(stereo)
    lufs = pyloudnorm.Meter(native_sr).integrated_loudness(stereo.T)
    y = librosa.resample(stereo.mean(axis=0), orig_sr=native_sr, target_sr=ANALYSIS_SR)
    sr = ANALYSIS_SR
    duration = len(y) / sr

    grid = detect_grid(y, sr, duration, backend, bpm_hint)
    bar_len = 4 * grid.period
    bar_starts = grid.downbeats[grid.downbeats + bar_len / 2 <= duration]

    energy, feats = bar_features(y, sr, bar_starts, bar_len)
    nov = novelty(feats)
    sections = segment(energy, nov, bar_starts, bar_len)
    pc, minor, key_conf = estimate_key(y, sr)

    # Local grid alignment per bar, from the window containing the bar's first beat.
    window = np.minimum((grid.phase + 4 * np.arange(len(bar_starts))) // ALIGN_WINDOW, len(grid.window_offsets_ms) - 1)

    return Analysis(
        version=ANALYSIS_VERSION,
        path=str(path),
        sha256=sha,
        duration=round(duration, 3),
        beat_backend=backend,
        bpm=round(grid.bpm, 4),
        beats=np.round(grid.beats, 4).tolist(),
        downbeats=np.round(bar_starts, 4).tolist(),
        onset_offset_ms=round(grid.onset_offset_ms, 1),
        grid_drift_ms=round(grid.drift_ms, 1),
        grid_ok=grid.ok,
        key=to_camelot(pc, minor),
        key_name=key_name(pc, minor),
        key_confidence=round(key_conf, 3),
        lufs=round(float(lufs), 2),
        bar_energy=np.round(energy, 3).tolist(),
        bar_novelty=np.round(nov, 3).tolist(),
        bar_offset_ms=np.round(grid.window_offsets_ms[window], 1).tolist(),
        bar_offset_confidence=np.round(grid.window_confidence[window], 2).tolist(),
        sections=sections,
    )


def analyze_file(
    path: Path, backend: str = "auto", bpm_hint: float | None = None, force: bool = False, stems: bool = False
) -> Analysis:
    """Analyze a track (cached). stems=True also separates it with Demucs and adds per-bar vocal activity."""
    a = _analyze_cached(path, backend, bpm_hint, force)
    if stems and a.sections_source != "stems":
        from autodj.analysis.stem_sections import sections_from_stems, stem_bar_features
        from autodj.stems import bar_vocal_activity, separate

        paths = separate(a.path, a.sha256)
        vocals = bar_vocal_activity(paths["vocals"], a.path, a.downbeats)
        a = a.model_copy(update={"bar_vocals": vocals})
        features = stem_bar_features({k: str(v) for k, v in paths.items()}, a.path, a.downbeats)
        a = a.model_copy(update={"sections": sections_from_stems(a, features), "sections_mix": a.sections,
                                 "sections_source": "stems"})
        a = a.model_copy(update=_stem_key(paths))
        _cache_path(a.sha256, a.beat_backend, bpm_hint).write_text(a.model_dump_json(indent=1))
    return a


def _stem_key(paths: dict) -> dict:
    """Key from the harmonic stems plus the bass line (more reliable than the full mix; see eval-keys)."""
    from autodj.analysis.key import describe, key_from_stems

    other, vocals, bass = (load_audio(paths[n], sr=ANALYSIS_SR, mono=True)[0] for n in ("other", "vocals", "bass"))
    n = min(len(other), len(vocals), len(bass))
    code, confidence = key_from_stems(other[:n] + vocals[:n], bass[:n], ANALYSIS_SR)
    return {"key": code, "key_name": describe(code), "key_confidence": confidence, "key_source": "stems"}


def _analyze_cached(path: Path, backend: str, bpm_hint: float | None, force: bool) -> Analysis:
    """Analyze a track. backend='auto' prefers beat_this downbeats (if its weights are downloaded), else heuristic.

    Pass backend='beat_this' explicitly to trigger the weights download.
    """
    path = path.expanduser().resolve()
    sha = _sha256(path)
    if backend == "auto":
        order = BACKENDS if beat_this_checkpoint_cached() else ("heuristic",)
    else:
        order = (backend,)
    for i, b in enumerate(order):
        cached = _cache_path(sha, b, bpm_hint)
        if cached.exists() and not force:
            return Analysis.model_validate_json(cached.read_text())
        try:
            result = _analyze(path, sha, b, bpm_hint)
        except BackendUnavailable as e:
            if i == len(order) - 1:
                raise
            print(f"warning: {e}; falling back to {order[i + 1]}", file=sys.stderr)
            continue
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(result.model_dump_json(indent=1))
        return result
    raise AssertionError("unreachable")
