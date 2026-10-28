"""Source separation (Demucs) and per-bar vocal activity, cached by file hash."""

from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf

from autodj.audio import load as load_audio

STEMS_DIR = Path(".cache/stems")
MODEL = "htdemucs"
STEM_NAMES = ("drums", "bass", "other", "vocals")
# Vocal stem level relative to the full mix, per bar. Sung/rapped sections measure about
# -3..-8 dB; drops with vocal chops or separation bleed about -15..-25 dB.
VOCAL_FULL_DB = -8.0
VOCAL_NONE_DB = -18.0
SILENT_BELOW_DB = 30.0  # bars this far under the track's loud bars are silence, not vocals


def stem_dir(sha: str) -> Path:
    return STEMS_DIR / f"{sha[:16]}.{MODEL}"


def stem_path(sha: str, name: str) -> Path:
    return stem_dir(sha) / f"{name}.flac"


def has_stems(sha: str) -> bool:
    return all(stem_path(sha, n).exists() for n in STEM_NAMES)


@cache
def _model():
    from demucs.pretrained import get_model

    return get_model(MODEL)


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def separate(path: str, sha: str) -> dict[str, Path]:
    """Split a track into drums/bass/other/vocals FLACs (cached)."""
    if has_stems(sha):
        return {n: stem_path(sha, n) for n in STEM_NAMES}
    import torch
    from demucs.apply import apply_model

    model = _model()
    audio, _ = load_audio(path, sr=model.samplerate, mono=False)
    audio = np.atleast_2d(audio)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    with torch.no_grad():
        out = apply_model(model, torch.tensor(audio, dtype=torch.float32)[None], device=_device(),
                          shifts=0, split=True, overlap=0.25, progress=False)[0].cpu().numpy()
    stem_dir(sha).mkdir(parents=True, exist_ok=True)
    for name, stem in zip(model.sources, out):
        sf.write(stem_path(sha, name), stem.T, model.samplerate, subtype="PCM_16")
    return {n: stem_path(sha, n) for n in STEM_NAMES}


def _bar_power_db(path: str | Path, downbeats: list[float]) -> np.ndarray:
    y, sr = load_audio(path, sr=22050, mono=True)
    bar_len = float(np.diff(downbeats).mean())
    edges = np.r_[downbeats, downbeats[-1] + bar_len]
    return np.array([10 * np.log10(np.mean(y[int(a * sr) : int(b * sr)] ** 2) + 1e-12) for a, b in zip(edges, edges[1:])])


def bar_vocal_activity(vocals: Path, mix: str, downbeats: list[float]) -> list[float]:
    """0..1 per bar: how prominent the vocal stem is within the mix."""
    voc, full = _bar_power_db(vocals, downbeats), _bar_power_db(mix, downbeats)
    activity = np.clip((voc - full - VOCAL_NONE_DB) / (VOCAL_FULL_DB - VOCAL_NONE_DB), 0, 1)
    activity[full < np.percentile(full, 95) - SILENT_BELOW_DB] = 0.0
    return np.round(activity, 3).tolist()
