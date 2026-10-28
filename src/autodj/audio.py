"""Audio loading: libsndfile (via librosa) first, ffmpeg for everything else (AAC/M4A, ALAC, ...)."""

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np


class AudioLoadError(RuntimeError):
    pass


def _ffmpeg_decode(path: str | Path, sr: int | None, channels: int | None) -> tuple[np.ndarray, int]:
    if not shutil.which("ffmpeg"):
        raise AudioLoadError(f"Can't decode {path}: install ffmpeg for this format")
    if sr is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        try:
            native_sr, native_ch = (int(x) for x in probe.stdout.strip().split(",")[:2])
        except ValueError as e:
            raise AudioLoadError(f"Not a readable audio file: {path}") from e
        sr, channels = native_sr, channels or native_ch
    channels = channels or 2
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise AudioLoadError(f"Not a readable audio file: {path} ({result.stderr.decode().strip()[:200]})")
    audio = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, channels).T
    return audio, sr


def load(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Like librosa.load (sr=None keeps the native rate), but also decodes formats libsndfile can't."""
    try:
        return librosa.load(path, sr=sr, mono=mono)
    except Exception:
        audio, rate = _ffmpeg_decode(path, sr, 1 if mono else None)
        return (audio[0] if mono else audio), rate
