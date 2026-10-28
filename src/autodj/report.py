"""Human checks for an analysis: a click-track overlay to listen to and a plot to look at."""

import re
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from autodj.models import Analysis  # noqa: E402

OUTPUT_DIR = Path("output")
LISTEN_SR = 44100
SECTION_COLORS = {
    "intro": "#9db4c0",
    "build": "#f2c14e",
    "drop": "#e4572e",
    "break": "#76b041",
    "body": "#c9cba3",
    "outro": "#9db4c0",
}


def report_dir(a: Analysis) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(a.path).stem).strip("-").lower()
    d = OUTPUT_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_clicks(a: Analysis, out: Path) -> Path:
    """Track at reduced level + soft clicks on beats, louder on downbeats, a tone at section starts."""
    y, _ = librosa.load(a.path, sr=LISTEN_SR, mono=True)
    n = len(y)
    beats = librosa.clicks(times=a.beats, sr=LISTEN_SR, click_freq=2000, click_duration=0.03, length=n)
    bars = librosa.clicks(times=a.downbeats, sr=LISTEN_SR, click_freq=1000, click_duration=0.06, length=n)
    starts = [s.start for s in a.sections]
    sections = librosa.clicks(times=starts, sr=LISTEN_SR, click_freq=440, click_duration=0.4, length=n)
    mix = 0.6 * y + 0.15 * beats + 0.35 * bars + 0.5 * sections
    mix /= max(1.0, np.abs(mix).max())
    path = out / "clicks.wav"
    sf.write(path, mix, LISTEN_SR)
    return path


def write_plot(a: Analysis, out: Path) -> Path:
    y, sr = librosa.load(a.path, sr=11025, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True, height_ratios=[2, 1])
    for s in a.sections:
        ax1.axvspan(s.start, s.end, color=SECTION_COLORS.get(s.label, "#ddd"), alpha=0.45, lw=0)
        ax1.text(s.start + 0.5, 0.97, f"{s.label}\n{s.end_bar - s.start_bar}b", va="top", fontsize=8,
                 transform=ax1.get_xaxis_transform())
    ax1.fill_between(t, rms, color="#333", lw=0)
    for d in a.downbeats:
        ax1.axvline(d, color="#888", lw=0.3, alpha=0.6)
    for s in a.sections:
        ax1.axvline(s.start, color="#000", lw=0.8)
    ax1.set_ylabel("RMS")
    ax1.set_title(f"{Path(a.path).name} — {a.bpm} BPM, {a.key} ({a.key_name}), {a.lufs} LUFS, "
                  f"grid drift {a.grid_drift_ms:+} ms [{a.beat_backend}]")

    bar_t = np.array(a.downbeats)
    ax2.step(bar_t, a.bar_energy, where="post", label="energy")
    ax2.step(bar_t, a.bar_novelty, where="post", label="novelty", alpha=0.7)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("seconds")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "analysis.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
