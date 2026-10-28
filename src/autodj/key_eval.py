"""Compare key-detection methods against reference keys (`autodj eval-keys`)."""

import json
from pathlib import Path

import numpy as np

from autodj.analysis import key as keys
from autodj.audio import load as load_audio
from autodj.camelot import camelot_distance, parse_key
from autodj.models import Analysis

CACHE_DIR = Path(".cache/keys")
METHODS = ("mix", "stems", "essentia", "essentia-stems")


def _tonic_and_mode(code: str) -> tuple[int, bool]:
    minor = code.endswith("A")
    return next(pc for pc in range(12) if keys.to_camelot(pc, minor) == code), minor


def mirex_score(reference: str, estimate: str) -> float:
    """Standard weighted key score: exact 1, fifth 0.5, relative 0.3, parallel 0.2."""
    ref, est = parse_key(reference), parse_key(estimate)
    if ref == est:
        return 1.0
    (rp, rm), (ep, em) = _tonic_and_mode(ref), _tonic_and_mode(est)
    if rm == em and (ep - rp) % 12 in (5, 7):
        return 0.5
    if rm != em and ref[:-1] == est[:-1]:
        return 0.3
    if rp == ep:
        return 0.2
    return 0.0


def detect(a: Analysis, method: str) -> tuple[str, float]:
    """Run one method on a track (cached by file hash)."""
    cache = CACHE_DIR / f"{a.sha256[:16]}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    if method in done:
        return tuple(done[method])
    if method == "mix":
        y, sr = load_audio(a.path, sr=22050, mono=True)
        result = keys.key_from_mix(y, sr)
    else:
        from autodj.stems import separate

        if method == "essentia":
            result = keys.key_from_essentia(load_audio(a.path, sr=44100, mono=True)[0])
        else:
            stems = separate(a.path, a.sha256)
            sr = 44100 if method == "essentia-stems" else 22050
            other, vocals, bass = (load_audio(stems[n], sr=sr, mono=True)[0] for n in ("other", "vocals", "bass"))
            n = min(len(other), len(vocals), len(bass))
            harmonic = other[:n] + vocals[:n]
            if method == "stems":
                result = keys.key_from_stems(harmonic, bass[:n], sr)
            else:
                result = keys.key_from_essentia(harmonic + 0.5 * bass[:n])
    done[method] = list(result)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(done))
    return result


def summarize(rows: list[dict], method: str) -> dict:
    est = [r[method][0] for r in rows]
    return {
        "exact": float(np.mean([e == r["key"] for e, r in zip(est, rows)])),
        "any_source": float(np.mean([e == r["key"] or e in r["alt"] for e, r in zip(est, rows)])),
        "compatible": float(np.mean([camelot_distance(e, r["key"]) <= 1 for e, r in zip(est, rows)])),
        "mirex": float(np.mean([mirex_score(r["key"], e) for e, r in zip(est, rows)])),
    }
