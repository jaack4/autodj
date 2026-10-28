# M0: Analysis spike — findings (2026-10-27)

**Status:** partially complete. The analysis pipeline works end to end. Two
things are still open: checking downbeats by ear, and evaluating `beat_this`
(its model download is currently broken upstream).

## What was built
`autodj analyze <files>` → cached JSON in `.cache/analysis/`, plus a report in `output/<track>/`:
- `clicks.wav`: the track with clicks on beats (high), bars (low), and a tone at each section start. **Listen to this to check the analysis.**
- `analysis.png`: RMS envelope, bar lines, colored sections, and per-bar energy and novelty.

Pipeline: load → **comb search for tempo and beat phase** → downbeat phase
(`beat_this` model or heuristic) → per-bar features → novelty →
**phrase-aware segmentation** → section labels → key (Krumhansl profiles) →
loudness (LUFS) → grid drift check.

## Test tracks
| Track | BPM | Key | Verdict |
|---|---|---|---|
| ISOxo – how2fly (242 s) | 140.01 | 6B / Bb major (low confidence 0.07) | **Verified by ear**: beats, bars and section tones all correct |
| Knock2 – dashstar (VIP) (141 s) | 126.03 (was 122.41, wrong) | 9A / E minor (confidence 0.17) | The first version's tones were off by ear because of the wrong tempo. Fixed; needs a second listen. The file contains a splice (see below) |

Only 2 usable tracks so far. **Need 3–5 more**, ideally with a wider range of genres and tempos.

## Findings
1. **Beat trackers get tempo wrong; a full-track comb search doesn't.** On
   dashstar, librosa said 122.41 and `beat_this` said 123.31. A brute-force
   search found 126.00 with an unambiguous peak (score 0.65 vs 0.10 for the
   runner-up). `beat_this` loses the beat in the syncopated intro (local tempo
   jumps to 136–167 BPM). A grid fitted to tracker beats inherits that error,
   and a 2–3% tempo error slides the grid by 2 beats over 2 minutes.
   **Tempo and beat phase now come from the comb search**: the steady grid that
   best lines up with kick-band onsets over the whole track, with a tie-break
   toward double time. Trackers are only used for downbeats.
2. **Drift checks must track, not just match.** A check that matched each beat
   to the nearest kick within 60 ms passed a grid that had slid 2 beats,
   because it matched the wrong kicks. A ±½-beat search got captured by
   off-beat hi-hats. The working version follows the best kick-band offset
   window by window (±¼ beat from the previous window) and reports the drift
   from the first third to the last.
3. **Files can contain splices.** dashstar (VIP) lines up to within ~7 ms
   from 61 to 106 s, then everything after the silence at 108 s is
   consistently ~100 ms late. That's an edit that isn't a whole number of
   beats. The drift check flags it (`grid_ok: false`). Proper release files
   rarely have this, but rips and edits do. **Later:** piecewise grids that
   re-anchor at discontinuities, or refuse to mix through them.
4. **A fixed phrase grid is wrong for real tracks.** ISOxo's last drop arrives
   2 bars off the 8-bar grid. Replaced with **dynamic programming over
   boundaries**: it prefers 8/16/32-bar sections, allows odd lengths only for
   strong boundaries, and penalizes 1–3-bar slivers. On spliced files the
   odd-length penalty can merge real sections (dashstar's final drop is
   merged into the break).
5. **Downbeats:** on both tracks, `beat_this` and the heuristic agree once the
   grid is correct. With a correct grid, the heuristic's earlier "inconclusive"
   tests were really symptoms of the tempo error.
6. **Key detection confidence varies a lot** (0.07 vs 0.17). A low-confidence
   key should count for less in compatibility scoring. User-provided or
   djay-analyzed keys should override it.
7. **Speed:** about 10–20 s per track on an M5 (CPU), cached afterwards.

## `beat_this` weights
The official host (`cloud.cp.jku.at`) is down: it's rate-limiting all requests
(upstream issue #23, 2026-10-27). We use the community copy
[`safe-models/beat-this-final0`](https://huggingface.co/safe-models/beat-this-final0)
(safetensors, so it can't run code when loaded), converted with
`scripts/import_beat_this_weights.py`. That script checks the sha256, loads
the weights strictly into `BeatThis`, verifies the shared rotary-embedding
tensor matches the model's own computed values, and writes
`~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt`.
**Validation:** on how2fly (verified by ear) it gives the same tempo, the same
downbeat phase and the same sections (within 0.1 s) as the analysis we
checked by ear. 20.3M parameters, matching the published model.

## Ruled out for now
- **`allin1`:** depends on `madmom` 0.16.1 (from 2018, doesn't build on Python
  3.12) and NATTEN (no macOS wheels). The DP segmentation covers the same need.
- **Demucs stems:** deferred to M4 (disk space: ~4.7 GB free).

## Exit criteria for M0
- [x] Get `beat_this` weights and compare its downbeats with the heuristic.
- [x] Re-listen to dashstar `clicks.wav` (126.03 BPM grid): beats confirmed good.
- [ ] Compare BPM and key with djay's analysis of the same tracks.
- [ ] Analyze 3–5 more tracks.
