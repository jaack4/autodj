# M0: Analysis spike — findings (2026-10-27)

**Status:** partially complete. The analysis pipeline works end to end. Two
things are still open: checking downbeats by ear, and evaluating `beat_this`
(its model download is currently broken upstream).

## What was built
`autodj analyze <files>` → cached JSON in `.cache/analysis/`, plus a report in `output/<track>/`:
- `clicks.wav`: the track with clicks on beats (high), bars (low), and a tone at each section start. **Listen to this to check the analysis.**
- `analysis.png`: RMS envelope, bar lines, colored sections, and per-bar energy and novelty.

Pipeline: load → beat detection (`beat_this` or `librosa`) → **constant-tempo
grid fit** → downbeat phase → per-bar features → novelty → **phrase-aware
segmentation** → section labels → key (Krumhansl profiles) → loudness (LUFS).

## Test tracks
| Track | BPM | Key | Grid drift | Kicks on beats | Sections |
|---|---|---|---|---|---|
| ISOxo – how2fly (242 s) | 140.01 | 6B / Bb major (low confidence 0.07) | −0.3 ms | 87% | Drops found at 29 s, 84 s and 183 s, matching the energy curve |
| Knock2 – dashstar (VIP) (141 s) | 122.41 | 9A / E minor (confidence 0.17) | +19.8 ms | 39% | Main boundaries right; the last 30 s are under-segmented |

Only 2 usable tracks so far. **Need 3–5 more**, ideally with a wider range of genres and tempos.

## Findings
1. **Constant-tempo grid fitting works well for electronic music.** Even with a
   jittery tracker (librosa), a least-squares grid lands within ~20 ms over the
   whole track. Fitting on a short window and then widening it was necessary:
   a single global fit drifted by a whole beat when beats were missing (caught
   by a unit test).
2. **The tracker's own jitter is the wrong quality metric.** The first version
   flagged both tracks as "not constant tempo" just because librosa's raw beats
   wobble. The replacement measures **kick-onset vs grid drift** between the
   first and last third of the track. That catches a wrong BPM or tempo
   changes, and ignores constant offsets.
3. **A fixed phrase grid is wrong for real tracks.** ISOxo's last drop arrives
   2 bars off the 8-bar grid. Real tracks have extensions, gaps and edits.
   Replaced with **dynamic programming over boundaries**: it prefers 8/16/32-bar
   sections, allows odd lengths only for strong boundaries, and penalizes
   1–3-bar slivers.
4. **Downbeats are the weak point without a trained model.** The librosa
   fallback guesses which beat starts the bar from timbre changes and bass
   onsets. Two automatic checks (where level jumps land; which phase gives the
   sharpest segmentation) gave inconclusive or contradictory answers. If the
   downbeat is wrong, every transition lands half a bar off, so **this is the
   biggest remaining risk.**
5. **Key detection confidence varies a lot** (0.07 vs 0.17). A low-confidence
   key should count for less in compatibility scoring. User-provided or
   djay-analyzed keys should override it.
6. **Speed:** about 30–60 s per track on an M5 (CPU), cached afterwards. Most
   of the time goes to harmonic separation and CQT chroma for key detection,
   which can be optimized later.

## Blockers and environment notes
- **`beat_this` weights can't be downloaded.** The official host
  (`cloud.cp.jku.at`) is rate-limiting every request, and upstream issue #23
  (opened 2026-10-27) confirms downloads are broken. Third-party Hugging Face
  mirrors exist but are unverified, so they weren't used. Once
  `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` exists, `--backend auto`
  uses it automatically. To trigger the download explicitly:
  `autodj analyze --backend beat_this <file>`.
- **`allin1` was not attempted.** It depends on `madmom` 0.16.1 (from 2018,
  doesn't build on Python 3.12) and NATTEN (no macOS wheels). The DP
  segmentation above covers the same need for now.
- **Disk:** about 4.8 GB free; the environment uses about 0.9 GB (torch).
  Stem separation (Demucs) is deferred to M4.

## Exit criteria for M0
- [ ] Listen to `clicks.wav` for each track: are the low clicks on the "1" of each bar? Are section tones at real changes?
- [ ] Compare BPM and key with djay's analysis of the same tracks.
- [ ] Analyze 3–5 more tracks.
- [ ] Get `beat_this` weights and compare its downbeats with the librosa fallback (if the host recovers).
