# M2–M4: Planner, sets, transition library — findings (2026-10-27)

**Status:** built and unit-tested (58 tests). Only two real tracks so far, and
their keys clash (6B vs 9A), so nothing has been judged by ear on a
key-compatible playlist yet.

## M2: Choosing transition points (`planner.py`)
- **Candidates:** out-points on A's 4-bar phrase lines (from a third of the way
  in), in-points at B's section starts (within its first 40%), times allowed
  overlap lengths. Clashing keys only get short overlaps.
- **Scoring**, each term recorded as a human-readable reason:
  - how much of A gets played, and how much of B is skipped
  - drops buried under a blend, or A's later drops cut off
  - "mud" (both tracks loud at once)
  - leaving A, or entering B, on a section boundary, and landing on B's drop
  - key distance against blend length
  - grid jumps inside the blend
  - vocals in both tracks at once (with stems)
- **Splices:** per-bar grid offsets are smoothed with a 12-bar rolling median
  over confident bars. Syncopated outliers vanish; real splices stay as steps.
  Blends across a step are heavily penalized.

## M3: Sets (`setlist.py`, `autodj set`)
- **Input:** a playlist YAML (`examples/playlist.example.yaml`), a folder of
  audio files, or `pair A B`. User-given keys override detection; user-given
  BPMs resolve half/double time.
- **Ordering:** pair cost = key cost (scaled down for uncertain keys) + tempo
  gap² + an arc penalty (`warmup_peak` prefers rising tempo) − transition
  quality. Exact Held-Karp up to 12 tracks, greedy + 2-opt above.
  20 tracks plan in ~2.4 s.
- **Per-track tempo:** each track plays at its own BPM. The incoming track
  matches the outgoing tempo during the blend, then ramps to its own over 16
  bars. The ramp is cut short if the track's next blend comes sooner.
  `--bpm` locks everything to one tempo instead.
- **Collision rule:** a track's outgoing blend starts at least 8 bars after its
  incoming blend ends. The planner re-plans if needed.

### Findings
1. **Variable-rate stretching is accurate.** Rubber Band R2 with a
   per-sample stretch array holds beats within ~4 ms through a 5% ramp
   (guarded by a test). The renderer's own time map (integral of 1/factor)
   matches it.
2. **The tempo cost had to be symmetric.** `|b/a − 1|` made 140→126 look
   cheaper than 126→140 and flipped the warm-up order. Now uses `|log(b/a)|`.
3. **Open-loop nudging from the analysis can be wrong in sparse regions.**
   On dashstar's spliced, fading tail, the analysis estimated +138 ms but the
   audio measured +57 ms or +143 ms depending on where the search started:
   the kick envelope there has two plausible alignments.
   **Fix: closed-loop nudging at render time.** Measure both decks' kicks in
   the actual stretched overlap, centered on the analysis estimate, and align
   them. A nudge shifts all later decks and blends too. Measurements with too
   few kicks, or at the edge of the ±¼-beat search, fall back to the analysis
   estimate. Strong-kick test blends measure 0–5 ms apart, both at a fixed
   tempo and with ramps.
4. **One deck in memory at a time** (plus the previous one), so long sets
   don't need all tracks in RAM.
5. **Planner speed:** recomputing the smoothed offsets for every candidate made
   20 tracks take 87 s. Cached per analysis: 2.4 s.

## M4: Transition library (`render.py`)
| Type | What happens | When the planner picks it |
|---|---|---|
| `bass_swap_blend` | Incoming highs and mids fade in, basslines swap on a downbeat, outgoing fades out | Compatible keys (≤1 step) |
| `filter_sweep` | Bass-swap blend, plus a high-pass rising on A (20 Hz → 2 kHz) and a low-pass opening on B (250 Hz → 20 kHz) | Keys 2 steps apart (filters hide the clash); half the mud penalty |
| `echo_out` | A's dry signal cuts on the 1; its last beat repeats on a ¾-beat delay (band-passed 300 Hz–6 kHz); B starts on the cut | Clashing keys or >6% tempo gap; a good way out of awkward regions |
| `drop_cut` | A plays its build; where A would drop, it cuts to B's drop | A has a build → drop and B has a drop; best for clashing keys |
| `acapella_overlay` | B's vocal stem over A's instrumental; A's vocals duck; on the overlay's last bar B's beat replaces A's | Needs stems (`--stems`) and keys ≤1 step apart; best when B's beat lands on its drop |

- Filters are applied as an STFT magnitude mask (4th-order Butterworth shape)
  with exponential cutoff sweeps, so they're smooth and zipper-free.
- Hard cuts use a 4 ms crossfade. No click in the rendered demo: the largest
  sample-to-sample jump at the cut is below the track's 99.9th percentile.
- Cuts line up the groove using 4 bars on each side of the cut.

## Stems (`stems.py`, `--stems`)
- Demucs `htdemucs` from the author's Hugging Face account (official), on the
  GPU (MPS): ~7 s per minute of audio. Cached as 16-bit FLAC (~60 MB per
  track) in `.cache/stems/`.
- **Vocal activity per bar = vocal-stem level relative to the full mix.**
  Sung sections measure −3…−8 dB and drops with chops or bleed −15…−25 dB,
  so the scale runs from −18 dB (none) to −8 dB (full). The first version
  (relative to the track's own loudest vocal bars) marked 88–100% of every
  section as vocal.

## Demo renders (gitignored, `output/plans/demo-*.wav`)
Knock2 dashstar (VIP) → ISOxo how2fly, one per type, plus `demo-auto` (the
planner's own choice, an echo-out at dashstar's splice point). Keys clash, so
judge the mechanics, not the harmony.

## Known gaps
- The section labeller is heuristic (energy-based). Real tracks will show mislabels.
- No `loop_extend` yet. The planner has no energy-arc model beyond tempo.
- The alignment self-check is ambiguous in sparse or spliced regions (it says
  "can't tell" or shows low confidence).
- Only tested on two bass-music tracks.
