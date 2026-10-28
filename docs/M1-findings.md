# M1: Two-track render — findings (2026-10-27)

**Status:** engine works; waiting on a by-ear check, ideally with a key-compatible pair.

## What was built
```sh
autodj pair A.mp3 B.mp3 [--bpm 133] [--out-bar N] [--in-bar 0] [--overlap 16]   # writes output/plans/A -- B.json
autodj render output/plans/A\ --\ B.json                                          # writes .wav + .tracklist.json
```
- **MixPlan** (`models.py`): master BPM, tracks, and one `Transition` per pair
  (`out_bar` of A = `in_bar` of B, `overlap_bars`, optional `swap_bar`).
  `pair` picks A's last section start that fits the overlap.
- **Renderer** (`render.py`):
  1. Time-stretch every track to the master BPM (Rubber Band via `pedalboard`).
  2. Place each track so the chosen bars coincide.
  3. Split each track into 3 bands (zero-phase, sums back exactly).
  4. Automate gains with breakpoint curves.
  5. Sum, match loudness to −9 LUFS per track, limit, apply a −1 dBFS ceiling.
- **`bass_swap_blend`**: the incoming track's mids and highs fade in
  (equal power) over the first half with its bass cut. Basslines swap on the
  middle downbeat (a 50 ms ramp that ends on the beat). The outgoing track's
  mids and highs fade out over the second half.
- **Self-check:** for each overlap, each deck's kick-band offset against the
  master grid. The difference between the two is the flam you'd hear. If an
  overlap has too few kicks to measure, it says "can't tell" instead of
  reporting a number.

## Findings
1. **Rubber Band's high-quality R3 engine is unusable for beat-matching.**
   At ±5% stretch, beat positions drift 13→59 ms within a 2-minute track (the
   total length is still exact). The R2 engine (`high_quality=False`) holds
   beats to within 0.2 ms, with a constant ~±1.4 ms offset. This is guarded by
   `tests/test_render.py`. With R3, the first render had kicks 29 ms apart in
   the blend; with R2 the engine test gives **4.5 ms ("tight")**.
2. **JUCE's limiter outputs at 0 dBFS** (its threshold is input gain). Added
   −1 dB of output gain afterwards. Mixes peak at exactly −1.0 dBFS.
3. **Sparse intros can't be measured**, which matches the M0 drift-check
   noise. dashstar's intro gives confidence 1.3 (a strong-kick section gives
   3–4). The check now says so instead of reporting a false 99 ms.
4. Band levels across the test blend: highs flat (±0.5 dB, so the fades are
   equal-power), and bass back to full exactly on the swap bar.
5. **`pedalboard.time_stretch` accepts a per-sample stretch array** (tempo
   ramps between tracks with different BPMs) and pitch shifting (nudging keys
   to match). Both are useful for M3 and M4.
6. Render speed: ~18 s for a 6-minute two-track mix (M5 CPU), not optimized.

## Test renders (gitignored, in `output/plans/`)
| File | What it is |
|---|---|
| `ISOxo - how2fly -- Knock2 - dashstar (VIP).wav` | The musical version: how2fly's last breaks into dashstar's intro at 133 BPM (both ~5% stretched). **Keys clash (6B vs 9A)**, so judge timing, not harmony. Blend at 3:42–4:11, bass swap at 3:56.5 |
| `engine-test-in-bar-32.wav` | Same, but dashstar enters at bar 32 (strong kicks) so alignment can be measured: 4.5 ms apart |

## Next
- [ ] Listen: does the blend sit together (no flam or "galloping" kicks)? Does the bass swap land on the 1?
- [ ] Repeat with a key-compatible pair once new tracks arrive.
- [ ] Then M2: choose `out_bar` and `in_bar` automatically (phrase-aligned, energy-aware, avoiding drifting grids or splices).
