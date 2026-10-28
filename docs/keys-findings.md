# Key detection & matching — findings (2026-10-27)

## Reference keys
9 songs from `~/Music/mysongs`, keys looked up online (Tunebat, SongBPM, Musicstax, Beatport,
Mixgraph, Songdata, Wikipedia). Sources disagree on 6 of 9 (Beatport most often), so the
reference is the majority answer and minority answers count as "any source". Stored locally
in `labels/keys.json` (git-ignored).

## Methods (`autodj eval-keys ~/Music/mysongs/`)
| Method | Exact | Matches any source | Mixable (≤1 Camelot step) | MIREX |
|---|---|---|---|---|
| mix (Krumhansl on the full mix) | 56% | 78% | 89% | 0.70 |
| **stems** (harmonic stems + bass line) | 56% | **89%** | 89% | 0.70 |
| Essentia `edma` (optional, AGPL) | 33% | 67% | 78% | 0.53 |
| Essentia `edma` on stems | 22% | 67% | 67% | 0.42 |

- **The stem method is now the default** when stems are separated (`key_source: "stems"`).
- **Essentia lost on this music** and is confidently wrong (Uncaged: 4A at 0.61 vs 12A). It stays an
  optional extra (`uv sync --extra keys`) for future comparisons; it's not used by default.
- **The real bug was confidence, not detection.** The old score was the margin between the best and
  runner-up key, but the runner-up is usually the relative major/minor, which mixes just as well. So
  correct keys showed ~0 confidence and the planner discarded them. Confidence now means: how far the
  best key beats every key 2+ Camelot steps away.
- 9 tracks is small and the references are noisy. Re-run `eval-keys` as more songs get references.

## Key matching by pitch shift
- +1 semitone = 7 steps round the Camelot wheel, +2 = 2 steps, so shifts of up to ±2 fix nearly any clash.
- Ordering: each pair's key cost uses the best shift for the incoming track
  (penalty 0.2 for 1 semitone, 0.5 for 2). Then actual shifts are assigned in order, each matching the
  previous track's *shifted* key, and transitions are re-planned on the shifted keys. Keys with
  confidence < 0.03 aren't shifted. `set --max-shift 0` turns it off.
- Render: Rubber Band pitch shift (R2 engine, formants preserved), on the whole track and its vocal
  stem. Timing check: a constant 2–5 ms offset, spread < 1 ms; the closed-loop nudge absorbs it.
- On `mysongs`: 4 songs shifted by one semitone (Clarity +1, shyne 4 me −1, how2fly −1, Rad Cat −1),
  and every adjacent pair ended up in compatible keys.

## Side effect: every transition became an acapella overlay
Overlays need compatible keys, so once shifting made every pair compatible, overlays won all 8 pairs.
Fixes:
- **Overlays rebalanced:** base −0.2, section change +0.3, drop +0.3 (was +0.2/+0.4/+0.4).
- **Set-level variety penalty:** −0.5 for repeating the previous transition's type, −0.25 for the one
  before that, −0.15 per earlier use anywhere in the set.

Result: 3 overlays, 2 drop cuts, an echo-out, a filter sweep and a bass-swap blend.

## Listening feedback (round 1) and fixes
> "The transitions are far too choppy and sudden, there's no sense of mixing between the tracks."
> "Too many soft outros that have a loud, more intense song coming in."

- Only 2 of 8 transitions actually overlapped two tracks: drop cuts and echo outs are instant, and
  acapella overlays swapped beats in 4 ms.
- **`--style smooth` (new default):** real blends only (bass-swap blend, filter sweep), a bonus for long
  overlaps (16–32 bars), lighter variety penalties. `--style mixed` keeps the old behaviour.
- **Acapella overlays now end with a blend:** B's beat fades in over the last 4 bars, the bass swaps
  on the downbeat, and A fades out over 2 bars.
- **Energy matching:** penalty when B comes in more than 0.15 above A's energy in the overlap (×2),
  or much quieter (more than 0.3 below), and when A has already faded (energy < 0.25).
- **Don't chase A's last bar:** playing 85% of a track now counts as fully played, so blends leave
  during sections with real energy instead of in the fade.
- Result on `mysongs`: 8 blends (6 filter sweeps, 2 bass swaps), mostly 16 bars. Energy on each side
  now mostly within ±0.2 (e.g. 0.55 → 0.56, 0.81 → 0.80).

## Listening feedback (round 2) and fixes
> "Too many where the first song clearly just fades out... the first transition would be better if the
> vocals played throughout the whole transition." / "More creative transitions or mashups of songs if they
> fit in the middle of a song."

- **`stem_blend` (new, preferred in smooth style when stems exist):** parts swap one at a time instead of
  whole tracks crossfading. B's drums creep in during the first half (its kick and bass held back by the
  low band); drums and bass swap on the swap downbeat; melodies crossfade over the second half while
  **A's vocals carry on over B's beat** to the end of the blend, then finish over one bar as B's vocals
  come in. Needs stems and keys ≤1 step apart. Scoring: +0.2, plus +0.4 when A sings through the second
  half; half the mud penalty; no vocal-clash penalty (B's vocals are held back).
- Renderer: per-stem gains are applied as `mix + Σ (gain − 1) · stem`, so untouched parts stay exactly the
  original mix (no separation artefacts unless a part is changed).
- **Mid-song mashups (`--mashups 2`):** in a track's solo stretch, an instrumental window (drop/body/build,
  ≤25% vocals, energy ≥ 0.5) gets an 8–16 bar sung phrase (≥60% vocals) from another track in the set:
  tempo gap ≤ 8% (stretched to the host), keys within one step after a shift of ±2 semitones or less, bar-aligned
  with a nudge, faded in and out over a beat, 2 dB under. Teasers of later tracks score higher than
  callbacks; neighbours in the set lower. Max 2 per set, one per host, each guest used once.
- On `mysongs`: all 8 transitions are stem blends (Ur Alive → gettin' hott scores highest, +2.33,
  with Ur Alive's vocal carried over). 2 mashups: Uncaged's vocal phrase over shyne 4 me's drop, and
  Under Your Spell's sung drop over how2fly's second drop.

## Round 3: more transition types
New: `melody_first`, `reverb_wash`, `double_drop` (smooth style) and `beat_roll`, `tape_stop` (mixed style).
The smooth-style variety penalty went up to (0.35, 0.1, 0.05) so the new types actually get used. Reverb
wash has a slightly negative base, so it needs a reason (vocals to wash out, or a key clash) to beat a
plain blend. Demos in `output/demos/`. On `mysongs`: 4 stem blends, 3 melody-first, 1 double drop
(Uncaged → Under Your Spell), plus the 2 mashups.

## Round 4: listening feedback on the new types
> 1. melody_first: gettin' hott's melody too aggressive for Ur Alive's soft outro.
> 2. reverb_wash: too reverby for too long; big anticipation in Uncaged → calmer, bouncier drop in Under Your Spell.
> 3. double_drop: interrupts Uncaged's biggest drop; not a good transition.
> 4. beat_roll: could work if the vibes matched; Uncaged is too aggressive.
> 5. tape_stop: just a cut; wants pure mixes.

Root causes: no notion of *character* that compares across tracks (bar_energy is relative within a
track), and no respect for a track's arc.

- **New per-bar character, comparable across tracks:** loudness relative to the track's LUFS, grit
  (2–10 kHz spectral flatness) and drums-stem level. Ur Alive's outro and gettin' hott's intro are equally
  loud, but grit is 0.06 vs 0.29 and drums −36 dB vs −9 dB.
- **Transitions:** penalty when B is harsher or punchier than A's section (and, more mildly, much
  softer); a letdown penalty when B's landing drop is a step down from A's drops.
- **Arc:** −0.8 for leaving mid-build or just as A's own drop arrives; −1.0 for interrupting A's biggest
  drop (≥8-bar drops only; blending out of a drop's last 4 bars is fine).
- **Ordering:** a vibe cost between neighbours (drop character), plus an extra cost when the warm-up's
  energy falls back.
- **Pure mixes:** double drop removed from the smooth style (bug fix: it was being offered regardless of
  style). Reverb wash limited to ≤ 8 bars, wet ≤ 0.45 over the second half only, 1-bar tail.
- **Open:** Uncaged → Under Your Spell stays the weakest pair (score ≈ 0). Uncaged has only 4 bars after its
  last drop, so every option either leaves before that drop or blends from its tail into a louder intro.
  Needs loop-extend (repeat A's last bars to make room) or a different order.
