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
