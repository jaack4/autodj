# autodj — Project Plan

## Goal

Give the tool a playlist of local audio files (with BPM and key) and get back a
continuous, DJ-style mix: tracks put in a sensible order and joined with
transitions that sound intentional, not just crossfades.

**v0 scope:** offline. Input is a folder or playlist file; output is a rendered
WAV and a tracklist. No real-time playback yet.

**Non-goals for now:** streaming-service audio (Spotify and Apple Music don't
allow it), live performance, a GUI.

## Core design idea: plan, then render

The pipeline is split by one inspectable artifact, the **MixPlan** (JSON).

```
playlist.yaml ──► Ingest ──► Analyze ──► Plan set ──► Plan transitions ──► MixPlan.json ──► Render ──► mix.wav
                              (cached)    (order)      (where + how)                      (deterministic)
```

- Everything "smart" (rules, search, the LLM) only writes the MixPlan.
- The renderer is dumb and deterministic: the same plan always gives the same audio.
- You can hand-edit a plan and re-render. That makes debugging, testing and
  comparing ideas cheap.

## Pipeline stages

### 1. Ingest
- Read `playlist.yaml` (see `examples/playlist.example.yaml`): path, optional
  title and artist, BPM, key.
- Normalize keys to Camelot notation (`8A`, `C major` and `Am` should all parse).
- Missing BPM or key gets auto-detected in Analyze. Values the user provides win.

### 2. Analyze (once per file, cached by content hash in `.cache/`)
| Output | Purpose | Candidate tool |
|---|---|---|
| Beat grid and downbeats | Align bars | `allin1` or `beat_this` |
| Sections (intro, build, drop, break, outro) and phrase boundaries | Pick transition points | Bar-level novelty + energy, segmented with DP that prefers 8/16/32-bar sections (`allin1` ruled out for now, see M0 findings) |
| Stems (vocals, drums, bass, other) | Bass swaps, acapella overlays | Demucs (`htdemucs`) |
| Per-bar energy and vocal presence | Avoid vocal clashes, shape energy | RMS and onset density on the mix and the vocal stem |
| Integrated loudness | Level-match tracks | `pyloudnorm` |
| Key and BPM (when missing) | Planning | `librosa` / `essentia` |

### 3. Plan the set order
- Pairwise compatibility score for A→B:
  - **Key:** Camelot distance (same = best; ±1 or A↔B = good; anything else is penalized)
  - **Tempo:** ratio between BPMs, allowing half or double time. Stretching more than ~6% is penalized.
  - **Energy:** penalize big jumps, unless the target arc calls for one.
- Ordering is a path search over that graph (greedy + 2-opt, or beam search),
  plus a target **energy arc** (warm-up → peak → cool-down).
- Option `keep_order: true`: skip reordering and only plan transitions.

### 4. Plan transitions
For each adjacent pair:
1. **Candidate points:** out-points in A (outro, break, end of a phrase) ×
   in-points in B (intro, start of a phrase). Always phrase-aligned, always on downbeats.
2. **Candidate types**, from the transition library (below), filtered by
   constraints such as key distance, vocal overlap and available stems.
3. **Score and pick.** Rule-based scoring in v0. Later, an LLM "creative
   director" chooses among the *valid* candidates using structured output. The
   LLM never does audio math. It only picks from options the system has already
   checked.
4. **Tempo plan:** where to ramp the tempo, and over how many bars.

### 5. Render
- Time-stretch and pitch-shift with Rubber Band (`pyrubberband`).
- Line up the downbeats of B with the downbeats of A over the overlap.
- Apply automation curves (gain, 3-band EQ, HPF/LPF, delay) per track or per stem with `pedalboard`.
- Sum, apply a light limiter, and export: `mix.wav`, plus a `tracklist.json`
  with timestamps.

## Transition library (initial)
| Name | What it does | Needs |
|---|---|---|
| `bass_swap_blend` | Long overlap (16–32 bars). The incoming track's lows are cut, then the bass lines swap on a phrase boundary. | Compatible key and tempo |
| `filter_sweep` | A high-passes out while B low-passes in | Short overlap is fine |
| `echo_out` | A ends on a delay tail; B drops in on a downbeat | Nothing (works for incompatible keys) |
| `drop_cut` | Hard cut from A's build into B's drop | A has a build, B has a drop |
| `acapella_overlay` | B's vocal stem over A's instrumental, before a full switch | Stems, key distance ≤ 1, no vocals in that part of A |
| `loop_extend` | Loop 4–8 bars of A's outro to make room | Clean loop region |

## Data model (sketch)
```
Track        id, path, title, artist, bpm, key_camelot
Analysis     track_id, beats[], downbeats[], sections[{label,start,end}], stems{}, bar_energy[], bar_vocal[], lufs
Transition   from_track, to_track, type, out_bar, in_bar, overlap_bars, tempo_ramp, automation{}
MixPlan      tracks[], order[], transitions[], master{target_lufs}
```
Pydantic models, serialized to JSON.

## Tech stack
- **Python 3.12.** The ML audio stack (torch, madmom, natten) usually lags the latest Python.
- `uv` for environment and dependencies, `typer` for the CLI, `pydantic` for models.
- Audio: `librosa`, `soundfile`, `pedalboard` (Rubber Band time-stretching, R2 engine; limiter),
  `scipy` (band split), `pyloudnorm`, `beat_this` (downbeats); `demucs` in M4.
- LLM (M5, optional): the Claude API, with structured output for choosing transitions.

## Milestones
- [ ] **M0: Analysis spike.** Install the stack and run it on 3–5 real tracks.
      Check beat, downbeat and section accuracy by ear or with clicks. Decide
      between `allin1` and `beat_this`, and decide whether a phrase-boundary
      fallback is needed. *Biggest risk; do this first.*
      **Mostly done. See [M0-findings.md](M0-findings.md).** Beats, bars and
      sections verified by ear on 2 tracks; more tracks pending.
- [ ] **M1: Two-track render.** Hand-written MixPlan for A→B with
      `bass_swap_blend` at hand-picked bars. This proves time-stretching,
      alignment, automation and export.
      **Engine done. See [M1-findings.md](M1-findings.md).** Waiting on a by-ear check.
- [ ] **M2: Automatic transition points.** Choose phrase-aligned out- and
      in-points from the analysis.
- [ ] **M3: Full set.** Ordering by compatibility and energy arc, render N
      tracks, export the tracklist. **First usable version.**
- [ ] **M4: Transition library.** Filter sweep, echo out, drop cut, stems and
      acapella overlay, loop extend.
- [ ] **M5: LLM creative director.** Picks transitions and writes a narrative for the set.
- [ ] **Later:** a simple waveform and plan viewer, export to rekordbox or
      Mixxx, real-time playback.

## Quality checks
- **Automatic:** beat-alignment error during overlaps (cross-correlate onset
  envelopes), clipping, loudness consistency between tracks, no overlap of two
  vocal stems.
- **By ear:** a fixed test playlist rendered after every change. Keep A/B
  renders for comparison.

## Repo hygiene
- **Never commit audio.** The repo is public and music is copyrighted. Audio
  files, `.cache/` and `output/` are git-ignored.
- Tests use short synthetic signals (clicks and tones) generated in code.

## Open questions
- Main genre(s)? v0 assumes 4/4 time and 16/32-bar phrases, which fits
  house and techno best. Hip-hop and pop need looser rules.
- How much creative freedom: reorder freely, or respect the user's order by default?
- Target mix length: use full tracks, or trim them, as DJs often do?
