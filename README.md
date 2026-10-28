# autodj

An AI DJ: give it a playlist of songs and it plans the set order and mixes the
tracks together with DJ-style transitions (bass swaps, filter sweeps, echo outs,
drop cuts, acapella overlays), then renders one continuous mix.

> **Status:** M0–M5 built (analysis, rendering, transition planning, full sets,
> transition library, stems, an LLM creative director). Needs listening tests on real playlists. See
> [docs/PLAN.md](docs/PLAN.md) and the findings for [M0](docs/M0-findings.md),
> [M1](docs/M1-findings.md) and [M2–M4](docs/M2-M4-findings.md).

## How it works

```
playlist.yaml → analyze (tempo, bars, sections, key, vocals) → order tracks → plan transitions → MixPlan.json → render → mix.wav
```

The "smart" part only produces a human-readable `MixPlan.json`. Rendering audio
from it is deterministic, so you can inspect, tweak and re-render any plan.

## Setup

```sh
uv sync
uv run pytest        # 58 tests, ~5 s
```

Requires Python 3.12 and `ffmpeg`. For `beat_this` downbeats while its official
download is down, see `scripts/import_beat_this_weights.py`. Without it, a
heuristic decides downbeats.

## Try it

Use your own local audio files (MP3, WAV, FLAC, AIFF, M4A). Audio is never
committed to this repo. Everything is written to `output/` and cached in `.cache/`.

**1. Check the analysis of a track** (~15 s per track, then cached)
```sh
uv run autodj analyze ~/Music/new/*.mp3
```
For each track: BPM, key, loudness, grid health and sections. Each track also
gets `output/<track>/clicks.wav` and `analysis.png`.
*Listen for:* high clicks on every beat, a low click on every "1", and a tone
where sections change. *Look for:* "grid: ok". "DRIFTING" means the tempo isn't
constant or the file has an edit in it.

**2. Plan and render a whole set**
```sh
uv run autodj set ~/Music/new/               # a folder; or a playlist .yaml (see examples/)
uv run autodj render output/sets/new.json    # -> output/sets/new.wav, new.tracklist.json, new.html
```
Open `output/sets/new.html` in a browser. It shows the mix as one lane per
track, coloured by section, with every transition marked. Click a transition
in the table to jump to 4 bars before it, or click anywhere on the timeline.
`set` prints the order, each transition with the planner's score, and warnings
(key clashes, big tempo jumps, drifting grids). `render` prints each track's
start time and, for every blend, how far apart the two tracks' kicks are
("tight" ≤ 10 ms).
*Listen for:* kicks that flam or gallop during blends, bass swaps landing on
the 1, and cuts landing on the 1.

Options: `--stems` (separates stems with Demucs, ~30 s per track: better keys
and sections, avoids vocal clashes, enables acapella overlays), `--keep-order`,
`--arc flat|warmup_peak`, `--bpm 126` (lock everything to one tempo),
`--max-shift 2` (pitch-shift tracks up to 2 semitones so keys match; 0 = off).
Keys you know (from djay, Rekordbox, Beatport...) can go in a playlist file and
override detection. `autodj eval-keys <folder>` compares key detectors against
references in `labels/keys.json` (see [docs/keys-findings.md](docs/keys-findings.md)).

**3. Try one specific transition**
```sh
uv run autodj pair a.mp3 b.mp3                        # planner's choice, top 3 candidates with reasons
uv run autodj pair a.mp3 b.mp3 --type echo_out        # or bass_swap_blend, filter_sweep, drop_cut, acapella_overlay (needs --stems)
uv run autodj render "output/plans/a -- b.echo_out.json"
```

**4. Let Claude direct the set** (optional, needs `ANTHROPIC_API_KEY`)
```sh
uv run autodj direct output/sets/new.json            # -> output/sets/new.directed.json
uv run autodj render output/sets/new.directed.json   # render both and compare
```
The creative director (Claude Opus 5.5) sees each transition's checked options
(the planner's pick, the best of each type, and a few runners-up) plus a summary
of every track. It picks one option per transition, explains each pick, and
titles and narrates the set. It only chooses; the planner and renderer still
handle all timing and audio, and an invalid pick is replaced automatically. One
request per set: about $0.04 for 3 tracks, roughly $0.10–0.50 for 20.
`set --director` runs it straight after planning.

**5. Teach it your sections (improves analysis)**
```sh
uv run autodj label ~/Music/new/        # opens a page in your browser: press 1-6 at each section start
uv run autodj eval-sections             # scores both detectors (mix-only vs stem-based) against your labels
```
With `--stems`, sections are detected from the separated layers (kick, risers,
vocals) and repeats are grouped ("drop 2 repeats drop 1"). Your labels in
`labels/` (git-ignored) show whether that actually beats the mix-only method.

**6. Tweak a plan by hand.** Edit the JSON (bars, overlap, transition type,
`ramp_bars`, a track's `bpm`) and run `render` again.

## Playlist format
See [`examples/playlist.example.yaml`](examples/playlist.example.yaml). `bpm`
and `key` are optional; if given, they override detection. `key` accepts
Camelot (`8A`) or note names (`A minor`, `F#m`, `Bb`).

## License

MIT
