# autodj

An AI DJ: give it a playlist of songs (with BPM and key) and it plans the set
order and mixes the tracks together with DJ-style transitions (bass swaps,
filter sweeps, echo outs, acapella overlays), then renders one continuous mix.

> **Status:** M1 (two-track render) in progress. See [docs/PLAN.md](docs/PLAN.md) for the architecture and milestones, and the findings for [M0](docs/M0-findings.md) and [M1](docs/M1-findings.md).

## How it will work

```
playlist.yaml → analyze (beats, phrases, stems) → plan order → plan transitions → MixPlan.json → render → mix.wav
```

The "smart" part only produces a human-readable `MixPlan.json`. Rendering audio
from it is deterministic, so you can inspect, tweak and re-render any plan.

## Input

See [`examples/playlist.example.yaml`](examples/playlist.example.yaml). Use your
own local audio files (MP3, WAV, FLAC). Audio is never committed to this repo.

## Development

```sh
uv sync
uv run autodj analyze path/to/track.mp3   # beat grid, key, LUFS, sections
uv run autodj pair a.mp3 b.mp3            # draft a two-track MixPlan
uv run autodj render output/plans/a\ --\ b.json   # render it to WAV + tracklist
uv run pytest
```

`analyze` writes a cached analysis to `.cache/analysis/` and a report to
`output/<track>/`: `clicks.wav` (track + beat, bar and section clicks, so you can check by ear)
and `analysis.png`.

Requires Python 3.12 and `ffmpeg`. For `beat_this` downbeats while its official
download is down, see `scripts/import_beat_this_weights.py`.

## License

MIT
