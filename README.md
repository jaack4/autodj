# autodj

An AI DJ: give it a playlist of songs (with BPM and key) and it plans the set
order and mixes the tracks together with DJ-style transitions (bass swaps,
filter sweeps, echo outs, acapella overlays), then renders one continuous mix.

> **Status:** M0 (analysis) in progress. See [docs/PLAN.md](docs/PLAN.md) for the architecture and milestones, and [docs/M0-findings.md](docs/M0-findings.md) for current results.

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
uv run pytest
```

`analyze` writes a cached analysis to `.cache/analysis/` and a report to
`output/<track>/`: `clicks.wav` (track + beat, bar and section clicks, so you can check by ear)
and `analysis.png`.

Requires Python 3.12, plus `brew install rubberband ffmpeg` for time-stretching and decoding.

## License

MIT
