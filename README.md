# autodj

An AI DJ: give it a playlist of songs (with BPM and key) and it plans the set
order and mixes the tracks together with DJ-style transitions (bass swaps,
filter sweeps, echo outs, acapella overlays), then renders one continuous mix.

> **Status:** planning. See [docs/PLAN.md](docs/PLAN.md) for the architecture and milestones.

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
uv run autodj
```

Requires Python 3.12, plus `brew install rubberband ffmpeg` for time-stretching and decoding.

## License

MIT
