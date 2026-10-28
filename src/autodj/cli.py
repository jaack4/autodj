from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="autodj: plan and render DJ mixes.")


@app.callback()
def _root() -> None:
    pass


@app.command()
def analyze(
    paths: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    backend: Annotated[str, typer.Option(help="auto | beat_this | librosa")] = "auto",
    bpm: Annotated[float | None, typer.Option(help="Known BPM, resolves half/double time")] = None,
    report: Annotated[bool, typer.Option(help="Write clicks.wav and analysis.png to output/")] = True,
    force: Annotated[bool, typer.Option(help="Ignore the analysis cache")] = False,
) -> None:
    """Analyze tracks: beat grid, key, loudness, phrase-aligned sections."""
    from autodj.analysis import analyze_file
    from autodj.report import report_dir, write_clicks, write_plot

    for path in paths:
        a = analyze_file(path, backend=backend, bpm_hint=bpm, force=force)
        grid = "ok" if a.grid_ok else "DRIFTING (tempo not constant or BPM off)"
        typer.secho(f"\n{path.name}", bold=True)
        typer.echo(f"  {a.bpm} BPM  {a.key} ({a.key_name}, conf {a.key_confidence})  {a.lufs} LUFS  "
                   f"{a.duration:.0f}s  [{a.beat_backend}]")
        typer.echo(f"  grid: {grid} (drift {a.grid_drift_ms:+} ms, kick offset {a.kick_offset_ms:+} ms, "
                   f"kicks on {a.kick_coverage:.0%} of beats), {len(a.downbeats)} bars")
        for s in a.sections:
            typer.echo(f"    {s.start:7.1f}s  bar {s.start_bar:3d}-{s.end_bar:<3d} {s.label:<6} energy {s.energy:.2f}")
        if report:
            out = report_dir(a)
            write_clicks(a, out)
            write_plot(a, out)
            typer.echo(f"  report: {out}/")
