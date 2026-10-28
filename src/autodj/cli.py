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
    backend: Annotated[str, typer.Option(help="Downbeat method: auto | beat_this | heuristic")] = "auto",
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
        typer.echo(f"  grid: {grid} (drift {a.grid_drift_ms:+} ms, onset offset {a.onset_offset_ms:+} ms), "
                   f"{len(a.downbeats)} bars")
        for s in a.sections:
            typer.echo(f"    {s.start:7.1f}s  bar {s.start_bar:3d}-{s.end_bar:<3d} {s.label:<6} energy {s.energy:.2f}")
        if report:
            out = report_dir(a)
            write_clicks(a, out)
            write_plot(a, out)
            typer.echo(f"  report: {out}/")


def _default_out_bar(a, overlap: int) -> int:
    """Latest section start with room for the whole overlap (usually the outro or last break)."""
    n = len(a.downbeats)
    starts = [s.start_bar for s in a.sections if s.start_bar + overlap <= n]
    return max(starts) if starts else max(0, n - overlap)


@app.command()
def pair(
    track_a: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    track_b: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    bpm: Annotated[float | None, typer.Option(help="Master tempo (default: track A's)")] = None,
    out_bar: Annotated[int | None, typer.Option(help="Bar of A where the blend starts")] = None,
    in_bar: Annotated[int, typer.Option(help="Bar of B that lands on out_bar")] = 0,
    overlap: Annotated[int, typer.Option(help="Blend length in bars")] = 16,
    output: Annotated[Path | None, typer.Option("-o", help="Plan file to write")] = None,
) -> None:
    """Draft a two-track MixPlan (A into B with a bass-swap blend)."""
    from autodj.analysis import analyze_file
    from autodj.camelot import camelot_distance
    from autodj.models import MixPlan, PlanTrack, Transition
    from autodj.report import OUTPUT_DIR

    a, b = analyze_file(track_a), analyze_file(track_b)
    plan = MixPlan(
        bpm=bpm or a.bpm,
        tracks=[PlanTrack(path=a.path), PlanTrack(path=b.path)],
        transitions=[Transition(out_bar=_default_out_bar(a, overlap) if out_bar is None else out_bar,
                                in_bar=in_bar, overlap_bars=overlap)],
    )
    tr = plan.transitions[0]
    output = output or OUTPUT_DIR / "plans" / f"{track_a.stem} -- {track_b.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2))

    typer.echo(f"A: {track_a.name}  {a.bpm} BPM {a.key}  ({plan.bpm / a.bpm - 1:+.1%} tempo)")
    typer.echo(f"B: {track_b.name}  {b.bpm} BPM {b.key}  ({plan.bpm / b.bpm - 1:+.1%} tempo)")
    typer.echo(f"blend: A bar {tr.out_bar} = B bar {tr.in_bar}, {tr.overlap_bars} bars at {plan.bpm} BPM")
    dist = camelot_distance(a.key, b.key)
    if dist > 1:
        typer.secho(f"warning: keys are {dist} steps apart on the Camelot wheel; expect a harmonic clash", fg="yellow")
    for name, x in (("A", a), ("B", b)):
        if not x.grid_ok:
            typer.secho(f"warning: {name}'s beat grid drifts ({x.grid_drift_ms:+} ms); the blend may flam", fg="yellow")
    typer.echo(f"plan: {output}")


@app.command()
def render(
    plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("-o", help="WAV to write")] = None,
) -> None:
    """Render a MixPlan to a WAV, plus a tracklist and an alignment check."""
    import json

    import soundfile as sf

    from autodj.models import MixPlan
    from autodj.render import MIN_ALIGNMENT_CONFIDENCE, mix_alignment
    from autodj.render import render as render_plan

    plan = MixPlan.model_validate_json(plan_file.read_text())
    mix, decks, windows = render_plan(plan)
    output = output or plan_file.with_suffix(".wav")
    sf.write(output, mix.T, plan.sample_rate, subtype="PCM_24")

    def clock(t: float) -> str:
        return f"{int(t // 60)}:{t % 60:04.1f}"

    tracklist = {"plan": str(plan_file), "bpm": plan.bpm, "tracks": [], "transitions": []}
    for d in decks:
        tracklist["tracks"].append({"path": d.analysis.path, "start": round(d.offset + d.play_from, 3),
                                    "end": round(d.offset + d.play_to, 3), "stretch": round(d.rate, 4)})
        typer.echo(f"{clock(d.offset + d.play_from):>7}  {Path(d.analysis.path).stem}  (x{d.rate:.3f})")
    for w in windows:
        (off_a, conf_a), (off_b, conf_b) = mix_alignment(plan, w)
        flam = abs(off_a - off_b)
        tracklist["transitions"].append({"start": round(w.start, 3), "swap": round(w.swap, 3), "end": round(w.end, 3),
                                         "out_offset_ms": round(off_a, 1), "in_offset_ms": round(off_b, 1),
                                         "confidence": [round(conf_a, 2), round(conf_b, 2)]})
        if min(conf_a, conf_b) < MIN_ALIGNMENT_CONFIDENCE:
            verdict = "can't tell (too few kicks in the overlap), listen to it"
        else:
            verdict = "tight" if flam <= 10 else "audible flam" if flam > 20 else "borderline"
        typer.echo(f"  blend {clock(w.start)}-{clock(w.end)} (bass swap {clock(w.swap)}): kicks {off_a:+.1f} / "
                   f"{off_b:+.1f} ms vs grid (confidence {conf_a:.1f} / {conf_b:.1f}) -> {flam:.1f} ms apart, {verdict}")
    output.with_suffix(".tracklist.json").write_text(json.dumps(tracklist, indent=2))
    typer.echo(f"mix: {output}  ({mix.shape[1] / plan.sample_rate / 60:.1f} min)")
