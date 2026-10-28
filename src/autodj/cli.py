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


def _warn_pair(a, b) -> None:
    from autodj.camelot import camelot_distance
    from autodj.setlist import tempo_gap

    name_a, name_b = Path(a.path).stem, Path(b.path).stem
    dist = camelot_distance(a.key, b.key)
    if dist > 1:
        typer.secho(f"  warning: {name_a} -> {name_b}: keys {a.key}/{b.key} are {dist} steps apart; "
                    f"expect a harmonic clash", fg="yellow")
    if tempo_gap(a.bpm, b.bpm) > 0.06:
        typer.secho(f"  warning: {name_a} -> {name_b}: {a.bpm:.1f} vs {b.bpm:.1f} BPM needs a "
                    f"{tempo_gap(a.bpm, b.bpm):.0%} stretch during the blend", fg="yellow")
    for x in (a, b):
        if not x.grid_ok:
            typer.secho(f"  warning: {Path(x.path).stem}'s beat grid drifts ({x.grid_drift_ms:+} ms); "
                        f"blends are nudged locally but listen for flams", fg="yellow")


def _write_plan(plan, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2))
    typer.echo(f"plan: {output}\nrender it with: autodj render \"{output}\"")


@app.command()
def pair(
    track_a: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    track_b: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    bpm: Annotated[float | None, typer.Option(help="Lock both tracks to this tempo (default: each plays at its own, B matches A during the blend)")] = None,
    out_bar: Annotated[int | None, typer.Option(help="Bar of A where the blend starts (default: planner)")] = None,
    in_bar: Annotated[int | None, typer.Option(help="Bar of B that lands on out_bar (default: planner)")] = None,
    overlap: Annotated[int | None, typer.Option(help="Blend length in bars (default: planner)")] = None,
    output: Annotated[Path | None, typer.Option("-o", help="Plan file to write")] = None,
) -> None:
    """Draft a two-track MixPlan (A into B with a bass-swap blend)."""
    from autodj.analysis import analyze_file
    from autodj.models import MixPlan, PlanTrack, Transition
    from autodj.planner import candidates
    from autodj.report import OUTPUT_DIR

    a, b = analyze_file(track_a), analyze_file(track_b)
    if out_bar is None and in_bar is None and overlap is None:
        ranked = candidates(a, b)
        for i, c in enumerate(ranked[:3]):
            typer.echo(f"{'*' if i == 0 else ' '} {c.score:+.2f}  {c.describe()}")
            for why in c.reasons:
                typer.echo(f"         {why}")
        transition = ranked[0].transition()
        transition.notes = [f"score {ranked[0].score:+.2f}"] + ranked[0].reasons
    else:
        transition = Transition(out_bar=out_bar if out_bar is not None else a.n_bars - (overlap or 16),
                                in_bar=in_bar or 0, overlap_bars=overlap or 16)
    plan = MixPlan(name=f"{track_a.stem} -- {track_b.stem}", bpm=bpm,
                   tracks=[PlanTrack(path=a.path), PlanTrack(path=b.path)], transitions=[transition])
    _warn_pair(a, b)
    _write_plan(plan, output or OUTPUT_DIR / "plans" / f"{plan.name}.json")


@app.command(name="set")
def build_set(
    source: Annotated[Path, typer.Argument(exists=True, help="Playlist .yaml, or a folder of audio files")],
    keep_order: Annotated[bool | None, typer.Option("--keep-order/--reorder", help="Keep the playlist order")] = None,
    arc: Annotated[str | None, typer.Option(help="warmup_peak (tempo rises) | flat")] = None,
    bpm: Annotated[float | None, typer.Option(help="Lock every track to this tempo")] = None,
    output: Annotated[Path | None, typer.Option("-o", help="Plan file to write")] = None,
) -> None:
    """Plan a whole set: analyze, order by key/tempo/transition quality, plan every blend."""
    from autodj.analysis import analyze_file
    from autodj.report import OUTPUT_DIR
    from autodj.setlist import build_plan

    plan = build_plan(source, keep_order=keep_order, arc=arc, bpm=bpm)
    analyses = [analyze_file(Path(t.path), bpm_hint=t.bpm_hint) for t in plan.tracks]
    for k, (t, a) in enumerate(zip(plan.tracks, analyses)):
        key = t.key or a.key
        typer.echo(f"{k + 1:3d}. {Path(t.path).stem}  [{a.bpm:.1f} BPM, {key}]")
        if k < len(plan.transitions):
            tr = plan.transitions[k]
            typer.echo(f"       -> {tr.overlap_bars}-bar blend: bar {tr.out_bar} into bar {tr.in_bar} "
                       f"({tr.notes[0] if tr.notes else ''})")
            _warn_pair(a.model_copy(update={"key": key}),
                       analyses[k + 1].model_copy(update={"key": plan.tracks[k + 1].key or analyses[k + 1].key}))
    _write_plan(plan, output or OUTPUT_DIR / "sets" / f"{plan.name}.json")


@app.command()
def render(
    plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("-o", help="WAV to write")] = None,
) -> None:
    """Render a MixPlan to a WAV, plus a tracklist and an alignment check."""
    import json

    import soundfile as sf

    from autodj.analysis.beats import MIN_CONFIDENCE
    from autodj.models import MixPlan
    from autodj.render import mix_alignment
    from autodj.render import render as render_plan

    plan = MixPlan.model_validate_json(plan_file.read_text())
    mix, decks, windows = render_plan(plan)
    output = output or plan_file.with_suffix(".wav")
    sf.write(output, mix.T, plan.sample_rate, subtype="PCM_24")

    def clock(t: float) -> str:
        return f"{int(t // 60)}:{t % 60:04.1f}"

    tracklist = {"plan": str(plan_file), "tracks": [], "transitions": []}
    for d, t in zip(decks, plan.tracks):
        start = d.offset + d.play_from
        name = " - ".join(x for x in (t.artist, t.title) if x) or Path(d.analysis.path).stem
        tracklist["tracks"].append({"title": name, "path": d.analysis.path, "start": round(start, 3),
                                    "end": round(d.offset + d.play_to, 3), "native_bpm": d.native_bpm,
                                    "plays_at_bpm": round(d.target_bpm, 2)})
        typer.echo(f"{clock(start):>7}  {name}  ({d.native_bpm:.1f} BPM)")
    for w in windows:
        (off_a, conf_a), (off_b, conf_b) = mix_alignment(plan, w)
        flam = abs(off_a - off_b)
        tracklist["transitions"].append({"start": round(w.start, 3), "swap": round(w.swap, 3), "end": round(w.end, 3),
                                         "bpm": round(60 / w.beat, 2), "nudge_ms": round(w.nudge_ms, 1),
                                         "nudge_source": w.nudge_source, "out_offset_ms": round(off_a, 1),
                                         "in_offset_ms": round(off_b, 1), "confidence": [round(conf_a, 2), round(conf_b, 2)]})
        if min(conf_a, conf_b) < MIN_CONFIDENCE:
            verdict = "can't tell (too few kicks in the overlap), listen to it"
        else:
            verdict = "tight" if flam <= 10 else "audible flam" if flam > 20 else "borderline"
        typer.echo(f"         blend {clock(w.start)}-{clock(w.end)} at {60 / w.beat:.1f} BPM, bass swap {clock(w.swap)}, "
                   f"nudge {w.nudge_ms:+.0f} ms ({w.nudge_source}): kicks {flam:.1f} ms apart "
                   f"(confidence {conf_a:.1f}/{conf_b:.1f}), {verdict}")
    output.with_suffix(".tracklist.json").write_text(json.dumps(tracklist, indent=2))
    typer.echo(f"mix: {output}  ({mix.shape[1] / plan.sample_rate / 60:.1f} min)")
