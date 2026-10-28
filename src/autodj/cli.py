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
    stems: Annotated[bool, typer.Option(help="Also separate stems (Demucs) and measure vocals per bar")] = False,
) -> None:
    """Analyze tracks: beat grid, key, loudness, phrase-aligned sections."""
    from autodj.analysis import analyze_file
    from autodj.report import report_dir, write_clicks, write_plot

    for path in paths:
        a = analyze_file(path, backend=backend, bpm_hint=bpm, force=force, stems=stems)
        grid = "ok" if a.grid_ok else "DRIFTING (tempo not constant or BPM off)"
        if not a.has_steady_beat:
            grid = "NO STEADY BEAT (tempo is a guess)"
        typer.secho(f"\n{path.name}", bold=True)
        typer.echo(f"  {a.bpm} BPM  {a.key} ({a.key_name}, conf {a.key_confidence})  {a.lufs} LUFS  "
                   f"{a.duration:.0f}s  [{a.beat_backend}]")
        typer.echo(f"  grid: {grid} (drift {a.grid_drift_ms:+} ms, onset offset {a.onset_offset_ms:+} ms), "
                   f"{len(a.downbeats)} bars")
        for s in a.sections:
            vocals = ""
            if a.bar_vocals is not None:
                vocals = f"  vocals {sum(v > 0.5 for v in a.bar_vocals[s.start_bar:s.end_bar]) / (s.end_bar - s.start_bar):.0%}"
            typer.echo(f"    {s.start:7.1f}s  bar {s.start_bar:3d}-{s.end_bar:<3d} {s.label:<6} energy {s.energy:.2f}{vocals}")
        if report:
            out = report_dir(a)
            write_clicks(a, out)
            write_plot(a, out)
            typer.echo(f"  report: {out}/")


def _warn_pair(a, b, kind: str) -> None:
    from autodj.camelot import camelot_distance
    from autodj.planner import CUTS, tempo_gap

    name_a, name_b = Path(a.path).stem, Path(b.path).stem
    dist = camelot_distance(a.key, b.key)
    gap = tempo_gap(a.bpm, b.bpm)
    if dist > 1 and kind not in CUTS:
        typer.secho(f"  warning: {name_a} -> {name_b}: keys {a.key}/{b.key} are {dist} steps apart; "
                    f"expect some harmonic clash during the {kind}", fg="yellow")
    if gap > 0.06:
        how = ("B starts at A's tempo, then ramps" if kind in CUTS
               else "B is stretched that far while both play")
        typer.secho(f"  warning: {name_a} -> {name_b}: {a.bpm:.1f} vs {b.bpm:.1f} BPM is a {gap:.0%} gap ({how})",
                    fg="yellow")
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
    kind: Annotated[str | None, typer.Option("--type", help="bass_swap_blend | filter_sweep | echo_out | drop_cut | acapella_overlay (default: planner picks)")] = None,
    stems: Annotated[bool, typer.Option(help="Separate stems: avoids vocal clashes, enables acapella_overlay")] = False,
    output: Annotated[Path | None, typer.Option("-o", help="Plan file to write")] = None,
) -> None:
    """Draft a two-track MixPlan; the planner picks the points and transition type unless you do."""
    from autodj.analysis import analyze_file
    from autodj.models import MixPlan, PlanTrack, Transition
    from autodj.planner import candidates
    from autodj.report import OUTPUT_DIR

    a, b = analyze_file(track_a, stems=stems), analyze_file(track_b, stems=stems)
    if out_bar is None and in_bar is None and overlap is None:
        ranked = candidates(a, b, kinds=(kind,) if kind else None)
        if not ranked:
            raise typer.BadParameter(f"no {kind} transition fits these tracks")
        for i, c in enumerate(ranked[:3]):
            typer.echo(f"{'*' if i == 0 else ' '} {c.score:+.2f}  {c.describe()}")
            for why in c.reasons:
                typer.echo(f"         {why}")
        transition = ranked[0].transition()
        transition.notes = [f"score {ranked[0].score:+.2f}"] + ranked[0].reasons
    else:
        cut = kind in ("echo_out", "drop_cut")
        length = 0 if cut else (overlap or 16)
        transition = Transition(type=kind or "bass_swap_blend", in_bar=in_bar or 0, overlap_bars=length,
                                out_bar=out_bar if out_bar is not None else a.n_bars - max(length, 4))
    plan = MixPlan(name=f"{track_a.stem} -- {track_b.stem}", bpm=bpm,
                   tracks=[PlanTrack(path=a.path), PlanTrack(path=b.path)], transitions=[transition])
    _warn_pair(a, b, transition.type)
    _write_plan(plan, output or OUTPUT_DIR / "plans" / f"{plan.name}{'.' + kind if kind else ''}.json")


@app.command(name="set")
def build_set(
    source: Annotated[Path, typer.Argument(exists=True, help="Playlist .yaml, or a folder of audio files")],
    keep_order: Annotated[bool | None, typer.Option("--keep-order/--reorder", help="Keep the playlist order")] = None,
    arc: Annotated[str | None, typer.Option(help="warmup_peak (tempo rises) | flat")] = None,
    bpm: Annotated[float | None, typer.Option(help="Lock every track to this tempo")] = None,
    stems: Annotated[bool, typer.Option(help="Separate stems: avoids vocal clashes, enables acapella_overlay")] = False,
    max_shift: Annotated[int, typer.Option(help="Max pitch shift (semitones) to match keys; 0 turns it off")] = 2,
    director: Annotated[bool, typer.Option(help="Also ask Claude to pick transitions and narrate (needs ANTHROPIC_API_KEY)")] = False,
    output: Annotated[Path | None, typer.Option("-o", help="Plan file to write")] = None,
) -> None:
    """Plan a whole set: analyze, order by key/tempo/transition quality, plan every blend."""
    from autodj.analysis import analyze_file
    from autodj.report import OUTPUT_DIR
    from autodj.setlist import build_plan

    def log(msg: str) -> None:
        typer.secho(msg, err=True, fg="yellow" if msg.lstrip().startswith(("warning", "skipped")) else None)

    plan = build_plan(source, keep_order=keep_order, arc=arc, bpm=bpm, stems=stems, log=log, max_shift=max_shift)
    typer.echo("")
    from autodj.camelot import shift_key

    analyses = [analyze_file(Path(t.path), bpm_hint=t.bpm_hint) for t in plan.tracks]
    keys = [shift_key(t.key or a.key, t.semitones) for t, a in zip(plan.tracks, analyses)]
    for k, (t, a) in enumerate(zip(plan.tracks, analyses)):
        key = keys[k]
        shifted = f" (shifted {t.semitones:+d} semitones from {t.key or a.key})" if t.semitones else ""
        typer.echo(f"{k + 1:3d}. {Path(t.path).stem}  [{a.bpm:.1f} BPM, {key}{shifted}]")
        if k < len(plan.transitions):
            tr = plan.transitions[k]
            how = f"{tr.overlap_bars}-bar {tr.type}" if tr.overlap_bars else tr.type
            typer.echo(f"       -> {how}: bar {tr.out_bar} into bar {tr.in_bar} ({tr.notes[0] if tr.notes else ''})")
            _warn_pair(a.model_copy(update={"key": key}), analyses[k + 1].model_copy(update={"key": keys[k + 1]}),
                       tr.type)
    plan_path = output or OUTPUT_DIR / "sets" / f"{plan.name}.json"
    _write_plan(plan, plan_path)
    if director:
        _run_director(plan, plan_path, "claude-opus-5-5")


def _run_director(plan, plan_path: Path, model: str) -> Path | None:
    """Ask the creative director to re-pick transitions; writes <plan>.directed.json. Returns its path."""
    from autodj.analysis import analyze_file
    from autodj.director import DirectorError
    from autodj.director import direct as direct_plan

    from autodj.camelot import shift_key

    analyses = []
    for t in plan.tracks:  # keys as they'll be heard: user-given or detected, after any pitch shift
        a = analyze_file(Path(t.path), bpm_hint=t.bpm_hint)
        update = {"key": shift_key(t.key or a.key, t.semitones)}
        if t.key:
            update["key_confidence"] = 1.0
        analyses.append(a.model_copy(update=update))
    typer.echo(f"asking the creative director ({model})...", err=True)
    try:
        directed, repairs = direct_plan(plan, analyses, model=model)
    except DirectorError as e:
        typer.secho(f"director failed: {e}. The planner's plan is unchanged.", fg="red", err=True)
        return None
    d = directed.director
    typer.secho(f"\n{d['title']}", bold=True)
    typer.echo(d["narrative"])
    for k, (old, new) in enumerate(zip(plan.transitions, directed.transitions), 1):
        changed = (old.type, old.out_bar, old.in_bar, old.overlap_bars) != (new.type, new.out_bar, new.in_bar,
                                                                           new.overlap_bars)
        how = f"{new.overlap_bars}-bar {new.type}" if new.overlap_bars else new.type
        typer.echo(f"  {k}. {how}, bar {new.out_bar} into bar {new.in_bar}"
                   f"{'  (changed from the planner)' if changed else ''}\n     {new.notes[0]}")
    for r in repairs:
        typer.secho(f"  repaired: {r}", fg="yellow")
    cost = f", about ${d['cost_usd']:.2f}" if "cost_usd" in d else ""
    typer.echo(f"({d['model']}: {d['input_tokens']} input + {d['output_tokens']} output tokens{cost})")
    out = plan_path.with_name(plan_path.stem + ".directed.json")
    _write_plan(directed, out)
    return out


@app.command()
def direct(
    plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A plan from `set` or `pair`")],
    model: Annotated[str, typer.Option(help="Claude model")] = "claude-opus-5-5",
) -> None:
    """Have Claude (the creative director) pick each transition from the planner's options and narrate the set.

    Needs ANTHROPIC_API_KEY. One request per set (typically $0.05-0.50). Writes <plan>.directed.json next
    to the original, so you can render both and compare.
    """
    from autodj.models import MixPlan

    if _run_director(MixPlan.model_validate_json(plan_file.read_text()), plan_file, model) is None:
        raise typer.Exit(1)


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
    from autodj.viewer import encode_mp3, write_viewer

    plan = MixPlan.model_validate_json(plan_file.read_text())
    mix, decks, windows = render_plan(plan, log=lambda msg: typer.echo(msg, err=True))
    output = output or plan_file.with_suffix(".wav")
    sf.write(output, mix.T, plan.sample_rate, subtype="PCM_24")

    def clock(t: float) -> str:
        return f"{int(t // 60)}:{t % 60:04.1f}"

    tracklist = {"plan": str(plan_file), "tracks": [], "transitions": []}
    checks: list[dict] = []
    for d, t in zip(decks, plan.tracks):
        start = d.offset + d.play_from
        name = " - ".join(x for x in (t.artist, t.title) if x) or Path(d.analysis.path).stem
        tracklist["tracks"].append({"title": name, "path": d.analysis.path, "start": round(start, 3),
                                    "end": round(d.offset + d.play_to, 3), "native_bpm": d.native_bpm,
                                    "plays_at_bpm": round(d.target_bpm, 2)})
        typer.echo(f"{clock(start):>7}  {name}  ({d.native_bpm:.1f} BPM)")
    for w in windows:
        kind = w.transition.type
        if not w.transition.overlap_bars:
            tracklist["transitions"].append({"type": kind, "at": round(w.start, 3), "nudge_ms": round(w.nudge_ms, 1),
                                             "nudge_source": w.nudge_source})
            typer.echo(f"         {kind} at {clock(w.start)}, nudge {w.nudge_ms:+.0f} ms ({w.nudge_source})")
            checks.append({"verdict": "cut: no overlap to check"})
            continue
        (off_a, conf_a), (off_b, conf_b) = mix_alignment(plan, w)
        flam = abs(off_a - off_b)
        tracklist["transitions"].append({"type": kind, "start": round(w.start, 3), "swap": round(w.swap, 3),
                                         "end": round(w.end, 3),
                                         "bpm": round(60 / w.beat, 2), "nudge_ms": round(w.nudge_ms, 1),
                                         "nudge_source": w.nudge_source, "out_offset_ms": round(off_a, 1),
                                         "in_offset_ms": round(off_b, 1), "confidence": [round(conf_a, 2), round(conf_b, 2)]})
        if min(conf_a, conf_b) < MIN_CONFIDENCE:
            verdict = "can't tell (too few kicks in the overlap), listen to it"
        else:
            verdict = "tight" if flam <= 10 else "audible flam" if flam > 20 else "borderline"
        if kind == "acapella_overlay":
            verdict += " (beats don't overlap in an acapella overlay; this is the vocal's timing)"
        checks.append({"verdict": f"kicks {flam:.1f} ms apart: {verdict}"})
        swap = f"bass swap {clock(w.swap)}, " if kind in ("bass_swap_blend", "filter_sweep") else ""
        typer.echo(f"         {kind} {clock(w.start)}-{clock(w.end)} at {60 / w.beat:.1f} BPM, {swap}"
                   f"nudge {w.nudge_ms:+.0f} ms ({w.nudge_source}): kicks {flam:.1f} ms apart "
                   f"(confidence {conf_a:.1f}/{conf_b:.1f}), {verdict}")
    output.with_suffix(".tracklist.json").write_text(json.dumps(tracklist, indent=2))
    mp3 = encode_mp3(output)
    viewer = write_viewer(output.with_suffix(".html"), mp3, plan, decks, windows, checks)
    typer.echo(f"mix: {output}  ({mix.shape[1] / plan.sample_rate / 60:.1f} min; MP3: {mp3.name})")
    typer.echo(f"viewer: {viewer}  (open it in a browser to listen and jump between transitions)")


def _audio_files(sources: list[Path]) -> list[Path]:
    from autodj.setlist import AUDIO_EXTENSIONS, load_playlist

    files: list[Path] = []
    for src in sources:
        if src.is_dir() or src.suffix.lower() in (".yaml", ".yml"):
            files += [Path(t.path) for t in load_playlist(src)[1]]
        elif src.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(src)
    return files


@app.command()
def label(
    sources: Annotated[list[Path], typer.Argument(exists=True, help="Audio files, folders or playlists")],
    port: Annotated[int, typer.Option(help="Local port")] = 8765,
) -> None:
    """Open a page in your browser for marking where sections start, by ear (saved to labels/)."""
    import webbrowser

    from autodj.analysis import analyze_file
    from autodj.labeler import LABELS_DIR, serve

    analyses = []
    for f in _audio_files(sources):
        typer.echo(f"analyzing {f.name}", err=True)
        analyses.append(analyze_file(f))
    if not analyses:
        raise typer.BadParameter("no audio files found")
    server = serve(analyses, port)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"labelling {len(analyses)} track(s) at {url}  (labels save to {LABELS_DIR}/; Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command(name="eval-sections")
def eval_sections(
    labels_dir: Annotated[Path, typer.Argument(help="Folder of label files from `autodj label`")] = Path("labels"),
    tolerance: Annotated[float, typer.Option(help="Boundary tolerance in bars")] = 1.0,
    stems: Annotated[bool, typer.Option(help="Also score stem-based sections (separates stems if needed)")] = True,
) -> None:
    """Score the section detectors (mix-only and stem-based) against your hand labels."""
    import json

    from autodj.analysis import analyze_file
    from autodj.sections_eval import score_track, summarize

    files = sorted(labels_dir.glob("*.json"))
    if not files:
        raise typer.BadParameter(f"no label files in {labels_dir}/ (make some with `autodj label`)")
    by_method: dict[str, list] = {"mix": [], "stems": []}
    typer.echo(f"{'track':36s} {'method':7s} {'boundaries P / R / F1':>22s} {'labels':>7s}")
    for f in files:
        data = json.loads(f.read_text())
        if not data["sections"] or not Path(data["path"]).exists():
            continue
        a = analyze_file(Path(data["path"]), stems=stems)
        variants = {"mix": a.sections_mix or a.sections}
        if a.sections_source == "stems":
            variants["stems"] = a.sections
        for method, sections in variants.items():
            s = score_track(a, data["sections"], Path(data["path"]).stem, tolerance, sections)
            by_method[method].append(s)
            typer.echo(f"{s.name[:36]:36s} {method:7s} {s.precision:5.0%} / {s.recall:4.0%} / {s.f1:4.0%} "
                       f"{s.label_accuracy:7.0%}")
    if not by_method["mix"]:
        raise typer.BadParameter("no usable label files")
    typer.echo("")
    for method, scores in by_method.items():
        if scores:
            total = summarize(scores)
            typer.secho(f"{'ALL (' + str(len(scores)) + ' tracks)':36s} {method:7s} {total.precision:5.0%} / "
                        f"{total.recall:4.0%} / {total.f1:4.0%} {total.label_accuracy:7.0%}", bold=True)
    total = summarize(by_method["stems"] or by_method["mix"])
    typer.echo(f"\nmost common label mix-ups for {'stems' if by_method['stems'] else 'mix'} "
               f"(your label -> detected, in bars):")
    for (t, p), n in sorted(((k, v) for k, v in total.confusion.items() if k[0] != k[1]), key=lambda kv: -kv[1])[:6]:
        typer.echo(f"  {t:>6s} -> {p:<6s} {n}")


@app.command(name="eval-keys")
def eval_keys(
    sources: Annotated[list[Path], typer.Argument(exists=True, help="Folders or files with the referenced tracks")],
    refs: Annotated[Path, typer.Option(help="Reference keys: {track file stem: {key, alt}}")] = Path("labels/keys.json"),
) -> None:
    """Compare key-detection methods against reference keys."""
    import json

    from autodj.analysis import analyze_file
    from autodj.analysis.key import essentia_available
    from autodj.key_eval import METHODS, detect, summarize

    reference = {k: v for k, v in json.loads(refs.read_text()).items() if not k.startswith("_")}
    methods = [m for m in METHODS if essentia_available() or not m.startswith("essentia")]
    rows = []
    typer.echo(f"{'track':34s} {'ref':>4s}  " + "  ".join(f"{m:>15s}" for m in methods))
    for f in _audio_files(sources):
        if f.stem not in reference:
            continue
        a = analyze_file(f)
        row = {"name": f.stem, **reference[f.stem]}
        cells = []
        for m in methods:
            code, conf = detect(a, m)
            row[m] = (code, conf)
            mark = "=" if code == row["key"] else "~" if code in row["alt"] else " "
            cells.append(f"{code:>4s}{mark} ({conf:.2f})".rjust(15))
        rows.append(row)
        typer.echo(f"{f.stem[:34]:34s} {row['key']:>4s}  " + "  ".join(cells))
    if not rows:
        raise typer.BadParameter("none of the tracks have reference keys")
    typer.echo(f"\n{'method':15s} {'exact':>6s} {'any source':>11s} {'mixable (<=1 step)':>19s} {'MIREX':>6s}")
    for m in methods:
        s = summarize(rows, m)
        typer.echo(f"{m:15s} {s['exact']:6.0%} {s['any_source']:11.0%} {s['compatible']:19.0%} {s['mirex']:6.2f}")
    typer.echo("\n= matches the reference, ~ matches a minority source; (confidence)")
