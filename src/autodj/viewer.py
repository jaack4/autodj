"""A self-contained HTML page for listening to a rendered mix: track lanes with sections,
transitions you can click to jump to, and the planner's reasons."""

import html
import json
from pathlib import Path

from autodj.models import MixPlan
from autodj.render import Deck, TransitionWindow

SECTION_COLORS = {
    "intro": "#5b7083", "build": "#d9a441", "drop": "#d9534f",
    "break": "#5cb85c", "body": "#8a8f6a", "outro": "#5b7083",
}
PREROLL_BARS = 4  # jump this far ahead of a transition so you hear it coming


def _data(plan: MixPlan, decks: list[Deck], windows: list[TransitionWindow], checks: list[dict]) -> dict:
    tracks = []
    for d, t in zip(decks, plan.tracks):
        start, end = d.offset + d.play_from, d.offset + d.play_to
        sections = []
        for s in d.analysis.sections:
            a = max(start, d.offset + d.bar_time(s.start_bar))
            b = min(end, d.offset + d.bar_time(s.end_bar))
            if b > a:
                sections.append({"label": s.label, "start": a, "end": b})
        name = " - ".join(x for x in (t.artist, t.title) if x) or Path(d.analysis.path).stem
        tracks.append({"name": name, "start": start, "end": end, "bpm": round(d.native_bpm, 1),
                       "key": t.key or d.analysis.key, "sections": sections})
    transitions = []
    for k, (w, check) in enumerate(zip(windows, checks)):  # window k runs from deck k into deck k + 1
        tr = w.transition
        transitions.append({
            "type": tr.type, "start": w.start, "end": max(w.end, w.start + 0.5),
            "seek": max(0.0, w.start - PREROLL_BARS * 4 * w.beat),
            "from": tracks[k]["name"], "to": tracks[k + 1]["name"],
            "bars": f"bar {tr.out_bar} → bar {tr.in_bar}" + (f", {tr.overlap_bars} bars" if tr.overlap_bars else ""),
            "notes": tr.notes, "check": check.get("verdict", ""), "nudge": f"{w.nudge_ms:+.0f} ms ({w.nudge_source})",
        })
    return {"name": plan.name, "duration": max(t["end"] for t in tracks), "tracks": tracks,
            "transitions": transitions, "colors": SECTION_COLORS}


def write_viewer(path: Path, wav: Path, plan: MixPlan, decks: list[Deck], windows: list[TransitionWindow],
                 checks: list[dict]) -> Path:
    data = json.dumps(_data(plan, decks, windows, checks))
    page = TEMPLATE.replace("__TITLE__", html.escape(plan.name)).replace("__WAV__", html.escape(wav.name))
    path.write_text(page.replace("__DATA__", data.replace("</", "<\\/")))
    return path


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__ · autodj</title>
<style>
  :root { --bg:#111418; --panel:#1a1f25; --ink:#e7e9ec; --dim:#8b949e; --accent:#7aa2f7; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px 32px; background:var(--bg); color:var(--ink);
         font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  h1 { font-size:20px; margin:0 0 4px; } .sub { color:var(--dim); margin-bottom:16px; }
  audio { width:100%; margin-bottom:16px; }
  #timeline { background:var(--panel); border-radius:8px; padding:12px 12px 28px; cursor:pointer; }
  #inner { position:relative; }
  .lane { position:relative; height:34px; margin:4px 0; }
  .track { position:absolute; top:0; height:100%; border-radius:4px; overflow:hidden; background:#2a313a; }
  .sec { position:absolute; top:0; height:100%; opacity:.85; }
  .tname { position:absolute; left:6px; top:8px; font-size:12px; white-space:nowrap; text-shadow:0 1px 2px #000; pointer-events:none; }
  .tr { position:absolute; top:0; bottom:0; background:rgba(122,162,247,.18); border-left:2px solid var(--accent);
        border-right:1px solid rgba(122,162,247,.5); }
  .tr span { position:absolute; bottom:-18px; left:2px; font-size:11px; color:var(--accent); white-space:nowrap; }
  #head { position:absolute; top:0; bottom:0; width:2px; background:#fff; pointer-events:none; }
  .legend { margin:10px 0 20px; color:var(--dim); font-size:12px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 12px; vertical-align:-1px; }
  table { width:100%; border-collapse:collapse; background:var(--panel); border-radius:8px; overflow:hidden; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #262c33; vertical-align:top; }
  th { color:var(--dim); font-weight:500; font-size:12px; }
  tr.row { cursor:pointer; } tr.row:hover { background:#20262d; }
  .notes { color:var(--dim); font-size:12px; } .time { font-variant-numeric:tabular-nums; color:var(--accent); }
</style></head><body>
<h1>__TITLE__</h1><div class="sub" id="sub"></div>
<audio id="audio" controls src="__WAV__"></audio>
<div id="timeline"><div id="inner"><div id="head"></div></div></div>
<div class="legend" id="legend"></div>
<table><thead><tr><th>Time</th><th>Transition</th><th>From → to</th><th>Planner's reasons</th><th>Check</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
const D = __DATA__;
const audio = document.getElementById("audio"), tl = document.getElementById("timeline"),
      inner = document.getElementById("inner"), head = document.getElementById("head");
const pct = t => (100 * t / D.duration) + "%";
const clock = t => Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0");
const el = (tag, cls, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; parent.appendChild(e); return e; };
document.getElementById("sub").textContent = D.tracks.length + " tracks · " + clock(D.duration) + " · click anywhere on the timeline to jump";
D.tracks.forEach(t => {
  const lane = el("div", "lane", inner), box = el("div", "track", lane);
  box.style.left = pct(t.start); box.style.width = pct(t.end - t.start);
  t.sections.forEach(s => { const e = el("div", "sec", box); e.style.left = (100 * (s.start - t.start) / (t.end - t.start)) + "%";
    e.style.width = (100 * (s.end - s.start) / (t.end - t.start)) + "%"; e.style.background = D.colors[s.label] || "#444"; e.title = s.label; });
  el("div", "tname", box).textContent = t.name + "  ·  " + t.bpm + " BPM · " + t.key;
});
D.transitions.forEach(x => { const e = el("div", "tr", inner); e.style.left = pct(x.start); e.style.width = pct(x.end - x.start);
  el("span", "", e).textContent = x.type; });
Object.entries(D.colors).forEach(([k, c]) => { if (k === "outro") return; const l = document.getElementById("legend");
  l.insertAdjacentHTML("beforeend", `<i style="background:${c}"></i>${k}`); });
const rows = document.getElementById("rows");
D.transitions.forEach(x => {
  const r = el("tr", "row", rows);
  r.innerHTML = `<td class="time">${clock(x.start)}</td><td>${x.type}<div class="notes">${x.bars}</div></td>
    <td>${x.from}<div class="notes">→ ${x.to}</div></td><td class="notes"></td>
    <td class="notes">${x.check}<br>nudge ${x.nudge}</td>`;
  r.children[3].textContent = x.notes.join(" · ");
  r.onclick = () => { audio.currentTime = x.seek; audio.play(); };
});
tl.onclick = e => { const b = inner.getBoundingClientRect();
  audio.currentTime = Math.min(D.duration, Math.max(0, D.duration * (e.clientX - b.left) / b.width)); audio.play(); };
const tick = () => { head.style.left = pct(audio.currentTime); requestAnimationFrame(tick); };
tick();
</script></body></html>
"""
