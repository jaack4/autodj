"""`autodj label`: a small local web app for marking where sections start, by ear.

Labels are the ground truth for `autodj eval-sections`. Served on 127.0.0.1 only; it can only
read the tracks it was started with and only writes into the labels folder.
"""

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import librosa
import numpy as np

from autodj.audio import load as load_audio
from autodj.models import Analysis

LABELS_DIR = Path("labels")
LABELS = ("intro", "build", "drop", "break", "body", "outro")
ENVELOPE_POINTS = 2400


def labels_path(a: Analysis, directory: Path = LABELS_DIR) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(a.path).stem).strip("-").lower()
    return directory / f"{slug}.json"


def load_labels(a: Analysis, directory: Path = LABELS_DIR) -> list[dict]:
    path = labels_path(a, directory)
    return json.loads(path.read_text())["sections"] if path.exists() else []


def save_labels(a: Analysis, sections: list[dict], directory: Path = LABELS_DIR) -> Path:
    clean = sorted(({"start": round(float(s["start"]), 3), "label": s["label"]} for s in sections
                    if s.get("label") in LABELS), key=lambda s: s["start"])
    path = labels_path(a, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"path": a.path, "sha256": a.sha256, "duration": a.duration,
                                "sections": clean}, indent=1))
    return path


def _envelope(path: str) -> list[float]:
    y, sr = load_audio(path, sr=11025, mono=True)
    hop = max(1, len(y) // ENVELOPE_POINTS)
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    return np.round(rms / (rms.max() + 1e-9), 3).tolist()


class _Handler(BaseHTTPRequestHandler):
    analyses: list[Analysis] = []
    labels_dir: Path = LABELS_DIR
    envelopes: dict[int, list[float]] = {}
    lock = threading.Lock()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _track(self, part: str) -> tuple[int, Analysis] | None:
        if part.isdigit() and int(part) < len(self.analyses):
            return int(part), self.analyses[int(part)]
        self._send(b"not found", "text/plain", 404)
        return None

    def do_GET(self) -> None:
        parts = self.path.strip("/").split("/")
        if parts == [""]:
            names = [Path(a.path).stem for a in self.analyses]
            page = PAGE.replace("__TRACKS__", json.dumps(names).replace("</", "<\\/"))
            self._send(page.encode(), "text/html; charset=utf-8")
        elif parts[0] == "track" and len(parts) == 2 and (t := self._track(parts[1])):
            i, a = t
            with self.lock:
                if i not in self.envelopes:
                    self.envelopes[i] = _envelope(a.path)
            data = {"name": Path(a.path).stem, "duration": a.duration, "bpm": a.bpm, "downbeats": a.downbeats,
                    "envelope": self.envelopes[i], "labels": load_labels(a, self.labels_dir), "choices": LABELS}
            self._send(json.dumps(data).encode(), "application/json")
        elif parts[0] == "audio" and len(parts) == 2 and (t := self._track(parts[1])):
            self._send_audio(Path(t[1].path))
        else:
            self._send(b"not found", "text/plain", 404)

    def _send_audio(self, path: Path) -> None:
        """Serve the file with HTTP range support, which browsers need to seek."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not rng:
            self._send(path.read_bytes(), ctype, extra={"Accept-Ranges": "bytes"})
            return
        start = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2)))
        end = int(rng.group(2)) if rng.group(1) and rng.group(2) else size - 1
        end = min(end, size - 1)
        with path.open("rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self._send(body, ctype, 206, {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}"})

    def do_POST(self) -> None:
        parts = self.path.strip("/").split("/")
        if parts[0] == "labels" and len(parts) == 2 and (t := self._track(parts[1])):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            saved = save_labels(t[1], body.get("sections", []), self.labels_dir)
            self._send(json.dumps({"saved": str(saved)}).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)


def serve(analyses: list[Analysis], port: int = 8765, labels_dir: Path = LABELS_DIR) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"analyses": analyses, "labels_dir": labels_dir, "envelopes": {}})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>autodj · label sections</title>
<style>
  :root { --bg:#111418; --panel:#1a1f25; --ink:#e7e9ec; --dim:#8b949e; --accent:#7aa2f7; }
  body { margin:0; padding:20px 28px; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, sans-serif; }
  h1 { font-size:18px; margin:0 0 12px; } select, button { font:inherit; }
  select { background:var(--panel); color:var(--ink); border:1px solid #333a42; border-radius:6px; padding:6px 8px; }
  audio { width:100%; margin:12px 0; }
  canvas { width:100%; height:180px; background:var(--panel); border-radius:8px; cursor:pointer; display:block; }
  .bar { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; align-items:center; }
  .bar button { background:#262c33; color:var(--ink); border:1px solid #333a42; border-radius:6px; padding:6px 12px; cursor:pointer; }
  .bar button:hover { border-color:var(--accent); }
  .help { color:var(--dim); font-size:12px; } #status { color:var(--dim); margin-left:auto; }
  table { border-collapse:collapse; margin-top:8px; } td { padding:3px 12px 3px 0; font-variant-numeric:tabular-nums; }
  td button { background:none; border:none; color:#f0a35e; cursor:pointer; }
</style></head><body>
<h1>Label sections <select id="pick"></select></h1>
<div class="help">Listen, and at the start of each section press its number (it snaps to the nearest bar).
  Space: play/pause · ←/→: 1 bar · Shift+←/→: 8 bars · Backspace: delete the mark before the playhead ·
  click the waveform to jump. Changes save automatically.</div>
<audio id="audio" controls preload="auto"></audio>
<canvas id="wave" width="2400" height="360"></canvas>
<div class="bar" id="buttons"><label><input type="checkbox" id="snap" checked> snap to bars</label><span id="status"></span></div>
<table id="marks"></table>
<script>
const TRACKS = __TRACKS__;
const COLORS = {intro:"#5b7083", build:"#d9a441", drop:"#d9534f", break:"#5cb85c", body:"#8a8f6a", outro:"#3f5568"};
const audio = document.getElementById("audio"), canvas = document.getElementById("wave"), ctx = canvas.getContext("2d");
const pick = document.getElementById("pick"), status = document.getElementById("status");
let T = null, idx = 0, saveTimer = null;
TRACKS.forEach((n, i) => pick.add(new Option(n, i)));
pick.onchange = () => open(+pick.value);

async function open(i) {
  idx = i; T = await (await fetch("/track/" + i)).json();
  audio.src = "/audio/" + i;
  const buttons = document.getElementById("buttons");
  buttons.querySelectorAll("button").forEach(b => b.remove());
  const snapLabel = document.getElementById("snap").parentElement;
  T.choices.forEach((c, k) => { const b = document.createElement("button"); b.textContent = (k + 1) + " " + c;
    b.onclick = () => mark(c); buttons.insertBefore(b, snapLabel); });
  status.textContent = T.labels.length ? "loaded " + T.labels.length + " marks" : "no marks yet";
  table(); draw();
}
const barLen = () => T.downbeats.length > 1 ? (T.downbeats[T.downbeats.length - 1] - T.downbeats[0]) / (T.downbeats.length - 1) : 2;
function barAt(t) { let best = 0; T.downbeats.forEach((d, k) => { if (Math.abs(d - t) < Math.abs(T.downbeats[best] - t)) best = k; }); return best; }
function snap(t) { return document.getElementById("snap").checked ? T.downbeats[barAt(t)] : t; }
function mark(label) {
  const t = snap(audio.currentTime);
  T.labels = T.labels.filter(m => Math.abs(m.start - t) > 0.05).concat([{start: t, label}]).sort((a, b) => a.start - b.start);
  changed();
}
function changed() {
  table(); draw(); status.textContent = "saving…";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const r = await fetch("/labels/" + idx, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sections: T.labels})});
    status.textContent = r.ok ? "saved" : "save failed";
  }, 300);
}
function table() {
  const tb = document.getElementById("marks"); tb.innerHTML = "";
  T.labels.forEach((m, k) => { const r = tb.insertRow();
    r.innerHTML = `<td>${Math.floor(m.start / 60)}:${(m.start % 60).toFixed(1).padStart(4, "0")}</td><td>bar ${barAt(m.start)}</td>
      <td style="color:${COLORS[m.label]}">${m.label}</td><td><button>remove</button></td>`;
    r.querySelector("button").onclick = () => { T.labels.splice(k, 1); changed(); }; });
}
function draw() {
  if (!T) return;
  const W = canvas.width, H = canvas.height, x = t => W * t / T.duration;
  ctx.clearRect(0, 0, W, H);
  T.labels.forEach((m, k) => { const end = k + 1 < T.labels.length ? T.labels[k + 1].start : T.duration;
    ctx.fillStyle = COLORS[m.label] + "55"; ctx.fillRect(x(m.start), 0, x(end) - x(m.start), H);
    ctx.fillStyle = COLORS[m.label]; ctx.fillRect(x(m.start), 0, 3, H);
    ctx.font = "22px sans-serif"; ctx.fillText(m.label, x(m.start) + 6, 26); });
  ctx.fillStyle = "#c9d1d9";
  T.envelope.forEach((v, k) => { const px = W * k / T.envelope.length; ctx.fillRect(px, H / 2 - v * H * 0.42, 1, v * H * 0.84); });
  T.downbeats.forEach((d, k) => { ctx.fillStyle = k % 8 === 0 ? "#ffffff55" : "#ffffff18"; ctx.fillRect(x(d), H - (k % 8 === 0 ? 24 : 10), 1, 24);
    if (k % 8 === 0) { ctx.fillStyle = "#8b949e"; ctx.font = "16px sans-serif"; ctx.fillText(k, x(d) + 3, H - 6); } });
  ctx.fillStyle = "#7aa2f7"; ctx.fillRect(x(audio.currentTime), 0, 3, H);
}
canvas.onclick = e => { const b = canvas.getBoundingClientRect(); audio.currentTime = T.duration * (e.clientX - b.left) / b.width; draw(); };
document.addEventListener("keydown", e => {
  if (!T || e.target.tagName === "SELECT") return;
  const n = parseInt(e.key);
  if (n >= 1 && n <= T.choices.length) { mark(T.choices[n - 1]); e.preventDefault(); }
  else if (e.key === " ") { audio.paused ? audio.play() : audio.pause(); e.preventDefault(); }
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    audio.currentTime = Math.max(0, audio.currentTime + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 8 : 1) * barLen()); e.preventDefault(); }
  else if (e.key === "Backspace") { const before = T.labels.filter(m => m.start <= audio.currentTime + 0.05);
    if (before.length) { T.labels.splice(T.labels.indexOf(before[before.length - 1]), 1); changed(); } e.preventDefault(); }
});
const loop = () => { draw(); requestAnimationFrame(loop); }; loop();
open(0);
</script></body></html>
"""
