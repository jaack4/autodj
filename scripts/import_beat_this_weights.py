"""Convert the community safetensors copy of beat_this `final0` into the checkpoint
format beat_this loads, for while the official host (cloud.cp.jku.at) is down.

Source: https://huggingface.co/safe-models/beat-this-final0 (not published by the authors).
Validate it against tracks with known-good beats before trusting it; see docs/M0-findings.md.

    uv run python scripts/import_beat_this_weights.py ~/.cache/autodj/hf-beat-this-final0
"""

import json
import sys
from pathlib import Path

import torch
from beat_this.model.beat_tracker import BeatThis
from safetensors.torch import load_file

EXPECTED_SHA256 = "f758c77ba5d862cad3718a95705964e9af55298b1fb3544c62c9d79184ea0f2e"


def main(src: Path) -> None:
    import hashlib

    weights = src / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"sha256 mismatch: {digest}")
    config = json.loads((src / "config.json").read_text())
    state = load_file(weights)

    model = BeatThis(**config)
    # All attention layers share one RotaryEmbedding; safetensors stores that tensor once.
    # Its freqs are fixed at init (not learned), so check the stored copy and fill in the aliases.
    expected = model.state_dict()
    for key in set(expected) - set(state):
        if not key.endswith("rotary_embed.freqs"):
            sys.exit(f"missing weight: {key}")
        state[key] = expected[key]
    for key in (k for k in state if k.endswith("rotary_embed.freqs")):
        if not torch.equal(state[key], expected[key]):
            sys.exit(f"unexpected rotary freqs: {key}")
    model.load_state_dict(state, strict=True)  # fails on any unexpected or misshaped tensor
    n_params = sum(p.numel() for p in model.parameters())

    dest = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}, "hyper_parameters": config}, dest)
    print(f"ok: {len(state)} tensors, {n_params / 1e6:.1f}M params -> {dest}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
