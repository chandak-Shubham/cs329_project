"""Refresh the standalone demo's embedded research data; no runtime dependencies.

Run after regenerating research artifacts: python visualizer/export_web_data.py
Use --check to verify that the committed HTML matches the checkpoint and results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[1]


def build_payload() -> dict:
    checkpoint = ROOT / "results/checkpoints/simple_mlp_seed_42.pt"
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from its research sidecar")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    benchmarks = json.loads((ROOT / "results/phase4_benchmarks.json").read_text(encoding="utf-8"))
    return {
        "neural": {
            "seed": metadata["seed"],
            "environmentSteps": metadata["environment_steps"],
            "sha256": digest,
            "layers": [
                {"weight": state[f"{key}.weight"].tolist(), "bias": state[f"{key}.bias"].tolist()}
                for key in ("encoder.0", "encoder.2", "actor_head")
            ],
        },
        "benchmarks": [
            {
                "model": name,
                "survival": row["survival_rate"],
                "defense": row["successful_defense_rate"],
                "attacks": row["faced_invasion_count"],
            }
            for name, row in benchmarks["benchmarks"].items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "index.html"
    html = target.read_text(encoding="utf-8")
    payload = json.dumps(build_payload(), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    pattern = r'(<script id="research-data" type="application/json">).*?(</script>)'
    updated, count = re.subn(pattern, lambda match: match[1] + payload + match[2], html, flags=re.S)
    if count != 1:
        raise ValueError("Expected exactly one research-data script in index.html")
    if args.check:
        if updated != html:
            raise SystemExit("Embedded web data is stale. Run python visualizer/export_web_data.py")
        print("Embedded checkpoint and all 7 benchmark rows match research artifacts.")
    else:
        target.write_text(updated, encoding="utf-8", newline="\n")
        print(f"Embedded checkpoint and benchmarks in {target.name} ({len(updated.encode('utf-8')):,} bytes).")


if __name__ == "__main__":
    main()
