"""Query the official H1 flat-task checkpoint inside the required Isaac Kit context."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Probe official H1 flat locomotion checkpoint availability.")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/flat_checkpoint_probe_v1.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    result = {}
    for task in ("Isaac-Velocity-Flat-H1-v0", "Isaac-Velocity-Rough-H1-v0"):
        result[task] = get_published_pretrained_checkpoint("rsl_rl", task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps({"result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"result": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
