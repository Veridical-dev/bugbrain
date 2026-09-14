#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "controller" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect persist-once controller teacher runs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    episodes = []
    actions = None
    for task in manifest["tasks"]:
        candidates = [root / task["key"] / "teacher" / "trajectory.json" for root in args.root]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise SystemExit(f"Missing teacher trajectory for {task['key']}: {candidates}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if actions is None:
            actions = payload["actions"]
        elif payload["actions"] != actions:
            raise SystemExit(f"Action vocabulary mismatch in {source}")
        rows = payload.get("episodes") or []
        if len(rows) != 1:
            raise SystemExit(f"Expected exactly one episode in {source}")
        episode = rows[0]
        if episode["key"] != task["key"] or episode["split"] != task["split"]:
            raise SystemExit(f"Task contract mismatch in {source}")
        episode["metadata"]["trajectory_source"] = str(source)
        episodes.append(episode)

    output = {
        "schema": "bugbrain.trajectories/1",
        "actions": actions,
        "metadata": {
            "benchmark": manifest["schema"],
            "base_commit": manifest["base_commit"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "episodes": len(episodes),
            "terminal_reward_sum": sum(float(row["steps"][-1]["reward"]) for row in episodes),
        },
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

