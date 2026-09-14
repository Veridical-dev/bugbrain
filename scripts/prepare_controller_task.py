#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "controller" / "manifest.json"


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one isolated BugBrain controller task")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task = next((row for row in manifest["tasks"] if row["key"] == args.task), None)
    if task is None:
        raise SystemExit(f"Unknown task: {args.task}")
    destination = args.out.resolve()
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    clone = _run(["git", "clone", "--quiet", "--local", "--no-hardlinks", str(args.source), str(destination)])
    if clone.returncode:
        raise SystemExit(clone.stderr.strip() or "git clone failed")
    checkout = _run(
        ["git", "checkout", "--quiet", "--detach", str(manifest["base_commit"])],
        cwd=destination,
    )
    if checkout.returncode:
        raise SystemExit(checkout.stderr.strip() or "git checkout failed")
    mutation_path = (args.manifest.parent / task["mutation"]).resolve()
    applied = _run(["git", "apply", str(mutation_path)], cwd=destination)
    if applied.returncode:
        raise SystemExit(applied.stderr.strip() or "git apply failed")

    verifier = [sys.executable if part == "{python}" else str(part) for part in manifest["verifier"]]
    verifier_path = destination.parent / "verifier.json"
    verifier_path.write_text(json.dumps(verifier, indent=2) + "\n", encoding="utf-8")
    baseline = _run(verifier, cwd=destination)
    if baseline.returncode == 0:
        raise SystemExit(f"Mutation {task['key']} did not make the verifier fail")
    receipt = {
        "schema": "bugbrain.prepared-controller-task/1",
        "key": task["key"],
        "split": task["split"],
        "goal": task["goal"],
        "base_commit": manifest["base_commit"],
        "repository": str(destination),
        "mutation": str(mutation_path),
        "verifier_json": str(verifier_path),
        "baseline_returncode": baseline.returncode,
        "baseline_stdout_tail": baseline.stdout[-2_000:],
        "baseline_stderr_tail": baseline.stderr[-2_000:],
    }
    receipt_path = destination.parent / "task.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

