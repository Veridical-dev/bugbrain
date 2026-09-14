#!/usr/bin/env python3
"""Reconstruct BugBrain's public benchmark diffs from pinned Git commits."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


CASES = (
    (
        "mapt_860_pulumi_secret_state",
        "https://github.com/redhat-developer/mapt.git",
        "75a46d45655dd06acede7d3ceec2d62baf8b5abb",
        "08967c5da11db8df1d0bf33835da509e9903142c",
    ),
    (
        "formbricks_8588_interaction_parity",
        "https://github.com/formbricks/formbricks.git",
        "0ed9568c5c5ddbba116387bba85599e216fb9e73",
        "6a2646be5dd22b81726a05ab5e756db2b5c2dbf9",
    ),
    (
        "grafana_128331_cloud_role_login_regression",
        "https://github.com/grafana/grafana.git",
        "fd27c06870d19511be9e31fd1b90b8cc76517dff",
        "5562789be525a682634bc3505c6d8f0448f30119",
    ),
    (
        "openclaw_114519_exec_auto_review_bypass",
        "https://github.com/openclaw/openclaw.git",
        "981b4cc3feb198868bd73abcd50ba047253b565f",
        "a48ddf854448d560c837b5df593cde973ccf340b",
    ),
)


def run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def fetch_case(work: Path, key: str, url: str, base: str, head: str) -> str:
    checkout = work / key
    run("git", "init", "-q", str(checkout))
    run("git", "remote", "add", "origin", url, cwd=checkout)
    run("git", "fetch", "-q", "--depth=1", "origin", base, cwd=checkout)
    run("git", "fetch", "-q", "--depth=1", "origin", head, cwd=checkout)
    return run(
        "git",
        "diff",
        "--find-renames",
        "--unified=80",
        "--no-ext-diff",
        f"{base}..{head}",
        cwd=checkout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    cases_dir = destination / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "bugbrain.corpus/1", "cases": []}
    with tempfile.TemporaryDirectory(prefix="bugbrain-benchmark-") as temporary:
        work = Path(temporary)
        for key, url, base, head in CASES:
            print(f"fetching {key}", flush=True)
            diff = fetch_case(work, key, url, base, head)
            output = cases_dir / f"{key}.diff"
            output.write_text(diff, encoding="utf-8")
            manifest["cases"].append({"key": key, "path": f"cases/{output.name}"})
    (destination / "corpus.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(destination / "corpus.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
