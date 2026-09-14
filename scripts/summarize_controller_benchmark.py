#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ARMS = ("biological", "weight_shuffled", "degree_rewired", "observation_only")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [_round(centre - spread), _round(centre + spread)]


def exact_paired_sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def exact_task_bootstrap(deltas: list[float]) -> list[float]:
    if not deltas:
        raise ValueError("Paired bootstrap requires task deltas")
    estimates = sorted(
        statistics.fmean(deltas[index] for index in sample)
        for sample in itertools.product(range(len(deltas)), repeat=len(deltas))
    )

    def quantile(probability: float) -> float:
        position = (len(estimates) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    return [_round(quantile(0.025)), _round(quantile(0.975))]


def _mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return _round(statistics.fmean(values))


def _selected_trials(policy_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        candidates = [trial for trial in policy_report["trials"] if trial["arm"] == arm]
        if not candidates:
            raise ValueError(f"Policy report has no trials for {arm}")
        # The rule is fixed before reading held-out metrics: highest validation
        # action accuracy, then lowest seed as a deterministic tie breaker.
        selected[arm] = max(
            candidates,
            key=lambda trial: (trial["metrics"]["validation"]["action_accuracy"], -trial["seed"]),
        )
    return selected


def _live_row(run: dict[str, Any]) -> dict[str, Any]:
    steps = run["steps"]
    compliant = sum(bool(step["result"]["action_contract_compliant"]) for step in steps)
    non_patch_mutations = sum(
        step["selected_action"] != "patch"
        and step["repository_before"]["diff_sha256"]
        != step["repository_after"]["diff_sha256"]
        for step in steps
    )
    return {
        "repository_head": run["repository_before"]["head"],
        "passed": bool(run["reward"] == 1.0),
        "reward": run["reward"],
        "policy_steps": len(steps),
        "selected_actions": [step["selected_action"] for step in steps],
        "compliant_steps": compliant,
        "action_contract_compliance": _round(compliant / len(steps)),
        "non_patch_mutations": non_patch_mutations,
        "input_tokens": run["total_usage"].get("input_tokens", 0),
        "cached_input_tokens": run["total_usage"].get("cached_input_tokens", 0),
        "output_tokens": run["total_usage"].get("output_tokens", 0),
        "worker_wall_seconds": _round(
            sum(float(step["wall_seconds"] or 0.0) for step in steps)
        ),
    }


def _direct_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository_head": run["repository_before"]["head"],
        "passed": bool(run["reward"] == 1.0),
        "reward": run["reward"],
        "tool_actions": list(run["tool_actions"]),
        "input_tokens": run["usage"].get("input_tokens", 0),
        "cached_input_tokens": run["usage"].get("cached_input_tokens", 0),
        "output_tokens": run["usage"].get("output_tokens", 0),
        "worker_wall_seconds": _round(run["wall_seconds"]),
    }


def _arm_summary(rows: list[dict[str, Any]], *, controlled: bool) -> dict[str, Any]:
    passed = sum(row["passed"] for row in rows)
    summary = {
        "tasks": len(rows),
        "passed": passed,
        "pass_rate": _round(passed / len(rows)),
        "pass_rate_wilson_ci95": wilson_interval(passed, len(rows)),
        "mean_input_tokens": _mean(row["input_tokens"] for row in rows),
        "mean_output_tokens": _mean(row["output_tokens"] for row in rows),
        "mean_worker_wall_seconds": _mean(row["worker_wall_seconds"] for row in rows),
    }
    if controlled:
        steps = sum(row["policy_steps"] for row in rows)
        compliant = sum(row["compliant_steps"] for row in rows)
        summary.update(
            {
                "mean_policy_steps": _mean(row["policy_steps"] for row in rows),
                "action_contract_compliance": _round(compliant / steps),
                "non_patch_mutations": sum(row["non_patch_mutations"] for row in rows),
            }
        )
    return summary


def _paired(
    left_name: str,
    right_name: str,
    task_rows: dict[str, dict[str, dict[str, Any]]],
    task_keys: list[str],
) -> dict[str, Any]:
    deltas = [
        float(task_rows[key][left_name]["passed"]) - float(task_rows[key][right_name]["passed"])
        for key in task_keys
    ]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return {
        "left": left_name,
        "right": right_name,
        "pass_rate_delta": _round(statistics.fmean(deltas)),
        "exact_task_bootstrap_ci95": exact_task_bootstrap(deltas),
        "left_only_wins": wins,
        "right_only_wins": losses,
        "ties": len(deltas) - wins - losses,
        "two_sided_exact_sign_p": _round(exact_paired_sign_test(wins, losses)),
    }


def build_report(
    manifest_path: Path,
    policy_path: Path,
    trajectories_path: Path,
    live_root: Path,
    direct_root: Path,
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    policy_report = _load(policy_path)
    trajectories = _load(trajectories_path)
    task_keys = [task["key"] for task in manifest["tasks"] if task["split"] == "test"]
    if len(task_keys) != 6:
        raise ValueError(f"Expected six frozen test tasks, found {len(task_keys)}")
    selected = _selected_trials(policy_report)

    task_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for key in task_keys:
        direct_path = direct_root / key / "teacher" / "run.json"
        direct_run = _load(direct_path)
        direct_row = _direct_row(direct_run)
        direct_row["run_sha256"] = _digest(direct_path)
        task_rows[key] = {"direct": direct_row}
        for arm in ARMS:
            run_path = live_root / arm / key / "run" / "run.json"
            run = _load(run_path)
            receipt = run["contract"]["checkpoint_receipt"]
            expected = selected[arm]
            if receipt["arm"] != arm or receipt["seed"] != expected["seed"]:
                raise ValueError(f"{arm}/{key} did not use its validation-selected checkpoint")
            if not run["repository_before"]["head"].startswith(manifest["base_commit"]):
                raise ValueError(f"{arm}/{key} used the wrong base commit")
            live_row = _live_row(run)
            live_row["run_sha256"] = _digest(run_path)
            task_rows[key][arm] = live_row

    live_summary = {
        arm: _arm_summary([task_rows[key][arm] for key in task_keys], controlled=True)
        for arm in ARMS
    }
    direct_summary = _arm_summary(
        [task_rows[key]["direct"] for key in task_keys], controlled=False
    )
    comparisons = [
        _paired("biological", control, task_rows, task_keys)
        for control in ("weight_shuffled", "degree_rewired", "observation_only", "direct")
    ]
    biological_cost = {
        "mean_input_token_ratio": _round(
            live_summary["biological"]["mean_input_tokens"]
            / direct_summary["mean_input_tokens"]
        ),
        "mean_output_token_ratio": _round(
            live_summary["biological"]["mean_output_tokens"]
            / direct_summary["mean_output_tokens"]
        ),
        "mean_worker_wall_time_ratio": _round(
            live_summary["biological"]["mean_worker_wall_seconds"]
            / direct_summary["mean_worker_wall_seconds"]
        ),
    }
    episodes = trajectories["episodes"]
    action_counts: dict[str, int] = {action: 0 for action in trajectories["actions"]}
    for episode in episodes:
        for step in episode["steps"]:
            action_counts[step["action"]] += 1

    selected_receipts = {
        arm: {
            "seed": trial["seed"],
            "checkpoint_file": f"benchmarks/controller/checkpoints/{arm}-seed-{trial['seed']}.npz",
            "checkpoint_sha256": _load(
                live_root / arm / task_keys[0] / "run" / "run.json"
            )["contract"]["checkpoint_sha256"],
            "selection_validation_action_accuracy": trial["metrics"]["validation"][
                "action_accuracy"
            ],
            "heldout_imitation_action_accuracy": trial["metrics"]["test"]["action_accuracy"],
            "heldout_exact_trajectory_accuracy": trial["metrics"]["test"][
                "exact_trajectory_accuracy"
            ],
        }
        for arm, trial in selected.items()
    }
    return {
        "schema": "bugbrain.controller-benchmark-results/1",
        "status": "completed_real_repository_controller_experiment",
        "question": "Does FlyWire topology improve a repository-capable coding-agent controller?",
        "frozen_inputs": {
            "base_commit_ref": manifest["base_commit"],
            "base_commit": task_rows[task_keys[0]]["biological"].get(
                "repository_head", manifest["base_commit"]
            ),
            "manifest_sha256": _digest(manifest_path),
            "trajectory_sha256": _digest(trajectories_path),
            "policy_report_sha256": _digest(policy_path),
            "trajectory_episodes": len(episodes),
            "trajectory_steps": sum(len(episode["steps"]) for episode in episodes),
            "trajectory_terminal_reward_sum": sum(
                float(episode["steps"][-1]["reward"]) for episode in episodes
            ),
            "split_episodes": {
                split: sum(episode["split"] == split for episode in episodes)
                for split in ("train", "validation", "test")
            },
            "action_counts": action_counts,
        },
        "offline_imitation": policy_report["summary"],
        "live_protocol": {
            "task_keys": task_keys,
            "checkpoint_selection": (
                "highest validation action accuracy within each arm; lowest seed breaks ties"
            ),
            "selected_checkpoints": selected_receipts,
            "max_policy_steps": 12,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "worker_repository_access": "complete disposable checkout",
            "verifier": manifest["verifier"],
            "direct_baseline": "one normal Codex CLI run with full autonomy per held-out task",
        },
        "live_summary": {"direct": direct_summary, **live_summary},
        "biological_vs_direct_cost": biological_cost,
        "paired_live_comparisons": comparisons,
        "per_task": task_rows,
        "conclusion": {
            "biological_topology_advantage": "not demonstrated",
            "operational_result": (
                "The biological controller tied direct Codex at 5/6 verifier passes, with one "
                "different success and one different failure, while using more time and tokens."
            ),
            "useful_direction": (
                "Iterative repository-capable worker decomposition can rescue a direct-agent "
                "failure, but the current learned scheduler is not more reliable or efficient "
                "than direct Codex and the FlyWire-specific contribution is unsupported."
            ),
        },
        "interpretation_boundary": (
            "Offline estimates cover 30 paired seeds but only six held-out task episodes. Live "
            "rollouts use one validation-selected checkpoint and one nondeterministic Codex run "
            "per arm/task. Exact tests and intervals are reported; this is conclusive for this "
            "frozen POC, not for coding agents or connectomes in general."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the frozen controller benchmark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.manifest,
        args.policy_report,
        args.trajectories,
        args.live_root,
        args.direct_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
