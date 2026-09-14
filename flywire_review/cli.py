from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .connectome import build_cache, load_annotations, load_cache
from .diff_parser import parse_review_input, parse_unified_diff
from .download import download_data
from .router import coassignment_jaccard, route_units


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_CACHE = ROOT / "cache" / "flywire-v783-s5.csr"
DEFAULT_CORPUS = ROOT / "examples" / "corpus.json"


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", type=Path, default=DEFAULT_DATA / "neuron_annotations_flywire_v2.1.0.tsv.gz")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)


def _add_route_options(parser: argparse.ArgumentParser) -> None:
    _add_common_paths(parser)
    parser.add_argument("--diff", type=Path, help="Unified diff or frozen-input JSON; omit to read stdin")
    parser.add_argument("--bundles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=783)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--active-width", type=int, default=2048)
    parser.add_argument("--restart", type=float, default=0.15)
    parser.add_argument("--context-budget", type=int, default=30_000, help="Maximum diff characters per reviewer prompt")
    parser.add_argument("--signed", action="store_true", help="Apply annotation-derived inhibitory signs")
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout")


def _add_live_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("review", "implement"), default="implement")
    parser.add_argument("--codex-cli", default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--verifier-json", type=Path)
    parser.add_argument("--verifier-timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--use-user-config",
        action="store_true",
        help="Enable the operator's normal Codex configuration and MCP servers",
    )


def _read_units(path: Path | None):
    if path:
        return parse_review_input(path)
    return parse_unified_diff(sys.stdin.read())


def _write_json(payload: object, out: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(out)
    else:
        print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bugbrain",
        description="Route and train software-agent controllers through a fruit-fly connectome.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Fetch and checksum the CC BY 4.0 data release")
    download_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    download_parser.add_argument("--force", action="store_true")

    build_parser = subparsers.add_parser("build", help="Build a compact thresholded CSR graph cache")
    build_parser.add_argument("--connections", type=Path, default=DEFAULT_DATA / "connections_biological.csv.gz")
    build_parser.add_argument("--annotations", type=Path, default=DEFAULT_DATA / "neuron_annotations_flywire_v2.1.0.tsv.gz")
    build_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    build_parser.add_argument("--min-synapses", type=int, default=5)

    route_parser = subparsers.add_parser("route", help="Create reviewer bundles for one experimental arm")
    _add_route_options(route_parser)
    route_parser.add_argument("--arm", choices=("biological", "shuffled", "hash"), default="biological")

    compare_parser = subparsers.add_parser("compare", help="Generate matched biological, shuffled, and hash arms")
    _add_route_options(compare_parser)

    reservoir_parser = subparsers.add_parser(
        "reservoir-poc",
        help="Train matched FlyWire reservoir arms and route one held-out review",
    )
    _add_common_paths(reservoir_parser)
    reservoir_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    reservoir_parser.add_argument(
        "--holdout", default="formbricks_8588_interaction_parity"
    )
    reservoir_parser.add_argument("--neurons", type=int, default=512)
    reservoir_parser.add_argument("--features", type=int, default=96)
    reservoir_parser.add_argument("--passes", type=int, default=3)
    reservoir_parser.add_argument("--leak", type=float, default=0.9)
    reservoir_parser.add_argument("--input-scale", type=float, default=0.7)
    reservoir_parser.add_argument("--spectral-radius", type=float, default=0.99)
    reservoir_parser.add_argument("--ridge", type=float, default=0.01)
    reservoir_parser.add_argument("--seed", type=int, default=1714)
    reservoir_parser.add_argument("--bundles", type=int, default=8)
    reservoir_parser.add_argument("--reservoir-weight", type=float, default=0.25)
    reservoir_parser.add_argument("--context-budget", type=int, default=30_000)
    reservoir_parser.add_argument(
        "--truth-evidence",
        type=Path,
        help="Optional post-routing evidence file used only for colocation grading",
    )
    reservoir_parser.add_argument("--out", type=Path)

    benchmark_parser = subparsers.add_parser(
        "reservoir-benchmark",
        help="Run paired multi-seed leave-one-project-out reservoir checks",
    )
    _add_common_paths(benchmark_parser)
    benchmark_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    benchmark_parser.add_argument("--neurons", type=int, default=512)
    benchmark_parser.add_argument("--features", type=int, default=96)
    benchmark_parser.add_argument("--passes", type=int, default=3)
    benchmark_parser.add_argument("--leak", type=float, default=0.9)
    benchmark_parser.add_argument("--input-scale", type=float, default=0.7)
    benchmark_parser.add_argument("--spectral-radius", type=float, default=0.99)
    benchmark_parser.add_argument("--ridge", type=float, default=0.01)
    benchmark_parser.add_argument("--seed", type=int, default=1714)
    benchmark_parser.add_argument(
        "--seeds", type=int, default=10, help="Number of consecutive paired random seeds"
    )
    benchmark_parser.add_argument("--out", type=Path)

    standalone_parser = subparsers.add_parser(
        "standalone-review",
        help="Run one routed arm through agentic Codex CLI reviewers",
    )
    standalone_parser.add_argument("--routing", type=Path, required=True)
    standalone_parser.add_argument("--arm", default="biological")
    standalone_parser.add_argument("--out-dir", type=Path, required=True)
    standalone_parser.add_argument(
        "--repo", type=Path, required=True, help="Read-only exact-head Git checkout"
    )
    standalone_parser.add_argument("--base", required=True, help="Exact PR base commit")
    standalone_parser.add_argument("--head", required=True, help="Exact PR head commit")
    standalone_parser.add_argument(
        "--codex-cli", default="codex"
    )
    standalone_parser.add_argument("--model", default="gpt-5.6-sol")
    standalone_parser.add_argument("--reasoning-effort", default="low")
    standalone_parser.add_argument("--timeout-seconds", type=int, default=600)
    standalone_parser.add_argument("--max-reviewers", type=int, default=0)
    standalone_parser.add_argument(
        "--use-user-config",
        action="store_true",
        help="Enable the operator's normal Codex configuration and MCP servers",
    )

    competitor_parser = subparsers.add_parser(
        "standalone-compare",
        help="Score source-graded standalone findings against same-commit competitor evidence",
    )
    competitor_parser.add_argument("--review", type=Path, required=True)
    competitor_parser.add_argument("--truth-register", type=Path, required=True)
    competitor_parser.add_argument("--competitor-evidence", type=Path, required=True)
    competitor_parser.add_argument("--grades", type=Path, required=True)
    competitor_parser.add_argument("--out", type=Path, required=True)

    trace_parser = subparsers.add_parser(
        "trace-import",
        help="Convert `codex exec --json` events into one trainable agent trajectory",
    )
    trace_parser.add_argument("--jsonl", type=Path, required=True)
    trace_parser.add_argument("--key", required=True)
    trace_parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    trace_parser.add_argument("--goal", required=True)
    trace_parser.add_argument(
        "--reward",
        type=float,
        required=True,
        help="Terminal verifier reward recorded for this trajectory",
    )
    trace_parser.add_argument("--out", type=Path, required=True)
    trace_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing compatible trajectory document",
    )

    policy_parser = subparsers.add_parser(
        "policy-train",
        help="Train paired connectome software-agent policies from Codex trajectories",
    )
    _add_common_paths(policy_parser)
    policy_parser.add_argument(
        "--trajectories", type=Path, required=True, help="bugbrain.trajectories/1 JSON"
    )
    policy_parser.add_argument(
        "--neurons",
        type=int,
        default=2_048,
        help="Highest-degree induced core size; 0 retains the complete graph",
    )
    policy_parser.add_argument("--features", type=int, default=128)
    policy_parser.add_argument("--readout-width", type=int, default=32)
    policy_parser.add_argument("--epochs", type=int, default=80)
    policy_parser.add_argument("--learning-rate", type=float, default=0.015)
    policy_parser.add_argument("--reward-scale", type=float, default=0.35)
    policy_parser.add_argument("--seed", type=int, default=2305)
    policy_parser.add_argument(
        "--seeds", type=int, default=3, help="Number of consecutive paired seeds"
    )
    policy_parser.add_argument(
        "--unsigned", action="store_true", help="Ignore annotation-derived transmitter signs"
    )
    policy_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Save one reusable biological-policy checkpoint per seed",
    )
    policy_parser.add_argument("--out", type=Path, required=True)

    predict_parser = subparsers.add_parser(
        "policy-predict",
        help="Load any trained policy arm and choose actions for an observation history",
    )
    _add_common_paths(predict_parser)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="JSON array of observation objects, or an object with an observations array",
    )
    predict_parser.add_argument("--out", type=Path)

    run_parser = subparsers.add_parser(
        "policy-run",
        help="Let a trained policy control bounded actions of a repository-capable Codex worker",
    )
    _add_common_paths(run_parser)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--max-steps", type=int, default=8)
    _add_live_agent_options(run_parser)

    direct_parser = subparsers.add_parser(
        "direct-run",
        help="Run the same Codex worker directly as a standard agentic baseline",
    )
    _add_live_agent_options(direct_parser)
    return parser


def _route_from_args(args: argparse.Namespace, arm: str):
    units = _read_units(args.diff)
    if not units:
        raise SystemExit("No changed hunks found in the supplied unified diff")
    graph = load_cache(args.cache)
    annotations = load_annotations(args.annotations)
    if len(annotations) != graph.node_count:
        raise SystemExit("Annotation order/count differs from the graph cache; rebuild the cache")
    return route_units(
        units,
        graph,
        annotations,
        arm=arm,
        bundle_count=args.bundles,
        seed=args.seed,
        steps=args.steps,
        active_width=args.active_width,
        restart=args.restart,
        signed=args.signed,
        context_budget=args.context_budget,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        for path in download_data(args.data_dir, force=args.force):
            print(path)
        return 0
    if args.command == "build":
        metadata = build_cache(
            args.connections,
            args.annotations,
            args.cache,
            min_synapses=args.min_synapses,
        )
        _write_json(metadata, None)
        return 0
    if args.command == "route":
        result = _route_from_args(args, args.arm)
        _write_json(result.to_dict(), args.out)
        return 0
    if args.command == "compare":
        units = _read_units(args.diff)
        if not units:
            raise SystemExit("No changed hunks found in the supplied unified diff")
        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        if len(annotations) != graph.node_count:
            raise SystemExit("Annotation order/count differs from the graph cache; rebuild the cache")
        results = {
            arm: route_units(
                units,
                graph,
                annotations,
                arm=arm,
                bundle_count=args.bundles,
                seed=args.seed,
                steps=args.steps,
                active_width=args.active_width,
                restart=args.restart,
                signed=args.signed,
                context_budget=args.context_budget,
            )
            for arm in ("biological", "shuffled", "hash")
        }
        payload = {
            "experiment": "connectome-constrained reviewer routing",
            "controls": {
                "biological_vs_shuffled_coassignment_jaccard": round(
                    coassignment_jaccard(results["biological"], results["shuffled"]), 8
                ),
                "biological_vs_hash_coassignment_jaccard": round(
                    coassignment_jaccard(results["biological"], results["hash"]), 8
                ),
            },
            "arms": {arm: result.to_dict() for arm, result in results.items()},
        }
        _write_json(payload, args.out)
        return 0
    if args.command == "reservoir-poc":
        from .reservoir import load_corpus, run_reservoir_poc

        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        if len(annotations) != graph.node_count:
            raise SystemExit("Annotation order/count differs from the graph cache; rebuild the cache")
        result = run_reservoir_poc(
            load_corpus(args.corpus),
            graph,
            annotations,
            holdout_key=args.holdout,
            node_count=args.neurons,
            feature_width=args.features,
            passes=args.passes,
            leak=args.leak,
            input_scale=args.input_scale,
            target_radius=args.spectral_radius,
            ridge=args.ridge,
            seed=args.seed,
            bundle_count=args.bundles,
            reservoir_weight=args.reservoir_weight,
            context_budget=args.context_budget,
            evidence_path=args.truth_evidence,
        )
        _write_json(result, args.out)
        return 0
    if args.command == "reservoir-benchmark":
        from .reservoir import load_corpus, run_reservoir_benchmark

        if args.seeds < 1:
            raise SystemExit("--seeds must be positive")
        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        result = run_reservoir_benchmark(
            load_corpus(args.corpus),
            graph,
            annotations,
            seeds=tuple(range(args.seed, args.seed + args.seeds)),
            node_count=args.neurons,
            feature_width=args.features,
            passes=args.passes,
            leak=args.leak,
            input_scale=args.input_scale,
            target_radius=args.spectral_radius,
            ridge=args.ridge,
        )
        _write_json(result, args.out)
        return 0
    if args.command == "standalone-review":
        from .standalone import run_standalone_review

        result = run_standalone_review(args)
        print(args.out_dir / "review.json")
        print(f"findings={len(result['findings'])}")
        return 0
    if args.command == "standalone-compare":
        from .standalone import compare_with_competitor

        result = compare_with_competitor(args)
        print(args.out)
        for name in ("bugbrain", "competitor"):
            row = result[name]
            print(
                f"{name}: comments={row['comments']} P={row['precision']:.3f} "
                f"R={row['recall']:.3f} F1={row['f1']:.3f}"
            )
        return 0
    if args.command == "trace-import":
        from .policy import import_codex_jsonl

        payload = import_codex_jsonl(
            args.jsonl,
            key=args.key,
            split=args.split,
            goal=args.goal,
            final_reward=args.reward,
        )
        if args.append and args.out.exists():
            existing = json.loads(args.out.read_text(encoding="utf-8"))
            if existing.get("schema") != payload["schema"]:
                raise ValueError("Existing trajectory document has a different schema")
            if existing.get("actions") != payload["actions"]:
                raise ValueError("Existing trajectory document has a different action vocabulary")
            existing_keys = {str(row.get("key")) for row in existing.get("episodes", [])}
            new_key = str(payload["episodes"][0]["key"])
            if new_key in existing_keys:
                raise ValueError(f"Existing trajectory document already contains {new_key!r}")
            existing["episodes"].extend(payload["episodes"])
            payload = existing
        _write_json(payload, args.out)
        return 0
    if args.command == "policy-train":
        from .policy import load_trajectories, run_policy_experiment

        if args.seeds < 1:
            raise SystemExit("--seeds must be positive")
        actions, episodes = load_trajectories(args.trajectories)
        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        result = run_policy_experiment(
            actions,
            episodes,
            graph,
            annotations,
            seeds=tuple(range(args.seed, args.seed + args.seeds)),
            node_count=args.neurons,
            feature_width=args.features,
            readout_width=args.readout_width,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            reward_scale=args.reward_scale,
            signed=not args.unsigned,
            checkpoint_dir=args.checkpoint_dir,
        )
        _write_json(result, args.out)
        return 0
    if args.command == "policy-predict":
        from .policy import load_policy_checkpoint, predict_history

        raw_history = json.loads(args.history.read_text(encoding="utf-8"))
        if isinstance(raw_history, dict):
            raw_history = raw_history.get("observations")
        if not isinstance(raw_history, list) or not all(
            isinstance(observation, dict) for observation in raw_history
        ):
            raise ValueError("History must be a JSON array of observation objects")
        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        policy, encoder, checkpoint = load_policy_checkpoint(
            args.checkpoint,
            graph,
            annotations,
        )
        result = predict_history(policy, encoder, raw_history)
        result["checkpoint"] = checkpoint
        _write_json(result, args.out)
        return 0
    if args.command == "policy-run":
        from .controller import run_policy_controller
        from .policy import load_policy_checkpoint

        graph = load_cache(args.cache)
        annotations = load_annotations(args.annotations)
        policy, encoder, checkpoint = load_policy_checkpoint(
            args.checkpoint,
            graph,
            annotations,
        )
        result = run_policy_controller(args, policy, encoder, checkpoint)
        print(args.out_dir / "run.json")
        print(f"reward={result['reward']}")
        return 0
    if args.command == "direct-run":
        from .controller import run_direct_agent

        result = run_direct_agent(args)
        print(args.out_dir / "run.json")
        print(f"reward={result['reward']}")
        return 0
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
