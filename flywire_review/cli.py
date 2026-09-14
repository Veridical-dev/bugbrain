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
        description="Route code-review contexts through a fruit-fly connectome. For science. Probably.",
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
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
