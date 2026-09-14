from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np

from flywire_review.connectome import Connectome
from flywire_review.model import CodeUnit, NeuronAnnotation
from flywire_review.reservoir import (
    CorpusCase,
    FeatureHasher,
    ReservoirProjector,
    _scaled_core,
    grade_evidence_colocation,
    make_matched_controls,
    run_reservoir_benchmark,
    run_reservoir_poc,
)


def unit(key: str, path: str, line: int, token: str, tag: str) -> CodeUnit:
    return CodeUnit(
        key=key,
        path=path,
        new_line=line,
        heading=f"change {token}",
        excerpt=f"+ {token}",
        tokens=(token,),
        risk_tags=(tag,),
    )


def synthetic_connectome(size: int = 16) -> tuple[Connectome, list[NeuronAnnotation]]:
    targets = array("I")
    weights = array("f")
    offsets = array("Q", [0])
    out_strength = array("f")
    for source in range(size):
        local_weights = []
        for hop, weight in ((1, 3.0), (3, 2.0), (7, 1.0), (11, 4.0)):
            targets.append((source + hop) % size)
            weights.append(weight)
            local_weights.append(weight)
        offsets.append(len(targets))
        out_strength.append(sum(local_weights))
    annotations = [
        NeuronAnnotation(root_id=10_000 + index, cell_type=f"cell-{index}")
        for index in range(size)
    ]
    graph = Connectome(
        neuron_ids=array("Q", (annotation.root_id for annotation in annotations)),
        offsets=offsets,
        targets=targets,
        weights=weights,
        out_strength=out_strength,
        signs=array("b", [1]) * size,
        metadata={
            "dataset": "synthetic-test",
            "min_synapses": 1,
            "connections_md5": "test",
            "annotations_md5": "test",
        },
    )
    return graph, annotations


class ReservoirPocTests(unittest.TestCase):
    def test_feature_input_does_not_read_risk_targets(self) -> None:
        first = unit("a", "src/auth.py", 10, "token", "security/trust-boundaries")
        second = CodeUnit(
            key=first.key,
            path=first.path,
            new_line=first.new_line,
            heading=first.heading,
            excerpt=first.excerpt,
            tokens=first.tokens,
            risk_tags=("tests/observability",),
        )
        encoder = FeatureHasher(width=24, seed=4).fit((first,))
        np.testing.assert_array_equal(encoder.transform((first,)), encoder.transform((second,)))

    def test_controls_match_size_weight_multiset_and_radius(self) -> None:
        raw = np.zeros((20, 20), dtype=np.float64)
        for source in range(20):
            for hop, weight in ((1, 1.0), (2, 2.0), (5, 3.0), (9, 4.0)):
                raw[(source + hop) % 20, source] = weight
        biological = _scaled_core(
            "biological",
            raw,
            np.arange(20),
            target_radius=0.9,
            metadata={},
        )
        controls = make_matched_controls(biological, seed=9, target_radius=0.9)
        expected_weights = sorted(raw[raw != 0].tolist())
        expected_self_loops = int(np.count_nonzero(np.diag(raw)))
        for name in ("biological", "weight_shuffled", "random_topology"):
            control = controls[name]
            self.assertIsNotNone(control)
            assert control is not None
            self.assertEqual(np.count_nonzero(control.raw_matrix), np.count_nonzero(raw))
            self.assertEqual(sorted(control.raw_matrix[control.raw_matrix != 0].tolist()), expected_weights)
            self.assertEqual(int(np.count_nonzero(np.diag(control.raw_matrix))), expected_self_loops)
            self.assertAlmostEqual(control.spectral_radius, 0.9, places=6)
        self.assertFalse(np.array_equal(controls["random_topology"].raw_matrix, raw))

    def test_matched_projector_is_deterministic_and_topology_sensitive(self) -> None:
        raw = np.eye(8, k=1) + np.eye(8, k=-1)
        raw[0, -1] = raw[-1, 0] = 1.0
        biological = _scaled_core("bio", raw, np.arange(8), target_radius=0.8, metadata={})
        controls = make_matched_controls(biological, seed=3, target_radius=0.8)
        semantic = np.arange(32, dtype=np.float64).reshape(4, 8) / 31.0
        projector = ReservoirProjector(8, 8, seed=2, passes=3)
        first = projector.transform(semantic, biological)
        second = projector.transform(semantic, biological)
        shuffled = projector.transform(semantic, controls["random_topology"])
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.allclose(first, shuffled))

    def test_end_to_end_split_routes_each_file_atomically(self) -> None:
        graph, annotations = synthetic_connectome()
        security = "security/trust-boundaries"
        tests = "tests/observability"
        cases = (
            CorpusCase("train-a", Path("train-a"), (
                unit("a1", "auth/a.py", 10, "auth", security),
                unit("a2", "test/a.py", 20, "assert", tests),
            )),
            CorpusCase("train-b", Path("train-b"), (
                unit("b1", "auth/b.py", 10, "token", security),
                unit("b2", "test/b.py", 20, "test", tests),
            )),
            CorpusCase("holdout", Path("holdout"), (
                unit("h1", "same.py", 10, "auth", security),
                unit("h2", "same.py", 30, "assert", tests),
                unit("h3", "other.py", 50, "token", security),
            )),
        )
        result = run_reservoir_poc(
            cases,
            graph,
            annotations,
            holdout_key="holdout",
            node_count=16,
            feature_width=24,
            bundle_count=2,
            context_budget=1_500,
            seed=12,
        )
        self.assertEqual(set(result["arms"]), {
            "biological", "weight_shuffled", "random_topology", "semantic"
        })
        for route in result["arms"].values():
            key_to_bundle = {
                row["key"]: index
                for index, bundle in enumerate(route["bundles"])
                for row in bundle["units"]
            }
            self.assertEqual(key_to_bundle["h1"], key_to_bundle["h2"])
            self.assertEqual(set(key_to_bundle), {"h1", "h2", "h3"})
            self.assertTrue(all(len(bundle["prompt"]) <= 1_500 for bundle in route["bundles"]))

    def test_truth_evidence_is_a_post_route_colocation_grade(self) -> None:
        route = {
            "bundles": [{
                "units": [
                    {"key": "a", "path": "a.py", "new_line": 10},
                    {"key": "b", "path": "b.py", "new_line": 20},
                ],
                "included_unit_keys": ["a", "b"],
            }]
        }
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            evidence_path.write_text(json.dumps({"truth": [{
                "truth_id": "t1", "source": ["a.py:10", "b.py:20"]
            }]}), encoding="utf-8")
            grade = grade_evidence_colocation(route, evidence_path)
        self.assertEqual(grade["truth_roots"], 1)
        self.assertEqual(grade["evidence_pair_coverage"], 1.0)
        self.assertEqual(grade["prompt_evidence_pair_coverage"], 1.0)

    def test_benchmark_is_paired_across_holdouts_and_seeds(self) -> None:
        graph, annotations = synthetic_connectome()
        cases = (
            CorpusCase("a", Path("a"), (unit("a", "a.py", 1, "auth", "security/trust-boundaries"),)),
            CorpusCase("b", Path("b"), (unit("b", "b.py", 1, "test", "tests/observability"),)),
        )
        result = run_reservoir_benchmark(
            cases,
            graph,
            annotations,
            seeds=(4, 5),
            node_count=16,
            feature_width=16,
        )
        self.assertEqual(result["protocol"]["paired_trials"], 4)
        self.assertEqual(len(result["trials"]), 4)
        self.assertIn("random_topology", result["biological_deltas"])
        self.assertEqual(
            len(result["biological_deltas"]["random_topology"]["seed_block_bootstrap_95_interval"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
