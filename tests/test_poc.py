from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flywire_review.connectome import _sign_for, build_cache, load_annotations, load_cache
from flywire_review.diff_parser import parse_diff_file, parse_review_input
from flywire_review.router import coassignment_jaccard, route_units
from flywire_review.model import NeuronAnnotation


FIXTURES = Path(__file__).parent / "fixtures"


class FlyWirePocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name) / "tiny.csr"
        self.metadata = build_cache(
            FIXTURES / "connections.csv",
            FIXTURES / "annotations.tsv",
            self.cache,
            min_synapses=5,
            progress_every=0,
        )
        self.graph = load_cache(self.cache)
        self.annotations = load_annotations(FIXTURES / "annotations.tsv")
        self.units = parse_diff_file(FIXTURES / "sample.diff")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cache_filters_and_round_trips(self) -> None:
        self.assertEqual(self.metadata["nodes"], 8)
        self.assertEqual(self.metadata["edges"], 8)
        self.assertEqual(self.graph.node_count, 8)
        self.assertEqual(self.graph.edge_count, 8)
        self.assertEqual(self.graph.offsets[-1], 8)
        self.assertEqual(self.metadata["sign_policy"], "effectome-v1-primary-transmitter")

    def test_effectome_sign_uses_primary_cotransmitter(self) -> None:
        self.assertEqual(_sign_for(NeuronAnnotation(1, neurotransmitter="gaba,acetylcholine")), -1)
        self.assertEqual(_sign_for(NeuronAnnotation(2, neurotransmitter="serotonin")), -1)
        self.assertEqual(_sign_for(NeuronAnnotation(3, neurotransmitter="acetylcholine")), 1)

    def test_diff_parser_makes_stable_units_and_risk_tags(self) -> None:
        self.assertEqual(len(self.units), 3)
        self.assertEqual(self.units[0].path, "app/auth.py")
        self.assertIn("security/trust-boundaries", self.units[0].risk_tags)
        self.assertIn("concurrency/ordering", self.units[1].risk_tags)
        self.assertIn("tests/observability", self.units[2].risk_tags)

    def test_frozen_input_parser_accepts_benchmark_shape(self) -> None:
        input_path = Path(self.temporary.name) / "frozen.json"
        input_path.write_text(
            '{"files":[{"filename":"src/a.py","patch":"@@ -1 +1 @@\\n-old\\n+new"}]}',
            encoding="utf-8",
        )
        units = parse_review_input(input_path)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].path, "src/a.py")

    def test_biological_route_is_deterministic_and_covers_every_unit(self) -> None:
        first = route_units(self.units, self.graph, self.annotations, bundle_count=2, seed=7, steps=3)
        second = route_units(self.units, self.graph, self.annotations, bundle_count=2, seed=7, steps=3)
        self.assertEqual(first.to_dict(), second.to_dict())
        routed = [unit.key for bundle in first.bundles for unit in bundle.units]
        self.assertCountEqual(routed, [unit.key for unit in self.units])
        self.assertTrue(all("NOT evidence" in bundle.prompt for bundle in first.bundles))

    def test_prompt_context_budget_is_enforced(self) -> None:
        result = route_units(
            self.units,
            self.graph,
            self.annotations,
            bundle_count=1,
            seed=7,
            steps=3,
            context_budget=2_500,
        )
        self.assertLessEqual(len(result.bundles[0].prompt), 3_200)

    def test_target_shuffle_preserves_degrees_and_target_multiset(self) -> None:
        shuffled = self.graph.shuffled_targets(11)
        self.assertEqual(shuffled.offsets, self.graph.offsets)
        self.assertEqual(sorted(shuffled.targets), sorted(self.graph.targets))
        self.assertNotEqual(shuffled.targets, self.graph.targets)

    def test_compare_arms_produces_bounded_jaccard(self) -> None:
        biological = route_units(self.units, self.graph, self.annotations, arm="biological", bundle_count=2, seed=9, steps=3)
        shuffled = route_units(self.units, self.graph, self.annotations, arm="shuffled", bundle_count=2, seed=9, steps=3)
        score = coassignment_jaccard(biological, shuffled)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
