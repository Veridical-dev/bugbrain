from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np

from flywire_review.connectome import Connectome
from flywire_review.model import NeuronAnnotation
from flywire_review.policy import (
    DEFAULT_ACTIONS,
    ObservationHasher,
    Trajectory,
    TrajectoryStep,
    _softmax,
    build_sparse_core,
    discounted_returns,
    import_codex_jsonl,
    load_connectome_checkpoint,
    load_policy_checkpoint,
    load_trajectories,
    make_sparse_controls,
    predict_history,
    run_policy_experiment,
)


def synthetic_graph(size: int = 24) -> tuple[Connectome, list[NeuronAnnotation]]:
    targets = array("I")
    weights = array("f")
    offsets = array("Q", [0])
    out_strength = array("f")
    for source in range(size):
        local = []
        for hop, weight in ((1, 4.0), (3, 2.0), (7, 3.0)):
            targets.append((source + hop) % size)
            weights.append(weight)
            local.append(weight)
        offsets.append(len(targets))
        out_strength.append(sum(local))
    annotations = []
    for index in range(size):
        flow = "afferent" if index < 6 else "efferent" if index >= size - 6 else "intrinsic"
        transmitter = "gaba" if index % 7 == 0 else "acetylcholine"
        annotations.append(
            NeuronAnnotation(
                root_id=50_000 + index,
                flow=flow,
                cell_type=f"test-{index}",
                neurotransmitter=transmitter,
            )
        )
    graph = Connectome(
        neuron_ids=array("Q", (row.root_id for row in annotations)),
        offsets=offsets,
        targets=targets,
        weights=weights,
        out_strength=out_strength,
        signs=array("b", (-1 if row.neurotransmitter == "gaba" else 1 for row in annotations)),
        metadata={"dataset": "synthetic-policy-test"},
    )
    return graph, annotations


def trajectory(key: str, split: str, variant: str) -> Trajectory:
    phases = (
        ("need_symbol", "search"),
        ("symbol_found", "inspect"),
        ("behavior_unknown", "test"),
        ("failure_explained", "reason"),
        ("repair_selected", "patch"),
        ("working_tree_changed", "verify"),
        ("verifier_passed", "stop"),
    )
    return Trajectory(
        key=key,
        split=split,
        metadata={},
        steps=tuple(
            TrajectoryStep(
                {
                    "phase": phase,
                    "repository": variant,
                    "previous_exit": 1 if phase == "failure_explained" else 0,
                },
                action,
                1.0 if action == "stop" else 0.0,
            )
            for phase, action in phases
        ),
    )


class ConnectomePolicyTests(unittest.TestCase):
    def test_trajectory_loader_validates_splits_and_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectories.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "bugbrain.trajectories/1",
                        "actions": list(DEFAULT_ACTIONS),
                        "episodes": [
                            {
                                "key": "train",
                                "split": "train",
                                "steps": [
                                    {"observation": {"phase": "start"}, "action": "search"}
                                ],
                            },
                            {
                                "key": "test",
                                "split": "test",
                                "steps": [
                                    {"observation": {"phase": "done"}, "action": "stop", "reward": 1}
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            actions, episodes = load_trajectories(path)
        self.assertEqual(actions, DEFAULT_ACTIONS)
        self.assertEqual({episode.split for episode in episodes}, {"train", "test"})

    def test_trajectory_loader_rejects_a_missing_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectories.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "bugbrain.trajectories/1",
                        "actions": list(DEFAULT_ACTIONS),
                        "episodes": [
                            {
                                "key": "train-only",
                                "split": "train",
                                "steps": [
                                    {"observation": {"phase": "start"}, "action": "search"}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "test split"):
                load_trajectories(path)

    def test_encoder_does_not_fit_on_holdout(self) -> None:
        encoder = ObservationHasher(width=32, seed=8).fit([{"phase": "train-only"}])
        before = encoder.idf.copy()
        transformed = encoder.transform([{"phase": "holdout-secret"}])
        np.testing.assert_array_equal(before, encoder.idf)
        self.assertAlmostEqual(float(np.linalg.norm(transformed[0])), 1.0)

    def test_discounted_terminal_reward_reaches_earlier_actions(self) -> None:
        episode = Trajectory(
            "reward",
            "train",
            (
                TrajectoryStep({"step": 0}, "search"),
                TrajectoryStep({"step": 1}, "inspect"),
                TrajectoryStep({"step": 2}, "stop", 1.0),
            ),
            {},
        )
        np.testing.assert_allclose(
            discounted_returns(episode),
            np.asarray([0.97**2, 0.97, 1.0]),
        )

    def test_softmax_is_stable_for_large_logits(self) -> None:
        probabilities = _softmax(np.asarray([10_000.0, 10_001.0]))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)

    def test_sparse_controls_preserve_degrees_and_weight_multiset(self) -> None:
        graph, annotations = synthetic_graph()
        core = build_sparse_core(graph, annotations, node_count=24)
        controls = make_sparse_controls(core, seed=19)
        original_targets = core.targets.copy()
        original_weights = core.weights.copy()
        rewired = controls["degree_rewired"]
        np.testing.assert_array_equal(
            np.bincount(core.sources, minlength=core.node_count),
            np.bincount(rewired.sources, minlength=core.node_count),
        )
        np.testing.assert_array_equal(
            np.bincount(core.targets, minlength=core.node_count),
            np.bincount(rewired.targets, minlength=core.node_count),
        )
        np.testing.assert_allclose(np.sort(core.weights), np.sort(rewired.weights))
        self.assertFalse(np.array_equal(core.targets, rewired.targets))
        np.testing.assert_array_equal(core.targets, original_targets)
        np.testing.assert_array_equal(core.weights, original_weights)

    def test_end_to_end_training_updates_internal_dynamics(self) -> None:
        graph, annotations = synthetic_graph()
        episodes = [
            *(trajectory(f"train-{index}", "train", f"repo-{index}") for index in range(6)),
            trajectory("validation", "validation", "heldout-validation"),
            trajectory("test", "test", "heldout-test"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = run_policy_experiment(
                DEFAULT_ACTIONS,
                episodes,
                graph,
                annotations,
                seeds=(7,),
                node_count=24,
                feature_width=48,
                readout_width=12,
                epochs=35,
                learning_rate=0.02,
                checkpoint_dir=temporary,
            )
            checkpoint = Path(temporary) / "biological-seed-7.npz"
            self.assertTrue(checkpoint.exists())
            self.assertTrue((Path(temporary) / "weight_shuffled-seed-7.npz").exists())
            self.assertTrue((Path(temporary) / "degree_rewired-seed-7.npz").exists())
            linear_checkpoint = Path(temporary) / "observation_only-seed-7.npz"
            self.assertTrue(linear_checkpoint.exists())
            policy, encoder, receipt = load_connectome_checkpoint(
                checkpoint,
                graph,
                annotations,
            )
            prediction = predict_history(
                policy,
                encoder,
                [step.observation for step in episodes[-1].steps],
            )
            linear_policy, linear_encoder, linear_receipt = load_policy_checkpoint(
                linear_checkpoint,
                graph,
                annotations,
            )
            linear_prediction = predict_history(
                linear_policy,
                linear_encoder,
                [step.observation for step in episodes[-1].steps],
            )
            for arm in ("weight_shuffled", "degree_rewired"):
                _, _, arm_receipt = load_policy_checkpoint(
                    Path(temporary) / f"{arm}-seed-7.npz",
                    graph,
                    annotations,
                )
                self.assertEqual(arm_receipt["arm"], arm)
            changed_annotations = list(annotations)
            first = changed_annotations[0]
            changed_annotations[0] = NeuronAnnotation(
                root_id=first.root_id,
                flow="intrinsic",
                cell_type=first.cell_type,
                neurotransmitter=first.neurotransmitter,
            )
            with self.assertRaisesRegex(ValueError, "interface"):
                load_connectome_checkpoint(checkpoint, graph, changed_annotations)
        self.assertEqual(receipt["topology_sha256"], result["base_graph"]["topology_sha256"])
        self.assertIn(prediction["next_action"], DEFAULT_ACTIONS)
        self.assertEqual(prediction["history_length"], len(episodes[-1].steps))
        self.assertEqual(linear_receipt["arm"], "observation_only")
        self.assertIn(linear_prediction["next_action"], DEFAULT_ACTIONS)
        self.assertEqual(result["schema"], "bugbrain.connectome-policy/1")
        self.assertEqual(
            {trial["arm"] for trial in result["trials"]},
            {"biological", "weight_shuffled", "degree_rewired", "observation_only"},
        )
        biological = next(trial for trial in result["trials"] if trial["arm"] == "biological")
        changes = biological["parameters"]["dynamics_l2_change"]
        self.assertTrue(any(value > 0 for value in changes.values()))
        self.assertLess(
            biological["training"]["final_epoch_loss"],
            biological["training"]["first_epoch_loss"],
        )
        self.assertGreaterEqual(biological["metrics"]["test"]["action_accuracy"], 0.5)
        bootstrap = result["summary"]["biological_deltas"]["degree_rewired"][
            "task_seed_bootstrap"
        ]
        self.assertEqual(bootstrap["test_tasks"], 1)
        self.assertEqual(bootstrap["paired_seeds"], 1)

    def test_codex_jsonl_import_uses_only_previous_tool_result(self) -> None:
        events = [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "I will inspect the code."},
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/bin/zsh -lc \"rg auth src\"",
                    "aggregated_output": "src/auth.py:12",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "sed -n '1,80p' src/auth.py",
                    "aggregated_output": "def authenticate(): pass",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "done"},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
            result = import_codex_jsonl(
                path,
                key="review-1",
                split="train",
                goal="review auth",
                final_reward=1.0,
            )
        steps = result["episodes"][0]["steps"]
        self.assertEqual([step["action"] for step in steps], ["search", "inspect", "stop"])
        self.assertNotIn("command", json.dumps(steps[0]["observation"]))
        self.assertIn("src/auth.py", json.dumps(steps[1]["observation"]))
        self.assertEqual(steps[-1]["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
