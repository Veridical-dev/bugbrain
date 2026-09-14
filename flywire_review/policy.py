from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .connectome import Connectome
from .model import NeuronAnnotation


DEFAULT_ACTIONS = (
    "search",
    "inspect",
    "test",
    "reason",
    "patch",
    "verify",
    "stop",
)
TRAJECTORY_SCHEMA = "bugbrain.trajectories/1"
REPORT_SCHEMA = "bugbrain.connectome-policy/1"
CHECKPOINT_SCHEMA = "bugbrain.connectome-checkpoint/1"
LINEAR_CHECKPOINT_SCHEMA = "bugbrain.linear-checkpoint/1"
_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$.-]{1,}")


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    observation: dict[str, Any]
    action: str
    reward: float = 0.0


@dataclass(frozen=True, slots=True)
class Trajectory:
    key: str
    split: str
    steps: tuple[TrajectoryStep, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SparseCore:
    name: str
    node_ids: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    input_nodes: np.ndarray
    output_nodes: np.ndarray
    flow_groups: np.ndarray
    metadata: dict[str, Any]

    @property
    def node_count(self) -> int:
        return int(len(self.node_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.sources))

    def matvec(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.targets,
            weights=self.weights * values[self.sources],
            minlength=self.node_count,
        ).astype(np.float64, copy=False)

    def transpose_matvec(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.sources,
            weights=self.weights * values[self.targets],
            minlength=self.node_count,
        ).astype(np.float64, copy=False)


def _digest(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=16).digest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        raw = np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> Iterable[str]:
    """Turn a structured agent observation into stable, non-executable features."""
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value[key], child)
        return
    if isinstance(value, (list, tuple)):
        yield f"{prefix}:length={len(value)}"
        for item in value:
            yield from _flatten(item, prefix)
        return
    if isinstance(value, bool):
        yield f"{prefix}:bool={str(value).lower()}"
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            magnitude = 0 if numeric == 0 else int(math.floor(math.log2(abs(numeric)) + 1))
            sign = "neg" if numeric < 0 else "pos"
            yield f"{prefix}:number={sign}:{magnitude}"
        else:
            yield f"{prefix}:number=nonfinite"
        return
    rendered = str(value).lower()
    tokens = _TOKEN.findall(rendered)
    if not tokens:
        yield f"{prefix}:empty"
    for token in tokens[:256]:
        yield f"{prefix}:token={token}"


class ObservationHasher:
    """Signed hashing encoder fitted on training observations only."""

    def __init__(self, width: int = 128, seed: int = 2305) -> None:
        if width < 8:
            raise ValueError("feature width must be at least 8")
        self.width = width
        self.seed = seed
        self.idf = np.ones(width, dtype=np.float64)

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = _digest(self.seed, feature)
        return int.from_bytes(digest[:8], "little") % self.width, (1.0 if digest[8] & 1 else -1.0)

    def fit(self, observations: Sequence[dict[str, Any]]) -> ObservationHasher:
        frequency = np.zeros(self.width, dtype=np.float64)
        for observation in observations:
            buckets = {self._bucket(feature)[0] for feature in _flatten(observation)}
            if buckets:
                frequency[list(buckets)] += 1.0
        self.idf = np.log((1.0 + len(observations)) / (1.0 + frequency)) + 1.0
        return self

    def transform(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        encoded = np.zeros((len(observations), self.width), dtype=np.float64)
        for row, observation in enumerate(observations):
            for feature in _flatten(observation):
                index, sign = self._bucket(feature)
                encoded[row, index] += sign * self.idf[index]
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return encoded / norms


def load_trajectories(path: str | Path) -> tuple[tuple[str, ...], list[Trajectory]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(f"Trajectory file must use schema {TRAJECTORY_SCHEMA!r}")
    raw_actions = payload.get("actions", DEFAULT_ACTIONS)
    if not isinstance(raw_actions, list) or len(raw_actions) < 2:
        raise ValueError("Trajectory file requires at least two actions")
    actions = tuple(str(action).strip() for action in raw_actions)
    if any(not action for action in actions) or len(set(actions)) != len(actions):
        raise ValueError("Trajectory actions must be unique non-empty strings")

    rows = payload.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Trajectory file contains no episodes")
    episodes: list[Trajectory] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every episode must be an object")
        key = str(row.get("key") or "").strip()
        split = str(row.get("split") or "").strip().lower()
        if not key or key in seen:
            raise ValueError(f"Invalid or duplicate episode key: {key!r}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Episode {key!r} has invalid split {split!r}")
        seen.add(key)
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"Episode {key!r} contains no steps")
        steps: list[TrajectoryStep] = []
        for position, raw in enumerate(raw_steps):
            if not isinstance(raw, dict) or not isinstance(raw.get("observation"), dict):
                raise ValueError(f"Episode {key!r} step {position} lacks an observation object")
            action = str(raw.get("action") or "").strip()
            if action not in actions:
                raise ValueError(f"Episode {key!r} step {position} has unknown action {action!r}")
            try:
                reward = float(raw.get("reward", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Episode {key!r} step {position} has invalid reward") from exc
            if not math.isfinite(reward):
                raise ValueError(f"Episode {key!r} step {position} has non-finite reward")
            steps.append(TrajectoryStep(dict(raw["observation"]), action, reward))
        episodes.append(
            Trajectory(key, split, tuple(steps), dict(row.get("metadata") or {}))
        )
    if not any(episode.split == "train" for episode in episodes):
        raise ValueError("Trajectory file requires a train split")
    if not any(episode.split == "test" for episode in episodes):
        raise ValueError("Trajectory file requires a test split")
    return actions, episodes


def _flow_group(annotation: NeuronAnnotation) -> int:
    flow = annotation.flow.casefold()
    if flow in {"afferent", "sensory"}:
        return 0
    if flow in {"efferent", "motor", "descending"}:
        return 2
    return 1


def build_sparse_core(
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    node_count: int = 2_048,
    recurrence_scale: float = 0.92,
    signed: bool = True,
) -> SparseCore:
    """Build an induced sparse graph; node_count=0 retains the complete cache."""
    if len(annotations) != graph.node_count:
        raise ValueError("Annotation order/count differs from the graph cache")
    if node_count < 0 or node_count > graph.node_count:
        raise ValueError("node_count must be zero or no larger than the graph")
    if not 0.0 < recurrence_scale <= 1.0:
        raise ValueError("recurrence_scale must be in (0, 1]")

    full_targets = np.asarray(graph.targets, dtype=np.int64)
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    out_degree = np.diff(offsets)
    in_degree = np.bincount(full_targets, minlength=graph.node_count)
    if node_count == 0 or node_count == graph.node_count:
        selected = np.arange(graph.node_count, dtype=np.int64)
    else:
        ids = np.asarray(graph.neuron_ids, dtype=np.uint64)
        order = np.lexsort((ids, -(out_degree + in_degree)))
        selected = np.sort(order[:node_count].astype(np.int64, copy=False))

    reverse = np.full(graph.node_count, -1, dtype=np.int64)
    reverse[selected] = np.arange(len(selected), dtype=np.int64)
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    full_weights = np.asarray(graph.weights, dtype=np.float64)
    signs = np.asarray(graph.signs, dtype=np.float64)
    for local_source, full_source in enumerate(selected):
        start, stop = offsets[full_source], offsets[full_source + 1]
        local_targets = reverse[full_targets[start:stop]]
        keep = local_targets >= 0
        if not np.any(keep):
            continue
        source_rows.append(np.full(int(np.count_nonzero(keep)), local_source, dtype=np.int64))
        target_rows.append(local_targets[keep].astype(np.int64, copy=False))
        values = np.log1p(full_weights[start:stop][keep])
        if signed:
            values = values * signs[full_source]
        weight_rows.append(values)
    if not source_rows:
        raise ValueError("Selected connectome core contains no internal edges")
    sources = np.concatenate(source_rows)
    targets = np.concatenate(target_rows)
    raw_weights = np.concatenate(weight_rows)
    incoming = np.bincount(targets, weights=np.abs(raw_weights), minlength=len(selected))
    normalized = recurrence_scale * raw_weights / np.maximum(incoming[targets], 1.0)

    groups = np.asarray([_flow_group(annotations[index]) for index in selected], dtype=np.int8)
    input_nodes = np.flatnonzero(groups == 0).astype(np.int64)
    output_nodes = np.flatnonzero(groups == 2).astype(np.int64)
    if not len(input_nodes):
        input_nodes = np.arange(len(selected), dtype=np.int64)
    if not len(output_nodes):
        output_nodes = np.arange(len(selected), dtype=np.int64)

    node_ids = np.asarray(graph.neuron_ids, dtype=np.uint64)[selected]
    topology_payload = np.stack((sources, targets), axis=1).astype("<i8", copy=False).tobytes()
    core_sha256 = _array_digest(
        node_ids.astype("<u8", copy=False),
        sources.astype("<i8", copy=False),
        targets.astype("<i8", copy=False),
        normalized.astype("<f8", copy=False),
        groups,
        input_nodes.astype("<i8", copy=False),
        output_nodes.astype("<i8", copy=False),
    )
    return SparseCore(
        name="biological",
        node_ids=node_ids,
        sources=sources,
        targets=targets,
        weights=normalized,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        flow_groups=groups,
        metadata={
            "dataset": graph.metadata.get("dataset"),
            "nodes": int(len(selected)),
            "edges": int(len(sources)),
            "selection": "complete graph" if len(selected) == graph.node_count else "highest-degree induced core",
            "signed": signed,
            "weight_initialization": "signed log1p(synapse_count), incoming-L1 normalized",
            "recurrence_scale": recurrence_scale,
            "input_nodes": int(len(input_nodes)),
            "output_nodes": int(len(output_nodes)),
            "topology_sha256": hashlib.sha256(topology_payload).hexdigest(),
            "weights_sha256": hashlib.sha256(
                normalized.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "core_sha256": core_sha256,
        },
    )


def make_sparse_controls(core: SparseCore, *, seed: int) -> dict[str, SparseCore]:
    """Create paired controls without changing node IDs, I/O maps, E, or weight multiset."""
    rng = np.random.default_rng(seed)
    shuffled_weights = core.weights.copy()
    rng.shuffle(shuffled_weights)

    # Shuffle target/weight pairs across fixed source slots. This exactly preserves every
    # source out-degree, every target in-degree, and the complete weight multiset. It is a
    # directed configuration multigraph, so parallel edges and changed self-loop counts are
    # disclosed rather than silently repaired.
    permutation = rng.permutation(core.edge_count)
    rewired_targets = core.targets[permutation]
    rewired_weights = core.weights[permutation]

    def variant(name: str, targets: np.ndarray, weights: np.ndarray, control: str) -> SparseCore:
        payload = np.stack((core.sources, targets), axis=1).astype("<i8", copy=False).tobytes()
        return SparseCore(
            name=name,
            node_ids=core.node_ids,
            sources=core.sources,
            targets=targets,
            weights=weights,
            input_nodes=core.input_nodes,
            output_nodes=core.output_nodes,
            flow_groups=core.flow_groups,
            metadata={
                **core.metadata,
                "control": control,
                "control_seed": seed,
                "topology_sha256": hashlib.sha256(payload).hexdigest(),
                "weights_sha256": hashlib.sha256(
                    weights.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "core_sha256": _array_digest(
                    core.node_ids.astype("<u8", copy=False),
                    core.sources.astype("<i8", copy=False),
                    targets.astype("<i8", copy=False),
                    weights.astype("<f8", copy=False),
                    core.flow_groups,
                    core.input_nodes.astype("<i8", copy=False),
                    core.output_nodes.astype("<i8", copy=False),
                ),
            },
        )

    return {
        "biological": core,
        "weight_shuffled": variant(
            "weight_shuffled",
            core.targets,
            shuffled_weights,
            "biological topology with permuted normalized edge weights",
        ),
        "degree_rewired": variant(
            "degree_rewired",
            rewired_targets,
            rewired_weights,
            "directed configuration multigraph preserving source out-degree, target in-degree, and target/weight pairs",
        ),
    }


def discounted_returns(trajectory: Trajectory, discount: float = 0.97) -> np.ndarray:
    returns = np.zeros(len(trajectory.steps), dtype=np.float64)
    running = 0.0
    for index in range(len(trajectory.steps) - 1, -1, -1):
        running = trajectory.steps[index].reward + discount * running
        returns[index] = running
    return returns


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


class _Adam:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step = 0

    def update(self, gradients: dict[str, np.ndarray], clip: float = 5.0) -> float:
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()))
        scale = min(1.0, clip / max(norm, 1e-12))
        self.step += 1
        for name, gradient in gradients.items():
            gradient = gradient * scale
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradient * gradient
            first = self.first[name] / (1.0 - 0.9**self.step)
            second = self.second[name] / (1.0 - 0.999**self.step)
            self.parameters[name] -= self.learning_rate * first / (np.sqrt(second) + 1e-8)
        return norm


class ConnectomeActionPolicy:
    """Sparse recurrent policy with trainable flow dynamics and action decoder."""

    def __init__(
        self,
        core: SparseCore,
        feature_width: int,
        actions: Sequence[str],
        *,
        readout_width: int = 32,
        seed: int = 0,
    ) -> None:
        if readout_width < 4:
            raise ValueError("readout width must be at least 4")
        self.core = core
        self.feature_width = feature_width
        self.actions = tuple(actions)
        self.readout_width = readout_width
        rng = np.random.default_rng(seed)

        self.input_assignment = rng.choice(core.input_nodes, size=feature_width, replace=True)
        self.input_sign = rng.choice(np.asarray([-1.0, 1.0]), size=feature_width)
        input_counts = np.bincount(self.input_assignment, minlength=core.node_count)
        self.input_norm = np.sqrt(np.maximum(input_counts, 1.0))

        self.output_bucket = rng.integers(0, readout_width, size=len(core.output_nodes))
        self.output_sign = rng.choice(np.asarray([-1.0, 1.0]), size=len(core.output_nodes))
        bucket_counts = np.bincount(self.output_bucket, minlength=readout_width)
        self.output_norm = np.sqrt(np.maximum(bucket_counts, 1.0))

        self.parameters = {
            "self_gain": np.asarray([0.45, 0.55, 0.45], dtype=np.float64),
            "message_gain": np.asarray([0.75, 0.85, 0.75], dtype=np.float64),
            "input_gain": np.asarray([0.8], dtype=np.float64),
            "flow_bias": np.zeros(3, dtype=np.float64),
            "readout": rng.normal(0.0, 0.03, size=(readout_width, len(actions))),
            "action_bias": np.zeros(len(actions), dtype=np.float64),
        }
        self.initial_dynamics = {
            name: value.copy()
            for name, value in self.parameters.items()
            if name in {"self_gain", "message_gain", "input_gain", "flow_bias"}
        }

    def _inject(self, features: np.ndarray) -> np.ndarray:
        drive = np.bincount(
            self.input_assignment,
            weights=features * self.input_sign,
            minlength=self.core.node_count,
        ).astype(np.float64, copy=False)
        return drive / self.input_norm

    def _pool(self, state: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.output_bucket,
            weights=state[self.core.output_nodes] * self.output_sign,
            minlength=self.readout_width,
        ).astype(np.float64, copy=False) / self.output_norm

    def _unpool(self, gradient: np.ndarray) -> np.ndarray:
        state_gradient = np.zeros(self.core.node_count, dtype=np.float64)
        state_gradient[self.core.output_nodes] = (
            gradient[self.output_bucket] * self.output_sign / self.output_norm[self.output_bucket]
        )
        return state_gradient

    def forward(self, features: np.ndarray) -> list[dict[str, np.ndarray]]:
        state = np.zeros(self.core.node_count, dtype=np.float64)
        records: list[dict[str, np.ndarray]] = []
        groups = self.core.flow_groups
        for row in features:
            previous = state
            message = self.core.matvec(previous)
            drive = self._inject(row)
            preactivation = (
                self.parameters["self_gain"][groups] * previous
                + self.parameters["message_gain"][groups] * message
                + self.parameters["input_gain"][0] * drive
                + self.parameters["flow_bias"][groups]
            )
            state = np.tanh(preactivation)
            pooled = self._pool(state)
            logits = pooled @ self.parameters["readout"] + self.parameters["action_bias"]
            records.append(
                {
                    "previous": previous,
                    "message": message,
                    "drive": drive,
                    "state": state,
                    "pooled": pooled,
                    "logits": logits,
                }
            )
        return records

    def train_episode(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray,
        optimizer: _Adam,
    ) -> tuple[float, float]:
        records = self.forward(features)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        direct_state = [np.zeros(self.core.node_count, dtype=np.float64) for _ in records]
        loss = 0.0
        normalization = max(float(np.sum(sample_weights)), 1e-12)
        for index, record in enumerate(records):
            probabilities = _softmax(record["logits"])
            weight = float(sample_weights[index]) / normalization
            loss -= weight * math.log(max(float(probabilities[targets[index]]), 1e-12))
            delta = probabilities
            delta[targets[index]] -= 1.0
            delta *= weight
            gradients["readout"] += np.outer(record["pooled"], delta)
            gradients["action_bias"] += delta
            direct_state[index] += self._unpool(self.parameters["readout"] @ delta)

        future = np.zeros(self.core.node_count, dtype=np.float64)
        groups = self.core.flow_groups
        for index in range(len(records) - 1, -1, -1):
            record = records[index]
            state_gradient = direct_state[index] + future
            delta_state = state_gradient * (1.0 - record["state"] * record["state"])
            gradients["self_gain"] += np.bincount(
                groups,
                weights=delta_state * record["previous"],
                minlength=3,
            )
            gradients["message_gain"] += np.bincount(
                groups,
                weights=delta_state * record["message"],
                minlength=3,
            )
            gradients["input_gain"][0] += float(np.dot(delta_state, record["drive"]))
            gradients["flow_bias"] += np.bincount(groups, weights=delta_state, minlength=3)
            future = (
                self.parameters["self_gain"][groups] * delta_state
                + self.core.transpose_matvec(self.parameters["message_gain"][groups] * delta_state)
            )
        gradient_norm = optimizer.update(gradients)
        return loss, gradient_norm

    def parameter_receipt(self) -> dict[str, Any]:
        dynamics = {
            name: [round(float(value), 8) for value in values]
            for name, values in self.parameters.items()
            if name in self.initial_dynamics
        }
        deltas = {
            name: round(float(np.linalg.norm(self.parameters[name] - initial)), 10)
            for name, initial in self.initial_dynamics.items()
        }
        return {
            "trainable_parameters": int(sum(value.size for value in self.parameters.values())),
            "trained_internal_dynamics": True,
            "fixed_edge_topology": True,
            "initial_dynamics": {
                name: [round(float(value), 8) for value in values]
                for name, values in self.initial_dynamics.items()
            },
            "final_dynamics": dynamics,
            "dynamics_l2_change": deltas,
        }


class LinearActionPolicy:
    """No-connectome observation-only control."""

    def __init__(self, feature_width: int, actions: Sequence[str], *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.actions = tuple(actions)
        self.parameters = {
            "weights": rng.normal(0.0, 0.03, size=(feature_width, len(actions))),
            "bias": np.zeros(len(actions), dtype=np.float64),
        }

    def train_episode(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray,
        optimizer: _Adam,
    ) -> tuple[float, float]:
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        loss = 0.0
        normalization = max(float(np.sum(sample_weights)), 1e-12)
        for row, target, raw_weight in zip(features, targets, sample_weights):
            probabilities = _softmax(row @ self.parameters["weights"] + self.parameters["bias"])
            weight = float(raw_weight) / normalization
            loss -= weight * math.log(max(float(probabilities[target]), 1e-12))
            delta = probabilities
            delta[target] -= 1.0
            delta *= weight
            gradients["weights"] += np.outer(row, delta)
            gradients["bias"] += delta
        return loss, optimizer.update(gradients)

    def forward(self, features: np.ndarray) -> list[dict[str, np.ndarray]]:
        return [
            {"logits": row @ self.parameters["weights"] + self.parameters["bias"]}
            for row in features
        ]

    def parameter_receipt(self) -> dict[str, Any]:
        return {
            "trainable_parameters": int(sum(value.size for value in self.parameters.values())),
            "trained_internal_dynamics": False,
            "fixed_edge_topology": False,
        }


def save_connectome_checkpoint(
    path: str | Path,
    policy: ConnectomeActionPolicy,
    encoder: ObservationHasher,
    *,
    seed: int,
    signed: bool,
) -> Path:
    """Persist one learned biological policy without pickled Python objects."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": int(seed),
        "arm": policy.core.name,
        "control_seed": policy.core.metadata.get("control_seed"),
        "signed": bool(signed),
        "nodes": policy.core.node_count,
        "edges": policy.core.edge_count,
        "topology_sha256": policy.core.metadata["topology_sha256"],
        "core_sha256": policy.core.metadata["core_sha256"],
        "recurrence_scale": policy.core.metadata["recurrence_scale"],
    }
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "actions": np.asarray(policy.actions),
        "node_ids": policy.core.node_ids,
        "feature_width": np.asarray(policy.feature_width, dtype=np.int64),
        "readout_width": np.asarray(policy.readout_width, dtype=np.int64),
        "encoder_seed": np.asarray(encoder.seed, dtype=np.int64),
        "encoder_idf": encoder.idf,
        "input_assignment": policy.input_assignment,
        "input_sign": policy.input_sign,
        "output_bucket": policy.output_bucket,
        "output_sign": policy.output_sign,
    }
    arrays.update(
        {f"parameter__{name}": value for name, value in policy.parameters.items()}
    )
    np.savez_compressed(destination, **arrays)
    return destination


def load_connectome_checkpoint(
    path: str | Path,
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
) -> tuple[ConnectomeActionPolicy, ObservationHasher, dict[str, Any]]:
    """Rebuild the fixed graph and load a learned policy after receipt checks."""
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"].item())
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError(f"Checkpoint must use schema {CHECKPOINT_SCHEMA!r}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        node_count = int(metadata["nodes"])
        base_core = build_sparse_core(
            graph,
            annotations,
            node_count=0 if node_count == graph.node_count else node_count,
            recurrence_scale=float(metadata["recurrence_scale"]),
            signed=bool(metadata["signed"]),
        )
        arm = str(metadata.get("arm") or "biological")
        if arm == "biological":
            core = base_core
        elif arm in {"weight_shuffled", "degree_rewired"}:
            control_seed = metadata.get("control_seed")
            if control_seed is None:
                raise ValueError("Graph-null checkpoint lacks its control seed")
            core = make_sparse_controls(base_core, seed=int(control_seed))[arm]
        else:
            raise ValueError(f"Unknown connectome checkpoint arm: {arm!r}")
        if core.metadata["topology_sha256"] != metadata["topology_sha256"]:
            raise ValueError("Checkpoint topology receipt differs from the supplied graph")
        if not np.array_equal(core.node_ids, archive["node_ids"]):
            raise ValueError("Checkpoint neuron IDs differ from the supplied graph")
        if core.metadata["core_sha256"] != metadata["core_sha256"]:
            raise ValueError("Checkpoint graph weights or neuron interface differ from the supplied graph")

        actions = tuple(str(action) for action in archive["actions"].tolist())
        feature_width = int(archive["feature_width"].item())
        readout_width = int(archive["readout_width"].item())
        policy = ConnectomeActionPolicy(
            core,
            feature_width,
            actions,
            readout_width=readout_width,
            seed=int(metadata["seed"]),
        )
        for name, destination in policy.parameters.items():
            source = archive[f"parameter__{name}"]
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint parameter {name!r} has the wrong shape")
            destination[...] = source
        for name in ("input_assignment", "input_sign", "output_bucket", "output_sign"):
            source = archive[name]
            destination = getattr(policy, name)
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint mapping {name!r} has the wrong shape")
            destination[...] = source
        input_counts = np.bincount(policy.input_assignment, minlength=core.node_count)
        policy.input_norm = np.sqrt(np.maximum(input_counts, 1.0))
        bucket_counts = np.bincount(policy.output_bucket, minlength=readout_width)
        policy.output_norm = np.sqrt(np.maximum(bucket_counts, 1.0))

        encoder = ObservationHasher(
            feature_width,
            seed=int(archive["encoder_seed"].item()),
        )
        encoder.idf = archive["encoder_idf"].astype(np.float64, copy=True)
    return policy, encoder, metadata


def save_linear_checkpoint(
    path: str | Path,
    policy: LinearActionPolicy,
    encoder: ObservationHasher,
    *,
    seed: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema=np.asarray(LINEAR_CHECKPOINT_SCHEMA),
        metadata_json=np.asarray(json.dumps({"seed": int(seed), "arm": "observation_only"})),
        actions=np.asarray(policy.actions),
        feature_width=np.asarray(encoder.width, dtype=np.int64),
        encoder_seed=np.asarray(encoder.seed, dtype=np.int64),
        encoder_idf=encoder.idf,
        parameter__weights=policy.parameters["weights"],
        parameter__bias=policy.parameters["bias"],
    )
    return destination


def load_linear_checkpoint(
    path: str | Path,
) -> tuple[LinearActionPolicy, ObservationHasher, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["schema"].item()) != LINEAR_CHECKPOINT_SCHEMA:
            raise ValueError(f"Checkpoint must use schema {LINEAR_CHECKPOINT_SCHEMA!r}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        actions = tuple(str(action) for action in archive["actions"].tolist())
        feature_width = int(archive["feature_width"].item())
        policy = LinearActionPolicy(feature_width, actions, seed=int(metadata["seed"]))
        for name, destination in policy.parameters.items():
            source = archive[f"parameter__{name}"]
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint parameter {name!r} has the wrong shape")
            destination[...] = source
        encoder = ObservationHasher(
            feature_width,
            seed=int(archive["encoder_seed"].item()),
        )
        encoder.idf = archive["encoder_idf"].astype(np.float64, copy=True)
    return policy, encoder, metadata


def load_policy_checkpoint(
    path: str | Path,
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
) -> tuple[ConnectomeActionPolicy | LinearActionPolicy, ObservationHasher, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"].item())
    if schema == CHECKPOINT_SCHEMA:
        return load_connectome_checkpoint(path, graph, annotations)
    if schema == LINEAR_CHECKPOINT_SCHEMA:
        return load_linear_checkpoint(path)
    raise ValueError(f"Unknown policy checkpoint schema: {schema!r}")


def predict_history(
    policy: ConnectomeActionPolicy | LinearActionPolicy,
    encoder: ObservationHasher,
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("At least one observation is required")
    features = encoder.transform(observations)
    predictions = []
    for position, record in enumerate(policy.forward(features)):
        probabilities = _softmax(record["logits"])
        order = np.argsort(-probabilities)
        predictions.append(
            {
                "step": position,
                "action": policy.actions[int(order[0])],
                "probabilities": {
                    policy.actions[int(index)]: round(float(probabilities[index]), 8)
                    for index in order
                },
            }
        )
    return {
        "schema": "bugbrain.action-prediction/1",
        "next_action": predictions[-1]["action"],
        "history_length": len(observations),
        "predictions": predictions,
    }


def _encoded_episode(
    episode: Trajectory,
    encoder: ObservationHasher,
    action_index: dict[str, int],
    reward_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = encoder.transform([step.observation for step in episode.steps])
    targets = np.asarray([action_index[step.action] for step in episode.steps], dtype=np.int64)
    returns = np.clip(discounted_returns(episode), -2.0, 2.0)
    weights = np.exp(reward_scale * returns)
    return features, targets, weights


def _metrics(
    policy: ConnectomeActionPolicy | LinearActionPolicy,
    episodes: Sequence[Trajectory],
    encoder: ObservationHasher,
    action_index: dict[str, int],
) -> dict[str, Any]:
    correct = 0
    count = 0
    nll = 0.0
    exact = 0
    terminal_correct = 0
    confusion = np.zeros((len(action_index), len(action_index)), dtype=np.int64)
    per_episode: list[dict[str, Any]] = []
    for episode in episodes:
        features = encoder.transform([step.observation for step in episode.steps])
        targets = np.asarray([action_index[step.action] for step in episode.steps], dtype=np.int64)
        predictions: list[int] = []
        episode_correct = 0
        episode_nll = 0.0
        for target, record in zip(targets, policy.forward(features)):
            probabilities = _softmax(record["logits"])
            prediction = int(np.argmax(probabilities))
            predictions.append(prediction)
            correct += int(prediction == target)
            episode_correct += int(prediction == target)
            count += 1
            item_nll = -math.log(max(float(probabilities[target]), 1e-12))
            nll += item_nll
            episode_nll += item_nll
            confusion[target, prediction] += 1
        is_exact = int(np.array_equal(np.asarray(predictions), targets))
        is_terminal_correct = int(predictions[-1] == targets[-1])
        exact += is_exact
        terminal_correct += is_terminal_correct
        per_episode.append(
            {
                "key": episode.key,
                "steps": len(targets),
                "action_accuracy": round(episode_correct / len(targets), 6),
                "mean_nll": round(episode_nll / len(targets), 6),
                "exact_trajectory": bool(is_exact),
                "terminal_action_correct": bool(is_terminal_correct),
            }
        )
    return {
        "episodes": len(episodes),
        "steps": count,
        "action_accuracy": round(correct / count, 6) if count else None,
        "mean_nll": round(nll / count, 6) if count else None,
        "exact_trajectory_accuracy": round(exact / len(episodes), 6) if episodes else None,
        "terminal_action_accuracy": round(terminal_correct / len(episodes), 6) if episodes else None,
        "confusion": confusion.tolist(),
        "per_episode": per_episode,
    }


def _paired_task_seed_bootstrap(
    trials: Sequence[dict[str, Any]],
    control: str,
    *,
    samples: int = 5_000,
    seed: int = 783,
) -> dict[str, Any]:
    cells: dict[str, dict[int, dict[str, float]]] = {}
    for trial in trials:
        arm = str(trial["arm"])
        if arm not in {"biological", control}:
            continue
        cells.setdefault(arm, {})[int(trial["seed"])] = {
            str(row["key"]): float(row["action_accuracy"])
            for row in trial["metrics"]["test"]["per_episode"]
        }
    seeds = sorted(set(cells.get("biological", {})) & set(cells.get(control, {})))
    if not seeds:
        return {"status": "unavailable"}
    keys = sorted(
        set.intersection(
            *(
                set(cells[arm][trial_seed])
                for arm in ("biological", control)
                for trial_seed in seeds
            )
        )
    )
    if not keys:
        return {"status": "unavailable"}
    matrix = np.asarray(
        [
            [cells["biological"][trial_seed][key] - cells[control][trial_seed][key] for key in keys]
            for trial_seed in seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        sampled_tasks = rng.integers(0, len(keys), size=len(keys))
        estimates[index] = float(matrix[np.ix_(sampled_seeds, sampled_tasks)].mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "paired_task_seed_bootstrap",
        "test_tasks": len(keys),
        "paired_seeds": len(seeds),
        "samples": samples,
        "macro_action_accuracy_delta": round(float(matrix.mean()), 6),
        "ci95": [round(float(low), 6), round(float(high), 6)],
        "probability_delta_above_zero": round(float(np.mean(estimates > 0.0)), 6),
        "probability_delta_at_least_0.05": round(float(np.mean(estimates >= 0.05)), 6),
    }


def _summarize(trials: Sequence[dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in arms:
        rows = [
            float(trial["metrics"]["test"]["action_accuracy"])
            for trial in trials
            if trial["arm"] == arm
        ]
        summary[arm] = {
            "trials": len(rows),
            "test_action_accuracy_mean": round(float(np.mean(rows)), 6),
            "test_action_accuracy_std": round(float(np.std(rows)), 6),
        }
    biological = {
        trial["seed"]: float(trial["metrics"]["test"]["action_accuracy"])
        for trial in trials
        if trial["arm"] == "biological"
    }
    deltas: dict[str, Any] = {}
    for arm in arms:
        if arm == "biological":
            continue
        paired = [
            biological[trial["seed"]] - float(trial["metrics"]["test"]["action_accuracy"])
            for trial in trials
            if trial["arm"] == arm and trial["seed"] in biological
        ]
        deltas[arm] = {
            "biological_minus_control_mean": round(float(np.mean(paired)), 6),
            "paired_values": [round(value, 6) for value in paired],
            "task_seed_bootstrap": _paired_task_seed_bootstrap(trials, arm),
        }
    return {"arms": summary, "biological_deltas": deltas}


def run_policy_experiment(
    actions: Sequence[str],
    episodes: Sequence[Trajectory],
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    seeds: Sequence[int] = (2305,),
    node_count: int = 2_048,
    feature_width: int = 128,
    readout_width: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.015,
    reward_scale: float = 0.35,
    signed: bool = True,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    splits = {
        name: [episode for episode in episodes if episode.split == name]
        for name in ("train", "validation", "test")
    }
    train_observations = [
        step.observation for episode in splits["train"] for step in episode.steps
    ]
    encoder = ObservationHasher(feature_width, seed=991).fit(train_observations)
    action_index = {action: index for index, action in enumerate(actions)}
    base_core = build_sparse_core(
        graph,
        annotations,
        node_count=node_count,
        signed=signed,
    )

    trials: list[dict[str, Any]] = []
    for seed in seeds:
        cores = make_sparse_controls(base_core, seed=seed + 100_000)
        for arm, core in cores.items():
            policy = ConnectomeActionPolicy(
                core,
                feature_width,
                actions,
                readout_width=readout_width,
                seed=seed,
            )
            optimizer = _Adam(policy.parameters, learning_rate)
            rng = np.random.default_rng(seed)
            losses: list[float] = []
            gradients: list[float] = []
            for _ in range(epochs):
                order = rng.permutation(len(splits["train"]))
                epoch_loss = 0.0
                epoch_gradient = 0.0
                for position in order:
                    encoded = _encoded_episode(
                        splits["train"][int(position)], encoder, action_index, reward_scale
                    )
                    loss, gradient = policy.train_episode(*encoded, optimizer)
                    epoch_loss += loss
                    epoch_gradient += gradient
                losses.append(epoch_loss / len(splits["train"]))
                gradients.append(epoch_gradient / len(splits["train"]))
            checkpoint = None
            if checkpoint_dir is not None:
                checkpoint = save_connectome_checkpoint(
                    Path(checkpoint_dir) / f"{arm}-seed-{seed}.npz",
                    policy,
                    encoder,
                    seed=seed,
                    signed=signed,
                )
            trials.append(
                {
                    "seed": int(seed),
                    "arm": arm,
                    "graph": core.metadata,
                    "parameters": policy.parameter_receipt(),
                    "training": {
                        "first_epoch_loss": round(losses[0], 8),
                        "final_epoch_loss": round(losses[-1], 8),
                        "final_mean_gradient_norm": round(gradients[-1], 8),
                    },
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "metrics": {
                        split: _metrics(policy, rows, encoder, action_index)
                        for split, rows in splits.items()
                    },
                }
            )

        baseline = LinearActionPolicy(feature_width, actions, seed=seed)
        optimizer = _Adam(baseline.parameters, learning_rate)
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(epochs):
            order = rng.permutation(len(splits["train"]))
            epoch_loss = 0.0
            for position in order:
                encoded = _encoded_episode(
                    splits["train"][int(position)], encoder, action_index, reward_scale
                )
                loss, _ = baseline.train_episode(*encoded, optimizer)
                epoch_loss += loss
            losses.append(epoch_loss / len(splits["train"]))
        checkpoint = None
        if checkpoint_dir is not None:
            checkpoint = save_linear_checkpoint(
                Path(checkpoint_dir) / f"observation_only-seed-{seed}.npz",
                baseline,
                encoder,
                seed=seed,
            )
        trials.append(
            {
                "seed": int(seed),
                "arm": "observation_only",
                "graph": None,
                "parameters": baseline.parameter_receipt(),
                "training": {
                    "first_epoch_loss": round(losses[0], 8),
                    "final_epoch_loss": round(losses[-1], 8),
                },
                "checkpoint": str(checkpoint) if checkpoint else None,
                "metrics": {
                    split: _metrics(baseline, rows, encoder, action_index)
                    for split, rows in splits.items()
                },
            }
        )

    arms = ("biological", "weight_shuffled", "degree_rewired", "observation_only")
    return {
        "schema": REPORT_SCHEMA,
        "status": "trained_offline_policy_experiment",
        "question": "Does connectome topology improve learned software-agent action selection?",
        "actions": list(actions),
        "protocol": {
            "training": "reward-weighted behavioral cloning with full-episode BPTT",
            "rewards": "discounted verifier rewards supplied by trajectory records",
            "feature_fit": "train split only",
            "graph_topology": "fixed in every arm",
            "trainable_connectome_components": "flow-conditioned recurrent dynamics and action decoder",
            "controls": ["weight_shuffled", "degree_rewired", "observation_only"],
            "paired_initialization": True,
            "seeds": [int(seed) for seed in seeds],
            "epochs": epochs,
            "learning_rate": learning_rate,
            "reward_scale": reward_scale,
            "feature_width": feature_width,
            "readout_width": readout_width,
            "split_episodes": {name: len(rows) for name, rows in splits.items()},
        },
        "base_graph": base_core.metadata,
        "summary": _summarize(trials, arms),
        "trials": trials,
        "interpretation_boundary": (
            "This trains a connectome-constrained software-agent policy, not a biological fly or a "
            "language model. Held-out action accuracy measures imitation of recorded agent decisions; "
            "repository repair/review claims require independent verifier rewards and live rollouts."
        ),
    }


def classify_codex_item(item: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Map a completed Codex JSONL item to BugBrain's deliberately small action space."""
    kind = str(item.get("type") or "")
    receipt: dict[str, Any] = {"item_type": kind}
    if kind == "command_execution":
        command = item.get("command")
        if isinstance(command, list):
            rendered = " ".join(str(part) for part in command)
        else:
            rendered = str(command or "")
        lowered = rendered.casefold()
        receipt.update(
            {
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
                "output": str(item.get("aggregated_output") or "")[-2_000:],
            }
        )
        if re.search(
            r"(?:^|[\s\"'])(?:pytest|cargo test|go test|npm test|npm run test|unittest|gradle test)",
            lowered,
        ):
            return "test", receipt
        if re.search(r"(?:^|[\s\"'])(?:rg|grep|find|git grep|fd)(?:\s|$)", lowered):
            return "search", receipt
        if re.search(
            r"(?:^|[\s\"'])(?:git diff|git status|git show)(?:\s|$)", lowered
        ):
            return "verify", receipt
        return "inspect", receipt
    if kind in {"file_change", "file_write", "apply_patch"}:
        return "patch", receipt
    if kind in {"web_search"}:
        return "search", receipt
    if kind in {"mcp_tool_call", "tool_call"}:
        name = str(item.get("tool") or item.get("name") or "").casefold()
        receipt["tool"] = name
        if any(token in name for token in ("search", "find", "list")):
            return "search", receipt
        if any(token in name for token in ("test", "check", "verify")):
            return "verify", receipt
        return "inspect", receipt
    if kind in {"agent_message", "message"}:
        return "stop", receipt
    return None, receipt


def import_codex_jsonl(
    path: str | Path,
    *,
    key: str,
    split: str,
    goal: str,
    final_reward: float,
) -> dict[str, Any]:
    """Convert `codex exec --json` output without leaking the current command into its input."""
    split = split.casefold()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    completed_items: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Codex JSONL at line {line_number}") from exc
        if event.get("type") == "item.completed" and isinstance(event.get("item"), dict):
            completed_items.append(event["item"])

    terminal_messages = [
        index
        for index, item in enumerate(completed_items)
        if str(item.get("type") or "") in {"agent_message", "message"}
    ]
    terminal_message = terminal_messages[-1] if terminal_messages else None
    steps: list[dict[str, Any]] = []
    previous: dict[str, Any] = {"status": "episode_started"}
    for item_index, item in enumerate(completed_items):
        if (
            str(item.get("type") or "") in {"agent_message", "message"}
            and item_index != terminal_message
        ):
            continue
        action, receipt = classify_codex_item(item)
        if action is None:
            continue
        observation = {
            "goal": goal,
            "step": len(steps),
            "previous_action": steps[-1]["action"] if steps else "none",
            "previous_result": previous,
        }
        steps.append({"observation": observation, "action": action, "reward": 0.0})
        previous = receipt
    if not steps:
        raise ValueError("Codex JSONL contained no completed actionable items")
    steps[-1]["reward"] = float(final_reward)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "actions": list(DEFAULT_ACTIONS),
        "episodes": [
            {
                "key": key,
                "split": split,
                "metadata": {
                    "source": "codex exec --json",
                    "source_path": str(path),
                    "reward_source": "operator-supplied verifier outcome",
                },
                "steps": steps,
            }
        ],
    }
