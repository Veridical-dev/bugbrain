from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise RuntimeError(
        "The reservoir POC requires NumPy. Install the project with `pip install -e .`."
    ) from exc

from .connectome import Connectome
from .diff_parser import parse_review_input
from .model import CodeUnit, NeuronAnnotation
from .router import TAG_TO_LENS


RISK_LABELS = tuple(TAG_TO_LENS)
_PATH_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$-]{1,}")
REFERENCE_IMPLEMENTATION = {
    "model_family": "echo-state network / reservoir computing",
    "source": "https://gitlab.com/EuropeanSpaceAgency/fly_connectome",
    "commit": "90679d672cd25905758b3abc6674fa6a28525d52",
    "adaptation": (
        "independent code hunks replace time-series samples; fixed random input projection, "
        "tanh recurrence, leaky update, spectral-radius normalization, and ridge readout "
        "follow the reference reservoir formulation"
    ),
}


@dataclass(frozen=True, slots=True)
class CorpusCase:
    key: str
    path: Path
    units: tuple[CodeUnit, ...]


@dataclass(frozen=True, slots=True)
class CoreGraph:
    name: str
    raw_matrix: np.ndarray
    matrix: np.ndarray
    selected_full_indices: np.ndarray
    raw_spectral_radius: float
    spectral_radius: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    thresholds: np.ndarray

    def scores(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.concatenate(
            (np.ones((len(normalized), 1), dtype=np.float64), normalized), axis=1
        )
        return design @ self.weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scores(features) >= self.thresholds


def _stable_digest(value: str, seed: int) -> bytes:
    return hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=16).digest()


def _unit_features(unit: CodeUnit) -> set[str]:
    """Feature set independent of the heuristic risk labels used as calibration targets."""
    features = {f"token:{token}" for token in unit.tokens}
    normalized_path = unit.path.replace("\\", "/").lower()
    parts = [part for part in normalized_path.split("/") if part]
    features.update(f"path:{part}" for part in parts)
    if parts:
        suffix = Path(parts[-1]).suffix
        if suffix:
            features.add(f"suffix:{suffix}")
    for raw in _PATH_TOKEN.findall(unit.heading):
        features.add(f"heading:{raw.lower()}")
    return features or {f"unit:{unit.key}"}


class FeatureHasher:
    """Deterministic signed feature hashing with train-fold-only IDF fitting."""

    def __init__(self, width: int = 96, seed: int = 1714) -> None:
        if width < 8:
            raise ValueError("feature width must be at least 8")
        self.width = width
        self.seed = seed
        self.idf = np.ones(width, dtype=np.float64)

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = _stable_digest(feature, self.seed)
        index = int.from_bytes(digest[:8], "little") % self.width
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def fit(self, units: Sequence[CodeUnit]) -> FeatureHasher:
        document_frequency = np.zeros(self.width, dtype=np.float64)
        for unit in units:
            buckets = {self._bucket(feature)[0] for feature in _unit_features(unit)}
            if buckets:
                document_frequency[list(buckets)] += 1.0
        self.idf = np.log((1.0 + len(units)) / (1.0 + document_frequency)) + 1.0
        return self

    def transform(self, units: Sequence[CodeUnit]) -> np.ndarray:
        output = np.zeros((len(units), self.width), dtype=np.float64)
        for row, unit in enumerate(units):
            for feature in _unit_features(unit):
                index, sign = self._bucket(feature)
                output[row, index] += sign * self.idf[index]
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return output / norms


def load_corpus(path: str | Path) -> list[CorpusCase]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Corpus manifest must contain at least two cases")
    cases: list[CorpusCase] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("key") or not row.get("path"):
            raise ValueError("Every corpus case requires key and path")
        key = str(row["key"])
        if key in seen:
            raise ValueError(f"Duplicate corpus key: {key}")
        seen.add(key)
        input_path = Path(str(row["path"]))
        if not input_path.is_absolute():
            input_path = (manifest_path.parent / input_path).resolve()
        units = tuple(parse_review_input(input_path))
        if not units:
            raise ValueError(f"Corpus case has no parsed hunks: {key}")
        cases.append(CorpusCase(key=key, path=input_path, units=units))
    return cases


def _estimate_radius_nonnegative(matrix: np.ndarray, iterations: int = 160) -> float:
    """Perron power iteration for the unsigned non-negative reservoir matrices."""
    if matrix.size == 0 or not np.any(matrix):
        return 0.0
    vector = np.full(matrix.shape[0], 1.0 / math.sqrt(matrix.shape[0]), dtype=np.float64)
    previous = 0.0
    estimate = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        norm = float(np.linalg.norm(product))
        if norm == 0.0:
            return 0.0
        vector = product / norm
        next_product = matrix @ vector
        estimate = float(np.dot(vector, next_product))
        if estimate > 0 and abs(estimate - previous) <= 1e-9 * max(1.0, estimate):
            break
        previous = estimate
    return abs(estimate)


def _scaled_core(
    name: str,
    raw_matrix: np.ndarray,
    selected: np.ndarray,
    *,
    target_radius: float,
    metadata: dict[str, Any],
) -> CoreGraph:
    radius = _estimate_radius_nonnegative(raw_matrix)
    if radius <= 0.0:
        raise ValueError(f"Reservoir arm {name!r} has no recurrent activity")
    matrix = raw_matrix * (target_radius / radius)
    measured = _estimate_radius_nonnegative(matrix)
    digest = hashlib.sha256(raw_matrix.astype("<f8", copy=False).tobytes()).hexdigest()
    return CoreGraph(
        name=name,
        raw_matrix=raw_matrix,
        matrix=matrix,
        selected_full_indices=selected,
        raw_spectral_radius=radius,
        spectral_radius=measured,
        metadata={
            **metadata,
            "nodes": int(raw_matrix.shape[0]),
            "edges": int(np.count_nonzero(raw_matrix)),
            "raw_spectral_radius": round(radius, 9),
            "scaled_spectral_radius": round(measured, 9),
            "target_spectral_radius": target_radius,
            "matrix_sha256": digest,
        },
    )


def build_biological_core(
    graph: Connectome,
    *,
    node_count: int = 512,
    target_radius: float = 0.99,
) -> CoreGraph:
    """Select the most connected neurons and build target-by-source recurrent weights."""
    if node_count < 4 or node_count > graph.node_count:
        raise ValueError("node_count must be between 4 and the connectome node count")
    targets = np.asarray(graph.targets, dtype=np.int64)
    out_degree = np.diff(np.asarray(graph.offsets, dtype=np.int64))
    in_degree = np.bincount(targets, minlength=graph.node_count)
    degree = out_degree + in_degree
    neuron_ids = np.asarray(graph.neuron_ids, dtype=np.uint64)
    order = np.lexsort((neuron_ids, -degree))
    selected = order[:node_count].astype(np.int64, copy=False)

    reverse = np.full(graph.node_count, -1, dtype=np.int64)
    reverse[selected] = np.arange(node_count, dtype=np.int64)
    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    weights = np.asarray(graph.weights, dtype=np.float64)
    for source_local, source_full in enumerate(selected):
        start, stop = offsets[source_full], offsets[source_full + 1]
        target_local = reverse[targets[start:stop]]
        keep = target_local >= 0
        if np.any(keep):
            np.add.at(matrix[:, source_local], target_local[keep], weights[start:stop][keep])
    return _scaled_core(
        "biological",
        matrix,
        selected,
        target_radius=target_radius,
        metadata={
            "selection": "highest total directed edge degree in biological graph",
            "orientation": "matrix[target, source]",
            "weight": "retained synapse count",
        },
    )


def make_matched_controls(
    biological: CoreGraph,
    *,
    seed: int = 783,
    target_radius: float = 0.99,
) -> dict[str, CoreGraph | None]:
    """Create topology and weight ablations with equal N, E and weight multiset."""
    rng = np.random.default_rng(seed)
    base = biological.raw_matrix
    positions = np.flatnonzero(base)
    weights = base.flat[positions].copy()
    shuffled_weights = weights.copy()
    rng.shuffle(shuffled_weights)

    weight_matrix = np.zeros_like(base)
    weight_matrix.flat[positions] = shuffled_weights
    weight_control = _scaled_core(
        "weight_shuffled",
        weight_matrix,
        biological.selected_full_indices,
        target_radius=target_radius,
        metadata={
            "control": "biological topology with globally permuted non-zero weights",
            "matched_to": "biological",
        },
    )

    node_count = base.shape[0]
    loop_count = int(np.count_nonzero(np.diag(base)))
    nonloop_count = len(positions) - loop_count
    diagonal = np.arange(node_count, dtype=np.int64) * (node_count + 1)
    if loop_count > node_count or nonloop_count > node_count * (node_count - 1):
        raise AssertionError("More edges than available matched matrix positions")
    random_loops = rng.choice(diagonal, size=loop_count, replace=False)
    off_diagonal = np.flatnonzero(~np.eye(node_count, dtype=bool))
    random_nonloops = rng.choice(off_diagonal, size=nonloop_count, replace=False)
    random_positions = np.concatenate((random_loops, random_nonloops))
    random_weights = weights.copy()
    rng.shuffle(random_weights)
    topology_matrix = np.zeros_like(base)
    topology_matrix.flat[random_positions] = random_weights
    topology_control = _scaled_core(
        "random_topology",
        topology_matrix,
        biological.selected_full_indices,
        target_radius=target_radius,
        metadata={
            "control": (
                "uniform random directed topology with biological raw-weight multiset "
                "and matched self-loop count"
            ),
            "matched_to": "biological",
            "self_loops": loop_count,
        },
    )
    return {
        "biological": biological,
        "weight_shuffled": weight_control,
        "random_topology": topology_control,
        "semantic": None,
    }


class ReservoirProjector:
    """Matched ESN input projection and independent-sample state expansion."""

    def __init__(
        self,
        node_count: int,
        feature_width: int,
        *,
        seed: int = 1714,
        input_scale: float = 0.7,
        bias_scale: float = 0.05,
        leak: float = 0.9,
        passes: int = 3,
    ) -> None:
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
        if passes < 1:
            raise ValueError("passes must be positive")
        rng = np.random.default_rng(seed)
        self.input_index = rng.integers(0, feature_width, size=node_count)
        self.input_sign = rng.choice(np.array([-1.0, 1.0]), size=node_count)
        self.bias = rng.uniform(-bias_scale, bias_scale, size=node_count)
        self.input_scale = input_scale
        self.leak = leak
        self.passes = passes

    def transform(self, semantic: np.ndarray, graph: CoreGraph | None) -> np.ndarray:
        if graph is None:
            return semantic.copy()
        drive = (
            semantic[:, self.input_index] * self.input_sign[None, :] * self.input_scale
            + self.bias[None, :]
        )
        state = np.zeros_like(drive)
        for _ in range(self.passes):
            proposal = np.tanh(state @ graph.matrix.T + drive)
            state = (1.0 - self.leak) * state + self.leak * proposal
        # ESN readouts conventionally see bias, reservoir state, and direct input.
        return np.concatenate((state, semantic), axis=1)

    @staticmethod
    def reservoir_state(representation: np.ndarray, graph: CoreGraph | None) -> np.ndarray:
        if graph is None:
            return representation
        return representation[:, : graph.matrix.shape[0]]


def targets_for(units: Sequence[CodeUnit]) -> np.ndarray:
    labels = np.zeros((len(units), len(RISK_LABELS)), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(RISK_LABELS)}
    for row, unit in enumerate(units):
        for tag in unit.risk_tags:
            if tag in lookup:
                labels[row, lookup[tag]] = 1.0
    return labels


def _binary_f1(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_readout(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-2) -> RidgeReadout:
    if len(features) != len(labels) or not len(features):
        raise ValueError("Readout requires matching non-empty feature and label rows")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (features - mean) / scale
    design = np.concatenate(
        (np.ones((len(normalized), 1), dtype=np.float64), normalized), axis=1
    )
    # Dual ridge is much cheaper here because corpus rows << reservoir features.
    gram = design @ design.T
    dual = np.linalg.solve(gram + ridge * np.eye(len(gram)), labels)
    weights = design.T @ dual
    scores = design @ weights
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    grid = np.linspace(-0.1, 1.1, 49)
    for column in range(labels.shape[1]):
        truth = labels[:, column].astype(bool)
        best = (-1.0, -1.0, -1.0, 0.5)
        for threshold in grid:
            precision, recall, f1 = _binary_f1(truth, scores[:, column] >= threshold)
            candidate = (f1, precision, recall, -abs(threshold - 0.5))
            if candidate > best:
                best = candidate
                thresholds[column] = threshold
    return RidgeReadout(mean=mean, scale=scale, weights=weights, thresholds=thresholds)


def multilabel_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth_bool = truth.astype(bool)
    prediction_bool = prediction.astype(bool)
    precision, recall, micro_f1 = _binary_f1(truth_bool, prediction_bool)
    per_label: dict[str, dict[str, float | int]] = {}
    f1s = []
    for index, label in enumerate(RISK_LABELS):
        p, r, f1 = _binary_f1(truth_bool[:, index], prediction_bool[:, index])
        f1s.append(f1)
        per_label[label] = {
            "support": int(truth_bool[:, index].sum()),
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
        }
    return {
        "rows": int(len(truth)),
        "precision_micro": round(precision, 6),
        "recall_micro": round(recall, 6),
        "f1_micro": round(micro_f1, 6),
        "f1_macro": round(float(np.mean(f1s)), 6),
        "exact_match": round(float(np.mean(np.all(truth_bool == prediction_bool, axis=1))), 6),
        "per_label": per_label,
    }


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def _file_blocks(units: Sequence[CodeUnit]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        grouped.setdefault(unit.path, []).append(index)
    return [grouped[path] for path in sorted(grouped)]


def _block_vector(values: np.ndarray, block: Sequence[int]) -> np.ndarray:
    vector = values[list(block)].mean(axis=0)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _cluster_file_blocks(
    units: Sequence[CodeUnit],
    semantic: np.ndarray,
    reservoir_state: np.ndarray,
    *,
    bundle_count: int,
    reservoir_weight: float,
    context_budget: int,
) -> list[list[int]]:
    if not 0.0 <= reservoir_weight <= 1.0:
        raise ValueError("reservoir_weight must be in [0, 1]")
    if bundle_count < 1:
        raise ValueError("bundle_count must be positive")
    blocks = _file_blocks(units)
    count = min(bundle_count, len(blocks))
    if count == 0:
        return []
    semantic_blocks = np.stack([_block_vector(semantic, block) for block in blocks])
    reservoir_blocks = np.stack([_block_vector(reservoir_state, block) for block in blocks])
    similarities = (
        (1.0 - reservoir_weight) * (semantic_blocks @ semantic_blocks.T)
        + reservoir_weight * (reservoir_blocks @ reservoir_blocks.T)
    )
    block_chars = np.array(
        [sum(len(units[index].excerpt) + 160 for index in block) for block in blocks],
        dtype=np.int64,
    )

    first = int(np.argmax(block_chars))
    centroids = [first]
    while len(centroids) < count:
        remaining = [index for index in range(len(blocks)) if index not in centroids]
        chosen = min(
            remaining,
            key=lambda index: (max(similarities[index, c] for c in centroids), index),
        )
        centroids.append(chosen)

    assignments: list[list[int]] = [[centroid] for centroid in centroids]
    loads = [int(block_chars[index]) for index in centroids]
    remaining = sorted(
        (index for index in range(len(blocks)) if index not in centroids),
        key=lambda index: (-int(block_chars[index]), index),
    )
    for block_index in remaining:
        feasible = [
            cluster
            for cluster in range(count)
            if loads[cluster] + int(block_chars[block_index]) <= context_budget
        ]
        candidates = feasible or list(range(count))
        chosen = max(
            candidates,
            key=lambda cluster: (
                similarities[block_index, centroids[cluster]]
                - 0.15 * loads[cluster] / max(1, context_budget),
                -loads[cluster],
                -cluster,
            ),
        )
        assignments[chosen].append(block_index)
        loads[chosen] += int(block_chars[block_index])
    return [
        [unit_index for block_index in cluster for unit_index in blocks[block_index]]
        for cluster in assignments
    ]


def _bounded_prompt(
    reviewer_id: str,
    lens: str,
    units: Sequence[CodeUnit],
    *,
    context_budget: int,
) -> tuple[str, list[str], list[str]]:
    header = (
        f"You are {reviewer_id}, one shadow code-review pass.\n"
        f"Primary learned lens: {lens}.\n\n"
        "A fixed fruit-fly connectome reservoir contributed only a secondary similarity signal "
        "after file-atomic code grouping. Neural state is routing metadata, never bug evidence. "
        "Review only the supplied code and return a JSON array of objects with file, line, "
        "severity, comment, and evidence. Return [] when no concrete regression is defensible.\n\n"
    )
    sections: list[str] = []
    included: list[str] = []
    truncated: list[str] = []
    used = len(header)
    for unit in units:
        prefix = f"### {unit.path}:{unit.new_line} — {unit.heading}\n```diff\n"
        suffix = "\n```\n\n"
        section = prefix + unit.excerpt + suffix
        remaining = context_budget - used
        if len(section) <= remaining:
            sections.append(section)
            included.append(unit.key)
            used += len(section)
            continue
        marker = "\n... [context truncated]\n```\n\n"
        available = remaining - len(prefix) - len(marker)
        if available > 160:
            sections.append(prefix + unit.excerpt[:available] + marker)
            truncated.append(unit.key)
            used = context_budget
        else:
            truncated.append(unit.key)
        if used >= context_budget:
            break
    prompt = (header + "".join(sections))[:context_budget]
    included_set = set(included)
    truncated.extend(unit.key for unit in units if unit.key not in included_set and unit.key not in truncated)
    return prompt, included, truncated


def _neuron_receipts(
    state: np.ndarray,
    selected: np.ndarray,
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = np.argsort(-np.abs(state), kind="stable")[:limit]
    receipts = []
    for local_index in ranked:
        full_index = int(selected[int(local_index)])
        annotation = annotations[full_index]
        receipts.append(
            {
                "root_id": int(graph.neuron_ids[full_index]),
                "activation": round(float(state[int(local_index)]), 9),
                "label": annotation.label(),
                "neurotransmitter": annotation.neurotransmitter,
            }
        )
    return receipts


def route_with_reservoir(
    units: Sequence[CodeUnit],
    semantic: np.ndarray,
    representation: np.ndarray,
    model: RidgeReadout,
    core: CoreGraph | None,
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    arm: str,
    bundle_count: int,
    reservoir_weight: float,
    context_budget: int,
) -> dict[str, Any]:
    state = ReservoirProjector.reservoir_state(representation, core)
    state = _normalized_rows(state)
    scores = model.scores(representation)
    clusters = _cluster_file_blocks(
        units,
        semantic,
        state,
        bundle_count=bundle_count,
        reservoir_weight=0.0 if core is None else reservoir_weight,
        context_budget=context_budget,
    )
    bundles = []
    total_included = 0
    total_truncated = 0
    for bundle_index, cluster in enumerate(clusters):
        selected_units = [units[index] for index in cluster]
        aggregate_scores = scores[cluster].mean(axis=0)
        label_index = int(np.argmax(aggregate_scores))
        lens = TAG_TO_LENS[RISK_LABELS[label_index]]
        reviewer_id = f"{arm}-reviewer-{bundle_index + 1:02d}"
        prompt, included, truncated = _bounded_prompt(
            reviewer_id,
            lens,
            selected_units,
            context_budget=context_budget,
        )
        total_included += len(included)
        total_truncated += len(truncated)
        if core is None:
            receipts: list[dict[str, Any]] = []
        else:
            aggregate_state = state[cluster].mean(axis=0)
            receipts = _neuron_receipts(
                aggregate_state,
                core.selected_full_indices,
                graph,
                annotations,
            )
        bundles.append(
            {
                "reviewer_id": reviewer_id,
                "lens": lens,
                "units": [asdict(unit) for unit in selected_units],
                "included_unit_keys": included,
                "truncated_unit_keys": truncated,
                "anchor": receipts[0] if receipts else None,
                "circuit_receipt": receipts,
                "prompt": prompt,
            }
        )
    routed = [unit["key"] for bundle in bundles for unit in bundle["units"]]
    expected = [unit.key for unit in units]
    if sorted(routed) != sorted(expected):
        raise AssertionError("File-atomic routing did not cover every unit exactly once")
    return {
        "arm": arm,
        "graph": {"name": arm, **(core.metadata if core is not None else {"nodes": 0, "edges": 0})},
        "parameters": {
            "bundle_count": bundle_count,
            "reservoir_similarity_weight": 0.0 if core is None else reservoir_weight,
            "context_budget": context_budget,
            "file_atomic": True,
        },
        "metrics": {
            "units": len(units),
            "files": len({unit.path for unit in units}),
            "bundles": len(bundles),
            "prompt_chars": [len(bundle["prompt"]) for bundle in bundles],
            "total_prompt_chars": sum(len(bundle["prompt"]) for bundle in bundles),
            "fully_included_units": total_included,
            "truncated_or_omitted_units": total_truncated,
        },
        "bundles": bundles,
    }


def grade_evidence_colocation(route: dict[str, Any], evidence_path: str | Path) -> dict[str, Any]:
    """Grade routing only; truth evidence is never passed to the router or reviewer prompts."""
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    all_units = [unit for bundle in route["bundles"] for unit in bundle["units"]]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for unit in all_units:
        by_path.setdefault(str(unit["path"]), []).append(unit)
    bundle_sets = [{unit["key"] for unit in bundle["units"]} for bundle in route["bundles"]]
    prompt_sets = [set(bundle.get("included_unit_keys") or []) for bundle in route["bundles"]]
    rows = []
    pair_hits = pair_total = 0
    prompt_pair_hits = 0
    for truth in evidence.get("truth", []):
        if truth.get("truth_register") == "audit":
            continue
        resolved: list[str] = []
        unresolved: list[str] = []
        for source in truth.get("source", []):
            raw = str(source)
            path, separator, line_text = raw.rpartition(":")
            if not separator or not line_text.isdigit() or path not in by_path:
                unresolved.append(raw)
                continue
            line = int(line_text)
            candidates = by_path[path]
            closest = min(candidates, key=lambda unit: (abs(int(unit["new_line"]) - line), unit["key"]))
            resolved.append(str(closest["key"]))
        keys = sorted(set(resolved))
        pairs = [(keys[a], keys[b]) for a in range(len(keys)) for b in range(a + 1, len(keys))]
        local_hits = sum(any(a in group and b in group for group in bundle_sets) for a, b in pairs)
        local_prompt_hits = sum(any(a in group and b in group for group in prompt_sets) for a, b in pairs)
        pair_hits += local_hits
        prompt_pair_hits += local_prompt_hits
        pair_total += len(pairs)
        rows.append(
            {
                "truth_id": truth.get("truth_id"),
                "resolved_units": keys,
                "unresolved_sources": unresolved,
                "all_evidence_colocated": bool(keys) and any(set(keys) <= group for group in bundle_sets),
                "all_evidence_fully_in_prompt": bool(keys) and any(set(keys) <= group for group in prompt_sets),
                "pair_coverage": round(local_hits / len(pairs), 6) if pairs else (1.0 if keys else 0.0),
                "prompt_pair_coverage": round(local_prompt_hits / len(pairs), 6) if pairs else (1.0 if keys else 0.0),
            }
        )
    return {
        "truth_roots": len(rows),
        "fully_colocated_roots": sum(row["all_evidence_colocated"] for row in rows),
        "fully_prompt_covered_roots": sum(row["all_evidence_fully_in_prompt"] for row in rows),
        "evidence_pair_coverage": round(pair_hits / pair_total, 6) if pair_total else 0.0,
        "prompt_evidence_pair_coverage": round(prompt_pair_hits / pair_total, 6) if pair_total else 0.0,
        "roots": rows,
    }


def run_reservoir_poc(
    cases: Sequence[CorpusCase],
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    holdout_key: str,
    node_count: int = 512,
    feature_width: int = 96,
    passes: int = 3,
    leak: float = 0.9,
    input_scale: float = 0.7,
    target_radius: float = 0.99,
    ridge: float = 1e-2,
    seed: int = 1714,
    bundle_count: int = 8,
    reservoir_weight: float = 0.25,
    context_budget: int = 30_000,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(cases) < 2:
        raise ValueError("Reservoir POC requires at least two corpus cases")
    if len(annotations) != graph.node_count:
        raise ValueError("Annotation order/count differs from graph")
    holdout = next((case for case in cases if case.key == holdout_key), None)
    if holdout is None:
        raise ValueError(f"Unknown holdout case: {holdout_key}")
    train_units = tuple(unit for case in cases if case.key != holdout_key for unit in case.units)
    if not train_units:
        raise ValueError("Holdout split left no training units")
    encoder = FeatureHasher(feature_width, seed).fit(train_units)
    projector = ReservoirProjector(
        node_count,
        feature_width,
        seed=seed,
        input_scale=input_scale,
        leak=leak,
        passes=passes,
    )
    biological = build_biological_core(graph, node_count=node_count, target_radius=target_radius)
    arms = make_matched_controls(biological, seed=seed, target_radius=target_radius)
    train_semantic = encoder.transform(train_units)
    test_semantic = encoder.transform(holdout.units)
    train_targets = targets_for(train_units)
    test_targets = targets_for(holdout.units)

    routes: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for arm_name, core in arms.items():
        train_representation = projector.transform(train_semantic, core)
        test_representation = projector.transform(test_semantic, core)
        model = fit_readout(train_representation, train_targets, ridge=ridge)
        metrics = multilabel_metrics(test_targets, model.predict(test_representation))
        route = route_with_reservoir(
            holdout.units,
            test_semantic,
            test_representation,
            model,
            core,
            graph,
            annotations,
            arm=arm_name,
            bundle_count=bundle_count,
            reservoir_weight=reservoir_weight,
            context_budget=context_budget,
        )
        if evidence_path is not None:
            route["evidence_colocation"] = grade_evidence_colocation(route, evidence_path)
        routes[arm_name] = route
        calibration[arm_name] = metrics

    return {
        "schema": "bugbrain.reservoir-review/2",
        "status": "shadow_only_not_a_public_quality_claim",
        "hypothesis": (
            "A fixed fly-connectome reservoir supplies a useful secondary routing bias after "
            "code-native file grouping, compared with matched topology/weight ablations."
        ),
        "holdout": holdout.key,
        "training_cases": [case.key for case in cases if case.key != holdout.key],
        "calibration_task": (
            "multi-label reviewer-lens prediction; labels are deterministic code-risk heuristics, "
            "not defect truth and not a promotion metric"
        ),
        "implementation_provenance": REFERENCE_IMPLEMENTATION,
        "data_provenance": {
            "dataset": graph.metadata.get("dataset"),
            "connections_md5": graph.metadata.get("connections_md5"),
            "annotations_md5": graph.metadata.get("annotations_md5"),
            "min_synapses": graph.metadata.get("min_synapses"),
            "input_cases": [
                {"key": case.key, "path": str(case.path), "units": len(case.units)}
                for case in cases
            ],
        },
        "contract": {
            "corpus_cases": len(cases),
            "training_units": len(train_units),
            "holdout_units": len(holdout.units),
            "feature_width": feature_width,
            "reservoir_nodes": node_count,
            "passes": passes,
            "leak": leak,
            "input_scale": input_scale,
            "target_spectral_radius": target_radius,
            "ridge": ridge,
            "seed": seed,
            "bundle_count": bundle_count,
            "reservoir_similarity_weight": reservoir_weight,
            "context_budget": context_budget,
            "truth_used_for_routing": False,
        },
        "controls": {
            "same_input_projection": True,
            "same_readout_parameter_count_across_graph_arms": True,
            "semantic_baseline_has_smaller_readout": True,
            "same_node_count": True,
            "same_nonzero_edge_count": True,
            "same_raw_nonzero_weight_multiset_before_spectral_scaling": True,
            "same_self_loop_count": True,
            "each_graph_scaled_to_same_spectral_radius": True,
            "truth_labels_used_to_fit_router": False,
            "code_risk_heuristics_used_only_for_lens_calibration": True,
        },
        "claim_boundary": (
            "Calibration measures heuristic lens prediction, not bug finding. Evidence colocation "
            "is computed after routing. Only a separate frozen-truth LLM evaluation can measure "
            "review efficacy."
        ),
        "calibration": calibration,
        "arms": routes,
    }


def run_reservoir_benchmark(
    cases: Sequence[CorpusCase],
    graph: Connectome,
    annotations: Sequence[NeuronAnnotation],
    *,
    seeds: Sequence[int],
    node_count: int = 512,
    feature_width: int = 96,
    passes: int = 3,
    leak: float = 0.9,
    input_scale: float = 0.7,
    target_radius: float = 0.99,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Run paired leave-one-project-out mechanism checks over several random seeds."""
    if not seeds:
        raise ValueError("Benchmark requires at least one seed")
    trials: list[dict[str, Any]] = []
    for holdout in cases:
        for seed in seeds:
            result = run_reservoir_poc(
                cases,
                graph,
                annotations,
                holdout_key=holdout.key,
                node_count=node_count,
                feature_width=feature_width,
                passes=passes,
                leak=leak,
                input_scale=input_scale,
                target_radius=target_radius,
                ridge=ridge,
                seed=seed,
                bundle_count=1,
                reservoir_weight=0.25,
                context_budget=30_000,
            )
            trials.append(
                {
                    "holdout": holdout.key,
                    "seed": seed,
                    "calibration": result["calibration"],
                }
            )

    arm_names = tuple(trials[0]["calibration"])
    aggregate: dict[str, Any] = {}
    for arm in arm_names:
        metrics: dict[str, Any] = {}
        for metric in ("f1_micro", "f1_macro", "exact_match"):
            values = np.asarray(
                [trial["calibration"][arm][metric] for trial in trials], dtype=np.float64
            )
            metrics[metric] = {
                "mean": round(float(values.mean()), 6),
                "population_std": round(float(values.std()), 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }
        aggregate[arm] = metrics

    by_holdout: dict[str, Any] = {}
    for holdout in (case.key for case in cases):
        holdout_trials = [trial for trial in trials if trial["holdout"] == holdout]
        by_holdout[holdout] = {
            arm: {
                metric: round(
                    float(np.mean([trial["calibration"][arm][metric] for trial in holdout_trials])),
                    6,
                )
                for metric in ("f1_micro", "f1_macro", "exact_match")
            }
            for arm in arm_names
        }

    biological_deltas: dict[str, Any] = {}
    for arm in arm_names:
        if arm == "biological":
            continue
        deltas = np.asarray(
            [
                trial["calibration"]["biological"]["f1_micro"]
                - trial["calibration"][arm]["f1_micro"]
                for trial in trials
            ],
            dtype=np.float64,
        )
        seed_blocks = np.asarray(
            [
                np.mean(
                    [
                        trial["calibration"]["biological"]["f1_micro"]
                        - trial["calibration"][arm]["f1_micro"]
                        for trial in trials
                        if trial["seed"] == seed
                    ]
                )
                for seed in seeds
            ],
            dtype=np.float64,
        )
        bootstrap_rng = np.random.default_rng(783)
        draws = seed_blocks[
            bootstrap_rng.integers(0, len(seed_blocks), size=(10_000, len(seed_blocks)))
        ].mean(axis=1)
        lower, upper = np.quantile(draws, (0.025, 0.975))
        biological_deltas[arm] = {
            "mean_paired_f1_micro_delta": round(float(deltas.mean()), 6),
            "seed_block_bootstrap_95_interval": [
                round(float(lower), 6),
                round(float(upper), 6),
            ],
            "wins": int(np.count_nonzero(deltas > 1e-12)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            "losses": int(np.count_nonzero(deltas < -1e-12)),
        }

    return {
        "schema": "bugbrain.reservoir-benchmark/1",
        "status": "mechanism_check_not_a_review_quality_claim",
        "implementation_provenance": REFERENCE_IMPLEMENTATION,
        "protocol": {
            "split": "leave one project out",
            "holdouts": [case.key for case in cases],
            "seeds": list(seeds),
            "paired_trials": len(trials),
            "labels": "deterministic code-risk heuristics; no defect truth",
            "reservoir_nodes": node_count,
            "feature_width": feature_width,
            "passes": passes,
            "leak": leak,
            "input_scale": input_scale,
            "target_spectral_radius": target_radius,
            "ridge": ridge,
        },
        "aggregate": aggregate,
        "by_holdout": by_holdout,
        "biological_deltas": biological_deltas,
        "uncertainty_note": (
            "Intervals are descriptive percentile bootstraps over random-seed blocks, with all "
            "project holdouts kept together; they are not a formal population-level significance test."
        ),
        "trials": trials,
    }
