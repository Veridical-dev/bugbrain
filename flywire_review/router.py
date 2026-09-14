from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .connectome import Connectome
from .model import Bundle, CodeUnit, NeuronAnnotation, NeuronReceipt, RouteResult


LENSES = (
    "security and trust boundaries",
    "state transitions and invariants",
    "concurrency, ordering, and retries",
    "error handling and recovery",
    "API and compatibility contracts",
    "data flow and serialization",
    "tests, observability, and silent failures",
    "cross-file propagation and consistency",
)

TAG_TO_LENS = {
    "security/trust-boundaries": LENSES[0],
    "state/invariants": LENSES[1],
    "concurrency/ordering": LENSES[2],
    "errors/recovery": LENSES[3],
    "api/contracts": LENSES[4],
    "data/serialization": LENSES[5],
    "tests/observability": LENSES[6],
    "dependencies/supply-chain": LENSES[7],
}


def _stable_u64(value: str, seed: int, lane: int = 0) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{lane}\0{value}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def sensory_pool(graph: Connectome, annotations: list[NeuronAnnotation]) -> list[int]:
    indices = [
        index
        for index, annotation in enumerate(annotations)
        if annotation.super_class == "sensory" or annotation.flow == "afferent"
    ]
    if not indices:
        indices = [index for index, strength in enumerate(graph.out_strength) if strength > 0]
    if not indices:
        raise ValueError("Connectome has no usable input neurons")
    return indices


def output_pool(annotations: list[NeuronAnnotation]) -> set[int]:
    return {
        index
        for index, annotation in enumerate(annotations)
        if annotation.flow == "efferent"
        or annotation.super_class in {"descending", "motor", "endocrine"}
    }


def encode_unit(unit: CodeUnit, pool: list[int], *, seed: int, seeds_per_token: int = 2) -> dict[int, float]:
    features = set(unit.tokens)
    features.update(part for part in unit.path.replace("\\", "/").split("/") if part)
    features.update(unit.risk_tags)
    if not features:
        features.add(unit.key)
    activations: dict[int, float] = {}
    for feature in sorted(features):
        for lane in range(seeds_per_token):
            index = pool[_stable_u64(feature, seed, lane) % len(pool)]
            activations[index] = activations.get(index, 0.0) + 1.0
    norm = sum(activations.values()) or 1.0
    return {index: value / norm for index, value in activations.items()}


def _hash_fingerprint(unit: CodeUnit, *, seed: int, width: int = 4096) -> dict[int, float]:
    values: dict[int, float] = {}
    for token in unit.tokens or (unit.key,):
        for lane in range(2):
            index = _stable_u64(token, seed, lane) % width
            values[index] = values.get(index, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {index: value / norm for index, value in values.items()}


def _magnitude(values: dict[int, float], *, limit: int = 256) -> dict[int, float]:
    winners = sorted(values, key=lambda index: (-abs(values[index]), index))[:limit]
    return {index: abs(values[index]) for index in winners}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _cluster(fingerprints: list[dict[int, float]], bundle_count: int) -> tuple[list[list[int]], list[int]]:
    if not fingerprints:
        return [], []
    count = min(bundle_count, len(fingerprints))
    centroids = [max(range(len(fingerprints)), key=lambda index: (len(fingerprints[index]), -index))]
    while len(centroids) < count:
        remaining = [index for index in range(len(fingerprints)) if index not in centroids]
        next_centroid = min(
            remaining,
            key=lambda index: (
                max(_cosine(fingerprints[index], fingerprints[chosen]) for chosen in centroids),
                index,
            ),
        )
        centroids.append(next_centroid)

    clusters: list[list[int]] = [[] for _ in centroids]
    max_cluster_size = math.ceil(len(fingerprints) / count)
    for unit_index, fingerprint in enumerate(fingerprints):
        scores = [_cosine(fingerprint, fingerprints[centroid]) for centroid in centroids]
        candidates = sorted(
            range(len(centroids)),
            key=lambda index: (-scores[index], len(clusters[index]), index),
        )
        chosen = next(index for index in candidates if len(clusters[index]) < max_cluster_size)
        clusters[chosen].append(unit_index)
    return clusters, centroids


def _choose_lens(units: Iterable[CodeUnit], fallback: int, used: set[str]) -> str:
    tags = Counter(tag for unit in units for tag in unit.risk_tags)
    ordered_tags = sorted(tags, key=lambda candidate: (-tags[candidate], candidate))
    start = fallback % len(LENSES)
    candidates = [TAG_TO_LENS[tag] for tag in ordered_tags if tag in TAG_TO_LENS]
    candidates.extend(LENSES[(start + offset) % len(LENSES)] for offset in range(len(LENSES)))
    for candidate in candidates:
        if candidate not in used:
            return candidate
    return candidates[0]


def _receipts(
    aggregate: dict[int, float],
    graph: Connectome,
    annotations: list[NeuronAnnotation],
    outputs: set[int],
    *,
    limit: int = 5,
) -> tuple[NeuronReceipt | None, tuple[NeuronReceipt, ...]]:
    ranked = sorted(aggregate, key=lambda index: (-abs(aggregate[index]), index))
    preferred = [index for index in ranked if index in outputs]
    anchor_index = (preferred or ranked or [None])[0]

    def receipt(index: int) -> NeuronReceipt:
        annotation = annotations[index]
        return NeuronReceipt(
            root_id=int(graph.neuron_ids[index]),
            activation=round(float(aggregate[index]), 10),
            label=annotation.label(),
            neurotransmitter=annotation.neurotransmitter,
        )

    anchor = receipt(anchor_index) if anchor_index is not None else None
    return anchor, tuple(receipt(index) for index in ranked[:limit])


def _prompt(
    reviewer_id: str,
    lens: str,
    units: tuple[CodeUnit, ...],
    anchor: NeuronReceipt | None,
    *,
    context_budget: int,
) -> str:
    per_unit_budget = max(400, (context_budget - 2_000) // max(1, len(units)))
    sections = []
    for unit in units:
        tags = ", ".join(unit.risk_tags) or "no heuristic tag"
        excerpt = unit.excerpt
        if len(excerpt) > per_unit_budget:
            excerpt = excerpt[: per_unit_budget - 26] + "\n... [context truncated]"
        sections.append(
            f"### {unit.path}:{unit.new_line} — {unit.heading}\n"
            f"Routing tags: {tags}\n```diff\n{excerpt}\n```"
        )
    anchor_text = str(anchor.root_id) if anchor else "none"
    return (
        f"You are {reviewer_id}, one decorrelated BugBrain review circuit.\n"
        f"Primary lens: {lens}.\n\n"
        "The code context below was grouped by a fixed fruit-fly connectome routing experiment "
        f"(circuit anchor {anchor_text}). Neural activation is routing metadata, NOT evidence that a bug exists. "
        "Every finding must be justified from the code.\n\n"
        "Review for concrete, newly introduced defects. Trace behavior across the supplied hunks, avoid style comments, "
        "and state the triggering input or execution path. Return a JSON array of objects with file, line, severity, "
        "comment, and evidence. Put the concise title and full explanation together in comment. Return [] if there is "
        "no defensible defect.\n\n"
        + "\n\n".join(sections)
    )


def route_units(
    units: list[CodeUnit],
    graph: Connectome,
    annotations: list[NeuronAnnotation],
    *,
    arm: str = "biological",
    bundle_count: int = 6,
    seed: int = 783,
    steps: int = 5,
    active_width: int = 2_048,
    restart: float = 0.15,
    signed: bool = False,
    context_budget: int = 30_000,
) -> RouteResult:
    if arm not in {"biological", "shuffled", "hash"}:
        raise ValueError("arm must be biological, shuffled, or hash")
    if arm == "shuffled":
        route_graph = graph.shuffled_targets(seed)
    else:
        route_graph = graph

    input_indices = sensory_pool(route_graph, annotations)
    outputs = output_pool(annotations)
    signed_fingerprints: list[dict[int, float]] = []
    fingerprints: list[dict[int, float]] = []
    for unit in units:
        if arm == "hash":
            activation = _hash_fingerprint(unit, seed=seed)
        else:
            seeds = encode_unit(unit, input_indices, seed=seed)
            activation = route_graph.propagate(
                seeds,
                steps=steps,
                active_width=active_width,
                restart=restart,
                signed=signed,
            )
            # Restart keeps sparse walks alive, but direct sensory activity would make
            # every experimental arm mostly an encoder comparison. Route on downstream
            # activity so biological versus shuffled wiring is load-bearing.
            activation = {index: value for index, value in activation.items() if index not in seeds}
        signed_fingerprints.append(activation)
        fingerprints.append(_magnitude(activation))

    clusters, centroids = _cluster(fingerprints, bundle_count)
    bundles: list[Bundle] = []
    used_lenses: set[str] = set()
    for bundle_index, cluster in enumerate(clusters):
        selected_units = tuple(units[index] for index in cluster)
        aggregate: dict[int, float] = {}
        for unit_index in cluster:
            for neuron_index, value in signed_fingerprints[unit_index].items():
                aggregate[neuron_index] = aggregate.get(neuron_index, 0.0) + value
        if arm == "hash":
            anchor = None
            circuit_receipt = ()
            fallback = centroids[bundle_index]
        else:
            anchor, circuit_receipt = _receipts(aggregate, route_graph, annotations, outputs)
            fallback = anchor.root_id if anchor else centroids[bundle_index]
        lens = _choose_lens(selected_units, fallback, used_lenses)
        used_lenses.add(lens)
        reviewer_id = f"{arm}-reviewer-{bundle_index + 1:02d}"
        bundles.append(
            Bundle(
                reviewer_id=reviewer_id,
                lens=lens,
                units=selected_units,
                anchor=anchor,
                circuit_receipt=circuit_receipt,
                prompt=_prompt(
                    reviewer_id,
                    lens,
                    selected_units,
                    anchor,
                    context_budget=context_budget,
                ),
            )
        )

    sizes = [len(bundle.units) for bundle in bundles]
    pair_similarities = [
        _cosine(fingerprints[left], fingerprints[right])
        for left in range(len(fingerprints))
        for right in range(left + 1, len(fingerprints))
    ]
    metrics = {
        "units": len(units),
        "bundles": len(bundles),
        "bundle_sizes": sizes,
        "bundle_size_stdev": round(statistics.pstdev(sizes), 6) if sizes else 0.0,
        "mean_unit_fingerprint_similarity": round(statistics.fmean(pair_similarities), 8) if pair_similarities else 0.0,
        "distinct_lenses": len({bundle.lens for bundle in bundles}),
        "sensory_pool": len(input_indices),
        "output_pool": len(outputs),
        "prompt_chars": [len(bundle.prompt) for bundle in bundles],
        "total_prompt_chars": sum(len(bundle.prompt) for bundle in bundles),
    }
    parameters = {
        "seed": seed,
        "steps": steps,
        "active_width": active_width,
        "restart": restart,
        "signed": signed,
        "bundle_count": bundle_count,
        "context_budget": context_budget,
    }
    return RouteResult(
        arm=arm,
        graph={**route_graph.metadata, "nodes": route_graph.node_count, "edges": route_graph.edge_count},
        parameters=parameters,
        bundles=tuple(bundles),
        metrics=metrics,
    )


def coassignment_jaccard(left: RouteResult, right: RouteResult) -> float:
    def pairs(result: RouteResult) -> set[tuple[str, str]]:
        grouped: set[tuple[str, str]] = set()
        for bundle in result.bundles:
            keys = sorted(unit.key for unit in bundle.units)
            grouped.update((keys[a], keys[b]) for a in range(len(keys)) for b in range(a + 1, len(keys)))
        return grouped

    left_pairs, right_pairs = pairs(left), pairs(right)
    union = left_pairs | right_pairs
    return len(left_pairs & right_pairs) / len(union) if union else 1.0
