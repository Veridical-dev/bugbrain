from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

from .model import NeuronAnnotation


MAGIC = b"VFLYCSR1"
HEADER = struct.Struct("<QQI")  # nodes, edges, metadata bytes
EDGE = struct.Struct("<IIf")


def _open_text(path: str | Path, *, newline: str = "") -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline=newline)
    return path.open("r", encoding="utf-8", newline=newline)


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotations(path: str | Path) -> list[NeuronAnnotation]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        annotations: list[NeuronAnnotation] = []
        for row in reader:
            raw_id = row.get("root_id", "").strip()
            if not raw_id:
                continue
            confidence = row.get("top_nt_conf", "").strip()
            annotations.append(
                NeuronAnnotation(
                    root_id=int(raw_id),
                    flow=row.get("flow", "").strip().lower(),
                    super_class=row.get("super_class", "").strip().lower(),
                    cell_class=row.get("cell_class", "").strip(),
                    cell_type=row.get("cell_type", "").strip(),
                    neurotransmitter=(row.get("known_nt", "").strip() or row.get("top_nt", "").strip()).lower(),
                    neurotransmitter_confidence=float(confidence) if confidence else 0.0,
                )
            )
    if not annotations:
        raise ValueError(f"No neuron annotations found in {path}")
    return annotations


def _native_copy(values: array) -> array:
    copied = array(values.typecode, values)
    if sys.byteorder != "little":
        copied.byteswap()
    return copied


def _write_array(handle: BinaryIO, values: array) -> None:
    _native_copy(values).tofile(handle)


def _read_array(handle: BinaryIO, typecode: str, count: int) -> array:
    values = array(typecode)
    values.fromfile(handle, count)
    if sys.byteorder != "little":
        values.byteswap()
    return values


@dataclass(slots=True)
class Connectome:
    neuron_ids: array
    offsets: array
    targets: array
    weights: array
    out_strength: array
    signs: array
    metadata: dict[str, object]
    _id_to_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_index = {root_id: index for index, root_id in enumerate(self.neuron_ids)}

    @property
    def node_count(self) -> int:
        return len(self.neuron_ids)

    @property
    def edge_count(self) -> int:
        return len(self.targets)

    def index_of(self, root_id: int) -> int | None:
        return self._id_to_index.get(root_id)

    def shuffled_targets(self, seed: int) -> Connectome:
        """Return a target-shuffled null preserving every source out-degree and global in-degree."""
        targets = array("I", self.targets)
        random.Random(seed).shuffle(targets)
        return Connectome(
            neuron_ids=self.neuron_ids,
            offsets=self.offsets,
            targets=targets,
            weights=self.weights,
            out_strength=self.out_strength,
            signs=self.signs,
            metadata={**self.metadata, "arm": "target-shuffled", "shuffle_seed": seed},
        )

    def propagate(
        self,
        seeds: dict[int, float],
        *,
        steps: int = 5,
        active_width: int = 2_048,
        restart: float = 0.15,
        signed: bool = False,
    ) -> dict[int, float]:
        if not seeds:
            return {}
        if not 0.0 <= restart <= 1.0:
            raise ValueError("restart must be in [0, 1]")
        seed_norm = sum(abs(value) for value in seeds.values()) or 1.0
        seed_state = {index: value / seed_norm for index, value in seeds.items()}
        state = dict(seed_state)
        for _ in range(steps):
            next_state: dict[int, float] = {}
            for source, source_value in state.items():
                strength = float(self.out_strength[source])
                if strength <= 0:
                    continue
                sign = int(self.signs[source]) if signed else 1
                start = int(self.offsets[source])
                stop = int(self.offsets[source + 1])
                scale = (1.0 - restart) * source_value * sign / strength
                for edge_index in range(start, stop):
                    target = int(self.targets[edge_index])
                    contribution = scale * float(self.weights[edge_index])
                    next_state[target] = next_state.get(target, 0.0) + contribution
            for index, value in seed_state.items():
                next_state[index] = next_state.get(index, 0.0) + restart * value
            if len(next_state) > active_width:
                winners = sorted(next_state, key=lambda idx: (-abs(next_state[idx]), idx))[:active_width]
                next_state = {index: next_state[index] for index in winners}
            norm = sum(abs(value) for value in next_state.values()) or 1.0
            state = {index: value / norm for index, value in next_state.items()}
        return state


def _sign_for(annotation: NeuronAnnotation) -> int:
    # FlyWire known_nt may include comma-separated co-transmitter annotations.
    primary = annotation.neurotransmitter.split(",", 1)[0].strip()
    if primary in {"gaba", "glutamate", "glut", "serotonin", "octopamine"}:
        return -1
    return 1


def build_cache(
    connections_path: str | Path,
    annotations_path: str | Path,
    cache_path: str | Path,
    *,
    min_synapses: int = 5,
    progress_every: int = 1_000_000,
) -> dict[str, object]:
    """Build a compact CSR cache in one CSV pass plus one compact-binary pass."""
    started = time.monotonic()
    connections_path = Path(connections_path)
    annotations_path = Path(annotations_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_path)
    neuron_ids = array("Q", (annotation.root_id for annotation in annotations))
    id_to_index = {root_id: index for index, root_id in enumerate(neuron_ids)}
    degrees = array("I", [0]) * len(neuron_ids)
    out_strength = array("f", [0.0]) * len(neuron_ids)
    signs = array("b", (_sign_for(annotation) for annotation in annotations))

    total_rows = 0
    kept_edges = 0
    kept_synapses = 0
    missing_nodes = 0
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix="flywire-edges-", suffix=".bin", dir=cache_path.parent, delete=False
    )
    temp_path = Path(temp_handle.name)
    buffer = bytearray()
    try:
        with temp_handle, _open_text(connections_path) as source:
            reader = csv.DictReader(source)
            required = {"pre_root_id", "post_root_id", "syn_count"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Connection table must contain {sorted(required)}")
            for row in reader:
                total_rows += 1
                weight = int(float(row["syn_count"]))
                if weight < min_synapses:
                    continue
                source_index = id_to_index.get(int(row["pre_root_id"]))
                target_index = id_to_index.get(int(row["post_root_id"]))
                if source_index is None or target_index is None:
                    missing_nodes += 1
                    continue
                buffer += EDGE.pack(source_index, target_index, float(weight))
                degrees[source_index] += 1
                out_strength[source_index] += weight
                kept_edges += 1
                kept_synapses += weight
                if len(buffer) >= 1024 * 1024:
                    temp_handle.write(buffer)
                    buffer.clear()
                if progress_every and total_rows % progress_every == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"scanned={total_rows:,} kept={kept_edges:,} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
            if buffer:
                temp_handle.write(buffer)

        offsets = array("Q", [0]) * (len(neuron_ids) + 1)
        running = 0
        for index, degree in enumerate(degrees):
            offsets[index] = running
            running += int(degree)
        offsets[len(neuron_ids)] = running
        if running != kept_edges:
            raise AssertionError("CSR degree count does not match retained edge count")

        targets = array("I", [0]) * kept_edges
        weights = array("f", [0.0]) * kept_edges
        cursor = array("Q", offsets[:-1])
        with temp_path.open("rb") as compact:
            while chunk := compact.read(EDGE.size * 100_000):
                if len(chunk) % EDGE.size:
                    raise ValueError("Temporary edge file is truncated")
                for source_index, target_index, weight in EDGE.iter_unpack(chunk):
                    position = int(cursor[source_index])
                    targets[position] = target_index
                    weights[position] = weight
                    cursor[source_index] += 1

        metadata: dict[str, object] = {
            "format": "VFLYCSR1",
            "dataset": "FlyWire FAFB v783",
            "min_synapses": min_synapses,
            "nodes": len(neuron_ids),
            "edges": kept_edges,
            "represented_synapses": kept_synapses,
            "source_rows": total_rows,
            "missing_annotation_edges": missing_nodes,
            "connections_md5": md5_file(connections_path),
            "annotations_md5": md5_file(annotations_path),
            "sign_policy": "effectome-v1-primary-transmitter",
            "built_seconds": round(time.monotonic() - started, 3),
        }
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        staged_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with staged_path.open("wb") as output:
            output.write(MAGIC)
            output.write(HEADER.pack(len(neuron_ids), kept_edges, len(metadata_bytes)))
            output.write(metadata_bytes)
            _write_array(output, neuron_ids)
            _write_array(output, offsets)
            _write_array(output, targets)
            _write_array(output, weights)
            _write_array(output, out_strength)
            _write_array(output, signs)
        os.replace(staged_path, cache_path)
        return metadata
    finally:
        temp_path.unlink(missing_ok=True)


def load_cache(path: str | Path) -> Connectome:
    cache_path = Path(path)
    with cache_path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Not a FlyWire CSR cache: {cache_path}")
        node_count, edge_count, metadata_length = HEADER.unpack(handle.read(HEADER.size))
        metadata = json.loads(handle.read(metadata_length))
        neuron_ids = _read_array(handle, "Q", node_count)
        offsets = _read_array(handle, "Q", node_count + 1)
        targets = _read_array(handle, "I", edge_count)
        weights = _read_array(handle, "f", edge_count)
        out_strength = _read_array(handle, "f", node_count)
        signs = _read_array(handle, "b", node_count)
        if handle.read(1):
            raise ValueError(f"Unexpected trailing bytes in cache: {cache_path}")
    if offsets[-1] != edge_count:
        raise ValueError("CSR cache has inconsistent offsets")
    return Connectome(neuron_ids, offsets, targets, weights, out_strength, signs, metadata)
