from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class NeuronAnnotation:
    root_id: int
    flow: str = ""
    super_class: str = ""
    cell_class: str = ""
    cell_type: str = ""
    neurotransmitter: str = ""
    neurotransmitter_confidence: float = 0.0

    def label(self) -> str:
        parts = [self.cell_type, self.cell_class, self.super_class, self.flow]
        return " / ".join(part for part in parts if part) or "unannotated"


@dataclass(frozen=True, slots=True)
class CodeUnit:
    key: str
    path: str
    new_line: int
    heading: str
    excerpt: str
    tokens: tuple[str, ...]
    risk_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NeuronReceipt:
    root_id: int
    activation: float
    label: str
    neurotransmitter: str


@dataclass(frozen=True, slots=True)
class Bundle:
    reviewer_id: str
    lens: str
    units: tuple[CodeUnit, ...]
    anchor: NeuronReceipt | None
    circuit_receipt: tuple[NeuronReceipt, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class RouteResult:
    arm: str
    graph: dict[str, Any]
    parameters: dict[str, Any]
    bundles: tuple[Bundle, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
