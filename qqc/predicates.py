"""SQL predicates to matching sets to amplitude-amplification oracles."""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate


@dataclass(frozen=True)
class Predicate:
    width: int

    def matches(self, v: int) -> bool:
        raise NotImplementedError

    def matching_set(self) -> list[int]:
        return [v for v in range(2 ** self.width) if self.matches(v)]


@dataclass(frozen=True)
class Point(Predicate):
    value: int = 0

    def matches(self, v: int) -> bool:
        return v == self.value


@dataclass(frozen=True)
class Range(Predicate):
    lo: int = 0
    hi: int = 1

    def matches(self, v: int) -> bool:
        return self.lo <= v < self.hi


@dataclass(frozen=True)
class Threshold(Predicate):
    op: str = "<"
    c: int = 0

    def matches(self, v: int) -> bool:
        return {"<": v < self.c, "<=": v <= self.c,
                ">": v > self.c, ">=": v >= self.c}[self.op]


@dataclass(frozen=True)
class InSet(Predicate):
    values: tuple[int, ...] = ()

    def matches(self, v: int) -> bool:
        return v in self.values


@dataclass(frozen=True)
class CrossColumnEqual:
    """col_A = col_B over two equal-width columns; not a Cartesian product."""
    width: int

    def matching_pairs(self) -> list[tuple[int, int]]:
        return [(a, a) for a in range(2 ** self.width)]

    def joint_matching_set(self) -> list[int]:
        w = self.width
        return [(a << w) | a for a in range(2 ** w)]


def marking_oracle(n: int, marks: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the basis states in marks."""
    qc = QuantumCircuit(n)
    for m in marks:
        bits = [(m >> b) & 1 for b in range(n)]
        for b, v in enumerate(bits):
            if v == 0:
                qc.x(b)
        if n == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))
        for b, v in enumerate(bits):
            if v == 0:
                qc.x(b)
    return qc


def optimal_iters(n: int, n_marks: int, cap: int = 8) -> int:
    n_marks = max(1, n_marks)
    return max(1, min(cap, int(round((math.pi / 4) * math.sqrt(2 ** n / n_marks)))))


def build_search_for_set(n: int, marks: list[int],
                         n_iter: int | None = None) -> tuple[QuantumCircuit, list[int]]:
    """Amplitude-amplification circuit retrieving a member of marks."""
    if n_iter is None:
        n_iter = optimal_iters(n, len(marks))
    oracle = marking_oracle(n, marks)
    grover = grover_operator(oracle)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(n_iter):
        qc.compose(grover, inplace=True)
    qc.measure_all()
    return qc, marks


@dataclass
class ConjunctiveQuery:
    """Conjunction of per-column predicates over disjoint bit groups."""
    columns: list[Predicate]

    @property
    def widths(self) -> list[int]:
        return [c.width for c in self.columns]

    @property
    def n(self) -> int:
        return sum(self.widths)

    def per_column_sets(self) -> list[list[int]]:
        return [c.matching_set() for c in self.columns]

    def joint_matching_set(self) -> list[int]:
        joint = [0]
        shift = 0
        for w, S in zip(self.widths, self.per_column_sets()):
            joint = [base | (s << shift) for base in joint for s in S]
            shift += w
        return sorted(joint)

    def selectivity(self) -> float:
        return len(self.joint_matching_set()) / (2 ** self.n)
