"""Search-class workloads and their synthesis vs routed compiled profiles."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit.transpiler import CouplingMap

from qqc.fidelity_model import CircuitProfile

FAKEFEZ_BASIS = ["cz", "id", "rz", "sx", "x"]

_TWO_Q = {"cz", "cx", "ecr", "cy", "swap", "iswap"}
_SWAP_W = {"swap": 3}


def _marked_oracle(n: int, n_marks: int = 1) -> QuantumCircuit:
    """Phase oracle marking n_marks computational-basis states (multi-controlled Z)."""
    qc = QuantumCircuit(n)
    for t in range(n_marks):
        bits = [(t >> b) & 1 for b in range(n)]
        for b, v in enumerate(bits):
            if v == 0:
                qc.x(b)
        qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))
        for b, v in enumerate(bits):
            if v == 0:
                qc.x(b)
    return qc


def build_search(n: int, n_iter: int | None = None, n_marks: int = 1) -> tuple[QuantumCircuit, int]:
    """Grover/amplitude-amplification search circuit over n qubits.
    Returns (circuit_with_measure, marked_index)."""
    import math
    if n_iter is None:
        n_iter = max(1, int(round((math.pi / 4) * math.sqrt(2 ** n / n_marks))))
        n_iter = min(n_iter, 6)               # cap for sim tractability
    oracle = _marked_oracle(n, n_marks)
    grover = grover_operator(oracle)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(n_iter):
        qc.compose(grover, inplace=True)
    qc.measure_all()
    return qc, 0      # marked index 0 (all-zero key after the X-conjugation above)


def _counts(circ: QuantumCircuit) -> tuple[int, int]:
    """(n_2q CX-equiv, n_1q physical) from a transpiled circuit's op counts."""
    ops = circ.count_ops()
    n2 = 0
    for name, c in ops.items():
        nm = name.lower()
        if nm in _TWO_Q:
            n2 += c * _SWAP_W.get(nm, 1)
    skip = {"barrier", "measure", "rz", "id"}        # rz virtual on IBM
    n1 = sum(c for k, c in ops.items()
             if k.lower() not in skip and k.lower() not in _TWO_Q)
    return n2, n1


def _two_q_depth(circ: QuantumCircuit) -> int:
    return circ.depth(lambda inst: inst.operation.name.lower() in _TWO_Q)


def _phys_depth(circ: QuantumCircuit) -> int:
    zero = {"rz", "id", "barrier", "measure", "delay"}
    return circ.depth(lambda inst: inst.operation.name.lower() not in zero)


def profile_for_width(n: int, seed: int = 42,
                      n_iter: int | None = None) -> dict:
    """Transpile a width-n search circuit two ways and return CircuitProfiles.
    Returns dict with 'circuit', 'marked', 'synth', 'routed'."""
    circ, marked = build_search(n, n_iter=n_iter)

    # Synthesis: basis only, no coupling (all-to-all) -> pre-routing counts.
    synth_c = transpile(circ, basis_gates=FAKEFEZ_BASIS,
                        optimization_level=1, seed_transpiler=seed)
    # Routed: sparse line heavy-hex slice -> SWAP cascades.
    line = CouplingMap.from_line(n)
    routed_c = transpile(circ, basis_gates=FAKEFEZ_BASIS, coupling_map=line,
                         optimization_level=3, seed_transpiler=seed)

    n2s, n1s = _counts(synth_c)
    n2r, n1r = _counts(routed_c)
    synth = CircuitProfile(n_2q=n2s, n_1q=n1s, n_qubits=n,
                           depth_2q=_two_q_depth(synth_c),
                           depth_tot=_phys_depth(synth_c), n_qubits_phys=n)
    routed = CircuitProfile(n_2q=n2r, n_1q=n1r, n_qubits=n,
                            depth_2q=_two_q_depth(routed_c),
                            depth_tot=_phys_depth(routed_c),
                            n_qubits_phys=routed_c.num_qubits)
    return {"circuit": circ, "routed_circuit": routed_c, "marked": marked,
            "synth": synth, "routed": routed,
            "n2_synth": n2s, "n2_routed": n2r}
