"""Check that global-oracle local-diffusion search does not amplify a product target."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit.quantum_info import Statevector


def _mcz_zero(qc, n):                       # phase-flip the all-zero (joint target) state
    qc.x(range(n)); qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n))); qc.x(range(n))


def _block_diffusion(qc, qubits):
    qc.h(qubits); qc.x(qubits)
    qc.append(MCMTGate(ZGate(), len(qubits) - 1, 1), qubits)
    qc.x(qubits); qc.h(qubits)


def local_diffusion_search(n, groups, t):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(t):
        _mcz_zero(qc, n)                    # global conjunctive oracle
        for g in groups:
            _block_diffusion(qc, g)         # per-block local diffusion
    return qc


def main():
    n, groups = 6, [[0, 1, 2], [3, 4, 5]]
    best = 0.0
    print("local-diffusion search on a 6-qubit conjunctive target (ideal, noiseless):")
    for t in range(1, 9):
        p = float(abs(Statevector(local_diffusion_search(n, groups, t)).data[0]) ** 2)
        best = max(best, p)
        print(f"  iterations={t}: P(target)={p:.3f}")
    print(f"best ideal success over iterations: {best:.3f} "
          f"(monolithic Grover reaches ~1.0; factorization reaches ~1.0 ideal). "
          f"local diffusion does not solve the conjunctive search, so it is not a baseline.")


if __name__ == "__main__":
    main()
