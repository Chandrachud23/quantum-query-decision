"""Noisy-simulation retrieval-success profiler, the baseline the analytical model replaces."""

from __future__ import annotations

import time

from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .workloads import build_search, FAKEFEZ_BASIS

_IDEAL = AerSimulator()


def _marked_bitstring(n: int, marked: int) -> str:
    return format(marked, f"0{n}b")


def ideal_success(n: int, n_iter=None, shots: int = 8192, seed: int = 0) -> float:
    circ, marked = build_search(n, n_iter=n_iter)
    tc = transpile(circ, _IDEAL, optimization_level=0)
    res = _IDEAL.run(tc, shots=shots, seed_simulator=seed).result().get_counts()
    key = _marked_bitstring(n, marked)
    total = sum(res.values())
    # measure_all stores in a register; match on the (possibly space-joined) key
    hits = sum(v for k, v in res.items() if k.replace(" ", "")[::-1].endswith("") and
               k.replace(" ", "") == key) or \
           sum(v for k, v in res.items() if k.replace(" ", "")[-n:] == key)
    return hits / total


def noisy_success(backend, n: int, n_iter=None, shots: int = 8192,
                  seed: int = 1) -> tuple[float, float]:
    """Return (P_noisy(marked), wall_seconds) for a width-n routed search query."""
    circ, marked = build_search(n, n_iter=n_iter)
    line = CouplingMap.from_line(n)
    routed = transpile(circ, basis_gates=FAKEFEZ_BASIS, coupling_map=line,
                       optimization_level=3, seed_transpiler=42)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm)
    t0 = time.perf_counter()
    res = sim.run(routed, shots=shots, seed_simulator=seed).result().get_counts()
    dt = time.perf_counter() - t0
    key = _marked_bitstring(n, marked)
    total = sum(res.values())
    hits = sum(v for k, v in res.items() if k.replace(" ", "")[-n:] == key)
    return hits / total, dt


def measured_fidelity(backend, n: int, n_iter=None, shots: int = 8192,
                      seed: int = 1) -> dict:
    p_ideal = ideal_success(n, n_iter=n_iter, shots=shots)
    p_noisy, dt = noisy_success(backend, n, n_iter=n_iter, shots=shots, seed=seed)
    import math
    f = min(1.0, p_noisy / p_ideal) if p_ideal > 0 else 0.0
    return {"n": n, "n_iter": n_iter, "P_ideal": p_ideal, "P_noisy": p_noisy,
            "F_meas": f, "log10_F_meas": math.log10(f) if f > 0 else -9.0,
            "profile_seconds": dt}


def measured_fidelity_seeds(backend, n: int, n_iter=None, shots: int = 8192,
                            seeds=(1, 2, 3)) -> dict:
    """Mean/std measured log10F over independent noisy-sim seeds (shot+noise noise)."""
    import statistics, math
    vals, pn, pid, t = [], [], None, 0.0
    for s in seeds:
        r = measured_fidelity(backend, n, n_iter=n_iter, shots=shots, seed=s)
        vals.append(r["log10_F_meas"]); pn.append(r["P_noisy"]); pid = r["P_ideal"]
        t += r["profile_seconds"]
    return {"n": n, "n_iter": n_iter, "P_ideal": pid,
            "P_noisy_mean": statistics.mean(pn),
            "log10_F_mean": statistics.mean(vals),
            "log10_F_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "profile_seconds": t}
