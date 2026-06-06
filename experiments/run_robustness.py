"""Routing penalty across device calibrations and across circuit classes (search/QAOA/GHZ)."""

from __future__ import annotations

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.stats import pearsonr, spearmanr

from qqc.fidelity_model import CircuitProfile, predict_log10_fidelity, routing_penalty
from qqc.cut_decision import GAMMA2_WIRE_CC
from experiments.device import load_fakefez, device_noise_profile, device_islands
from experiments.workloads import (build_search, profile_for_width, FAKEFEZ_BASIS,
                                   _counts, _two_q_depth, _phys_depth)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
DEVICES = ["FakeFez", "FakeMarrakesh", "FakeTorino"]


def _load(name):
    import importlib
    return getattr(importlib.import_module("qiskit_ibm_runtime.fake_provider"), name)()


def cross_device():
    """Routing penalty of a five-qubit search across calibrations (analytical, no simulation)."""
    profs = {w: profile_for_width(w) for w in (5,)}
    rows = []
    for name in DEVICES:
        try:
            backend = _load(name)
        except Exception as e:
            print(f"skip {name}: {e}"); continue
        noise = device_noise_profile(backend)
        region = max(i.size for i in device_islands(backend, z_thresh=1.0))
        pen = routing_penalty(profs[5]["synth"], profs[5]["routed"], noise)
        rows.append({"device": name, "p_2q": noise.p_2q, "region": region,
                     "routing_pen_n5": round(pen, 2), "plan": "factorize"})
        print(f"  {name}: p2q={noise.p_2q:.1e} region={region} pen(n5)={pen:.2f} -> factorize")
    md = ["# Cross-device robustness (five-qubit search, analytical)", "",
          "| device | $p_{2q}$ | low-noise region | routing pen. ($n{=}5$) | plan |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['device']} | {r['p_2q']:.1e} | {r['region']} | "
                  f"{r['routing_pen_n5']} | {r['plan']} |")
    with open(os.path.join(OUT, "robustness.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return rows


def _qaoa(n, gamma=0.8, beta=0.4):
    qc = QuantumCircuit(n); qc.h(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
    for i in range(n):
        qc.rx(2 * beta, i)
    qc.measure_all(); return qc


def _ghz(n):
    qc = QuantumCircuit(n); qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all(); return qc


def _xcorr(noisy, ideal):
    """F = sum p_n p_i / sum p_i^2, a noise-sensitive classical fidelity in [0,1]."""
    keys = set(noisy) | set(ideal)
    tn, ti = sum(noisy.values()) or 1, sum(ideal.values()) or 1
    num = sum((noisy.get(k, 0) / tn) * (ideal.get(k, 0) / ti) for k in keys)
    den = sum((ideal.get(k, 0) / ti) ** 2 for k in keys) or 1e-12
    return min(1.0, num / den)


def generality(shots=8192):
    """Predicted vs measured fidelity across Grover/QAOA/GHZ, with the routing-penalty verdict."""
    noise = device_noise_profile(load_fakefez())
    sim = AerSimulator(noise_model=NoiseModel.from_backend(load_fakefez()))
    ideal_sim = AerSimulator()
    thresh = math.log10(GAMMA2_WIRE_CC)
    build = {"search": lambda n: build_search(n)[0], "qaoa": _qaoa, "ghz": _ghz}
    rows = []
    for cls in ("search", "qaoa", "ghz"):
        for n in (3, 4, 5):
            circ = build[cls](n)
            synth = transpile(circ, basis_gates=FAKEFEZ_BASIS, optimization_level=1, seed_transpiler=42)
            routed = transpile(circ, basis_gates=FAKEFEZ_BASIS, coupling_map=CouplingMap.from_line(n),
                               optimization_level=3, seed_transpiler=42)
            def prof(c):
                n2, n1 = _counts(c)
                return CircuitProfile(n_2q=n2, n_1q=n1, n_qubits=n, depth_2q=_two_q_depth(c),
                                      depth_tot=_phys_depth(c), n_qubits_phys=n)
            pen = routing_penalty(prof(synth), prof(routed), noise)
            pred = predict_log10_fidelity(prof(routed), noise, saturate=True)["log10_F"]
            ideal = ideal_sim.run(transpile(circ, ideal_sim, optimization_level=0),
                                  shots=shots, seed_simulator=0).result().get_counts()
            noisy = sim.run(routed, shots=shots, seed_simulator=1).result().get_counts()
            meas = math.log10(_xcorr(noisy, ideal)) if _xcorr(noisy, ideal) > 0 else -9.0
            rows.append({"class": cls, "n": n, "routing_pen": round(pen, 2),
                         "pred_log10F": round(pred, 3), "meas_log10F": round(meas, 3),
                         "verdict": "routing-dominated" if pen > thresh else "routes cheaply"})
    r = pearsonr([x["pred_log10F"] for x in rows], [x["meas_log10F"] for x in rows])[0]
    rho = spearmanr([x["pred_log10F"] for x in rows], [x["meas_log10F"] for x in rows])[0]
    md = [f"# Generality across circuit classes (FakeFez): Pearson r={r:.3f}, Spearman rho={rho:.3f}",
          "", "| class | n | routing penalty | pred log10F | meas log10F | verdict |",
          "|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['class']} | {x['n']} | {x['routing_pen']} | {x['pred_log10F']} | "
                  f"{x['meas_log10F']} | {x['verdict']} |")
    with open(os.path.join(OUT, "generality.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"generality: r={r:.3f} rho={rho:.3f}")
    return {"pearson": r, "spearman": rho, "rows": rows}


def main():
    print("cross-device:"); cd = cross_device()
    print("generality:");   g = generality()
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump({"cross_device": cd, "generality": g}, f, indent=2)
    print("wrote robustness.md / generality.md / robustness.json")


if __name__ == "__main__":
    main()
