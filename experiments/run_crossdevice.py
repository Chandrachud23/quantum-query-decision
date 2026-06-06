"""Whole vs factorized retrieval of one conjunctive query across several real devices."""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import transpile

from qqc.predicates import build_search_for_set
from experiments._backend import resolve_backend, make_sampler
from experiments.run_tpch import p_in_set, run_batch
from experiments.device import device_noise_profile

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

WIDTHS = [3, 3]
MARKS = [[5], [2]]


def run_device(name, shots):
    backend, is_hw = resolve_backend(hardware=name)
    sampler = make_sampler(backend, is_hw)
    n = sum(WIDTHS)
    joint, shift = [0], 0
    for w, S in zip(WIDTHS, MARKS):
        joint = [b | (s << shift) for b in joint for s in S]
        shift += w
    M = sorted(joint)

    circuits = [transpile(build_search_for_set(n, M)[0], backend=backend,
                          optimization_level=3, seed_transpiler=42)]
    for w, S in zip(WIDTHS, MARKS):
        circuits.append(transpile(build_search_for_set(w, S)[0], backend=backend,
                                  optimization_level=3, seed_transpiler=42))
    counts = run_batch(circuits, backend, sampler, is_hw, shots)

    p_whole = p_in_set(counts[0], M)
    p_fac = p_in_set(counts[1], MARKS[0]) * p_in_set(counts[2], MARKS[1])
    p2q = device_noise_profile(backend).p_2q
    return dict(device=backend.name, p_2q=round(p2q, 5),
                P_whole=round(p_whole, 4), P_factorized=round(p_fac, 4),
                gain=round(p_fac / p_whole, 1) if p_whole > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+",
                    default=["ibm_fez", "ibm_marrakesh", "ibm_kingston"])
    ap.add_argument("--shots", type=int, default=4096)
    args = ap.parse_args()

    rows = []
    for dev in args.devices:
        try:
            r = run_device(dev, args.shots)
            rows.append(r)
            print(f"  {r['device']:16s} p2q={r['p_2q']}  whole={r['P_whole']:.4f}  "
                  f"factorized={r['P_factorized']:.4f}  gain={r['gain']}x")
        except Exception as e:
            print(f"  {dev}: FAILED ({e})")

    with open(os.path.join(OUT, "crossdevice.json"), "w") as f:
        json.dump(dict(widths=WIDTHS, shots=args.shots, rows=rows), f, indent=2)
    md = [f"# Factorization across three physical IBM Heron devices "
          f"(six-qubit conjunctive query, {args.shots} shots)", "",
          "One batched job per device; one-shot retrieval success of the correct record.", "",
          "| device | $p_{2q}$ | P(whole) | P(factorized) | gain |",
          "|---|---|---|---|---|"]
    for r in rows:
        g = "" if r["gain"] is None else f"{r['gain']}$\\times$"
        md.append(f"| {r['device']} | {r['p_2q']} | {r['P_whole']:.4f} | "
                  f"**{r['P_factorized']:.4f}** | {g} |")
    md.append("")
    with open(os.path.join(OUT, "crossdevice.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote crossdevice.md / crossdevice.json")


if __name__ == "__main__":
    main()
