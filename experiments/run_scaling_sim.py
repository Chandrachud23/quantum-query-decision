"""Factorization scaling by noisy simulation: monolithic vs two-group vs width-3 groups."""

from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from experiments.device import load_fakefez
from experiments.run_factorize import run_one, opt_iters

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
SHOTS = 8192


def group_success(sampler, backend, width, seed):
    """Measured one-shot retrieval success of a single width-w sub-search, best layout."""
    p, _ = run_one(backend, sampler, False, width, opt_iters(width), 0, "best", SHOTS, seed)
    return p


def main():
    backend = load_fakefez()
    sampler = AerSimulator(noise_model=NoiseModel.from_backend(backend))

    # Cache per-width sub-search success so each width is simulated once.
    cache = {}

    def F(width):
        if width not in cache:
            cache[width] = group_success(sampler, backend, width, 1 + width)
        return cache[width]

    widths = [4, 6, 8, 9, 10, 12]
    rows = []
    for n in widths:
        mono = F(n) if n <= 8 else None        # n>=10 monolithic is dead (routing pen > 9); skip cost
        two = F(-(-n // 2)) * F(n // 2)        # ceil, floor
        w3 = (F(3) ** (n // 3)) if n % 3 == 0 else None
        rows.append({"n": n,
                     "monolithic": None if mono is None else round(mono, 4),
                     "factor_2groups": round(two, 4),
                     "factor_width3": None if w3 is None else round(w3, 4)})
        print(f"  n={n}: monolithic={'~0 (skipped)' if mono is None else round(mono,4)} "
              f"2groups={round(two,4)} width3={'-' if w3 is None else round(w3,4)}")

    md = ["# Factorization scaling by full noisy simulation (FakeFez)", "",
          f"One-shot retrieval success, {SHOTS} shots, best device layout. Monolithic for "
          "$n\\ge10$ is skipped (routing penalty $>9$, success $<10^{-3}$).", "",
          "| n | monolithic | factorize, 2 groups | factorize, width-3 groups |",
          "|---|---|---|---|"]
    for r in rows:
        m = "$<10^{-3}$" if r["monolithic"] is None else f"{r['monolithic']}"
        w3 = "--" if r["factor_width3"] is None else f"{r['factor_width3']}"
        md.append(f"| {r['n']} | {m} | {r['factor_2groups']} | {w3} |")
    md += ["", f"Per-width sub-search success (measured once): "
           + ", ".join(f"F({w})={round(cache[w],3)}" for w in sorted(cache)) + "."]
    with open(os.path.join(OUT, "scaling_sim.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "scaling_sim.json"), "w") as f:
        json.dump({"shots": SHOTS, "subsearch_success": {k: round(v, 4) for k, v in cache.items()},
                   "rows": rows}, f, indent=2)
    print("wrote scaling_sim.md / scaling_sim.json")


if __name__ == "__main__":
    main()
