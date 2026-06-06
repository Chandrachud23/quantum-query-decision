"""Whole vs factorized retrieval across a matrix of conjunctive queries, batched into one job."""

from __future__ import annotations

import sys
import os
import json
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qqc.predicates import build_search_for_set
from experiments._backend import resolve_backend, make_sampler
from experiments.run_tpch import p_in_set, run_batch

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

LAYOUT_SEEDS = [42, 7, 123]      # transpiler seeds -> distinct placements

# (name, per-group widths, matches per group)
MATRIX = [
    ("3+3",       [3, 3],          [1, 1]),
    ("4+4",       [4, 4],          [1, 1]),
    ("3+3 (kNN)", [3, 3],          [2, 1]),
    ("3+3+3",     [3, 3, 3],       [1, 1, 1]),
    ("4+5",       [4, 5],          [1, 1]),
    ("3+4+5",     [3, 4, 5],       [1, 1, 1]),
    ("3+3+3+3",   [3, 3, 3, 3],    [1, 1, 1, 1]),
]


def per_group_sets(widths, marks):
    return [sorted({(2 ** w - 1 - j) for j in range(m)})
            for w, m in zip(widths, marks)]


def cartesian(widths, sets):
    joint, shift = [0], 0
    for w, S in zip(widths, sets):
        joint = [b | (s << shift) for b in joint for s in S]
        shift += w
    return sorted(joint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--max-n", type=int, default=99)
    ap.add_argument("--hardware", type=str, default=None)
    args = ap.parse_args()

    if args.hardware:
        backend, is_hw = resolve_backend(hardware=args.hardware)
        sampler, run_dev = make_sampler(backend, is_hw), backend
    else:
        from experiments.device import load_fakefez
        backend = load_fakefez()
        run_dev = AerSimulator(noise_model=NoiseModel.from_backend(backend))
        is_hw, sampler = False, None
    where = getattr(backend, "name", "FakeFez")

    circuits, plan = [], []
    for name, widths, marks in MATRIX:
        n = sum(widths)
        if n > args.max_n:
            continue
        sets = per_group_sets(widths, marks)
        M = cartesian(widths, sets)
        for seed in LAYOUT_SEEDS:
            wc, _ = build_search_for_set(n, M)
            circuits.append(transpile(wc, backend=backend, optimization_level=3,
                                      seed_transpiler=seed))
            whole_idx = len(circuits) - 1
            frag_idx = []
            for w, S in zip(widths, sets):
                fc, _ = build_search_for_set(w, S)
                circuits.append(transpile(fc, backend=backend, optimization_level=3,
                                          seed_transpiler=seed))
                frag_idx.append((len(circuits) - 1, S))
            plan.append(dict(name=name, n=n, widths=widths, marks=marks,
                             whole_idx=whole_idx, frag_idx=frag_idx, M=M))

    print(f"[{where}] {len(circuits)} circuits across {len({p['name'] for p in plan})} "
          f"queries x {len(LAYOUT_SEEDS)} layouts"
          f"{' -> one batched job' if is_hw else ''}")
    counts = run_batch(circuits, run_dev, sampler, is_hw, args.shots)

    rows = []
    byname = {}
    for p in plan:
        byname.setdefault(p["name"], []).append(p)
    for name, plans in byname.items():
        pw, pf = [], []
        for p in plans:
            pw.append(p_in_set(counts[p["whole_idx"]], p["M"]))
            prod = 1.0
            for idx, S in p["frag_idx"]:
                prod *= p_in_set(counts[idx], S)
            pf.append(prod)
        mw, sw = statistics.mean(pw), (statistics.stdev(pw) if len(pw) > 1 else 0.0)
        mf, sf = statistics.mean(pf), (statistics.stdev(pf) if len(pf) > 1 else 0.0)
        gain = round(mf / mw, 1) if mw > 0 else None
        rows.append(dict(name=name, n=plans[0]["n"], widths=plans[0]["widths"],
                         marks=plans[0]["marks"], layouts=len(plans),
                         P_whole=round(mw, 4), P_whole_std=round(sw, 4),
                         P_factorized=round(mf, 4), P_factorized_std=round(sf, 4),
                         gain=gain))
    rows.sort(key=lambda r: (r["n"], r["name"]))

    with open(os.path.join(OUT, "factorize_matrix.json"), "w") as f:
        json.dump(dict(backend=where, is_hw=is_hw, shots=args.shots,
                       layouts=LAYOUT_SEEDS, rows=rows), f, indent=2)
    md = [f"# Factorization matrix ({where}, {args.shots} shots, "
          f"{len(LAYOUT_SEEDS)} layouts each)", "",
          "| query | $n$ | groups | marks | P(whole) | P(factorized) | gain |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        g = "" if r["gain"] is None else f"{r['gain']}$\\times$"
        md.append(f"| {r['name']} | {r['n']} | {'+'.join(map(str,r['widths']))} | "
                  f"{'+'.join(map(str,r['marks']))} | {r['P_whole']:.4f}$\\pm${r['P_whole_std']:.3f} | "
                  f"**{r['P_factorized']:.4f}**$\\pm${r['P_factorized_std']:.3f} | {g} |")
    with open(os.path.join(OUT, "factorize_matrix.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    for r in rows:
        print(f"  {r['name']:<11} n={r['n']:<2} whole={r['P_whole']:.3f}"
              f"±{r['P_whole_std']:.3f}  fac={r['P_factorized']:.3f}±{r['P_factorized_std']:.3f}"
              f"  gain={r['gain']}x")
    print("wrote factorize_matrix.md / factorize_matrix.json")


if __name__ == "__main__":
    main()
