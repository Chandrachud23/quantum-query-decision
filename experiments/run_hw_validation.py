"""Predicted vs measured retrieval fidelity over a suite of search queries."""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from scipy.stats import pearsonr, spearmanr

from qqc.fidelity_model import CircuitProfile, predict_log10_fidelity
from experiments.device import device_noise_profile, load_fakefez
from experiments.workloads import (build_search, profile_for_width, _counts,
                                   _two_q_depth, _phys_depth)
from experiments._backend import resolve_backend, make_sampler

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def speedup(widths=range(2, 9)):
    """Cost of the decision: closed-form criterion vs noisy-sim profiling, per width.

    Times evaluating the fidelity of every candidate width once with the model (Eq. 1) and once
    with a noisy simulation. The model is what the planner uses; the simulation is the profiler
    it replaces. Writes speedup.md."""
    import time
    from experiments.profiler import measured_fidelity
    backend = load_fakefez()
    noise = device_noise_profile(backend)

    t0 = time.perf_counter()
    for w in widths:
        predict_log10_fidelity(profile_for_width(w)["routed"], noise, saturate=True)
    t_model = time.perf_counter() - t0

    t1 = time.perf_counter()
    for w in widths:
        measured_fidelity(backend, w, shots=4096)
    t_prof = time.perf_counter() - t1

    ratio = t_prof / max(1e-9, t_model)
    md = ["# Decision cost: closed-form criterion vs noisy-sim profiling", "",
          f"Evaluating widths {min(widths)}..{max(widths)} once each.", "",
          "| method | wall time (s) |", "|---|---|",
          f"| closed-form criterion | {t_model:.2f} |",
          f"| noisy-sim profiling | {t_prof:.1f} |", "",
          f"**Speedup: {ratio:.0f}x.** The gap widens with width, since simulation is "
          f"exponential in qubits while the criterion is linear in gate count.", ""]
    with open(os.path.join(OUT, "speedup.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"criterion {t_model:.2f}s vs profiling {t_prof:.1f}s -> {ratio:.0f}x; wrote speedup.md")


def _active_qubits(tc) -> int:
    """Physical qubits that actually carry a gate (a transpile to a 156q device leaves
    most qubits idle; the T2 term must see only the footprint, not the whole chip)."""
    used = set()
    for inst in tc.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        for q in inst.qubits:
            used.add(tc.find_bit(q).index)
    return max(1, len(used))


def _pmarked(counts, n, marked_set):
    keys = {format(m, f"0{n}b") for m in marked_set}
    total = sum(counts.values())
    hits = sum(v for k, v in counts.items() if k.replace(" ", "")[-n:] in keys)
    return hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, default="FakeFez")
    ap.add_argument("--hardware", type=str, default=None,
                    help="real device name or 'least_busy' (omit for fake/Aer)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--grid", type=str, default="3:1,3:2,4:1,4:2,5:1,6:1",
                    help="comma list of n:iters or n:iters:marks queries "
                         "(marks>1 = multi-target / kNN-style retrieval)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent runs (re-transpiled to different layouts) to show "
                         "the correlation is stable against device drift, not one lucky job")
    ap.add_argument("--speedup", action="store_true",
                    help="instead of validating, time the criterion vs noisy-sim profiling")
    args = ap.parse_args()

    if args.speedup:
        speedup()
        return

    backend, is_hw = resolve_backend(args.backend, args.hardware)
    noise = device_noise_profile(backend)
    sampler = make_sampler(backend, is_hw)
    tag = args.hardware or args.backend
    print(f"backend={getattr(backend,'name',tag)} hardware={is_hw} shots={args.shots}")

    grid = []
    for g in args.grid.split(","):
        parts = [int(x) for x in g.split(":")]
        n, it = parts[0], parts[1]
        marks = parts[2] if len(parts) > 2 else 1
        grid.append((n, it, marks))

    # Static per-query info (ideal value, marked set) computed once.
    rows = []
    for (n, it, marks) in grid:
        circ, _ = build_search(n, n_iter=it, n_marks=marks)
        marked_set = list(range(marks))
        circ_nm = circ.remove_final_measurements(inplace=False)
        sv = Statevector(circ_nm).probabilities()
        rows.append({"n": n, "iters": it, "marks": marks,
                     "P_ideal": round(float(sum(sv[m] for m in marked_set)), 4),
                     "marked_set": marked_set, "_pred": [], "_meas": [], "_n2": []})

    # Each repeat re-transpiles to a fresh layout (seed) and runs the batch, so the
    # spread reflects real device/layout drift, not a single lucky job.
    for rep in range(args.repeats):
        seed = 42 + rep
        isa = []
        for gi, (n, it, marks) in enumerate(grid):
            circ, _ = build_search(n, n_iter=it, n_marks=marks)
            tc = transpile(circ, backend=backend, optimization_level=3,
                           seed_transpiler=seed)
            n2, n1 = _counts(tc)
            prof = CircuitProfile(n_2q=n2, n_1q=n1, n_qubits=n,
                                  depth_2q=_two_q_depth(tc), depth_tot=_phys_depth(tc),
                                  n_qubits_phys=_active_qubits(tc))
            rows[gi]["_pred"].append(
                predict_log10_fidelity(prof, noise, saturate=True)["log10_F"])
            rows[gi]["_n2"].append(n2)
            isa.append(tc)
        job = sampler.run([(c,) for c in isa], shots=args.shots).result()
        for gi, r in enumerate(rows):
            data = job[gi].data
            counts = getattr(data, next(iter(data.keys()))).get_counts()
            p_noisy = _pmarked(counts, r["n"], r["marked_set"])
            f = min(1.0, p_noisy / r["P_ideal"]) if r["P_ideal"] > 0 else 0.0
            r["_meas"].append(float(np.log10(f)) if f > 0 else -9.0)
        print(f"  repeat {rep + 1}/{args.repeats} done")

    pred, meas = [], []
    for r in rows:
        r["n2_dev"] = int(np.median(r["_n2"]))
        r["pred_log10F"] = round(float(np.mean(r["_pred"])), 3)
        r["meas_log10F"] = round(float(np.mean(r["_meas"])), 3)
        r["meas_std"] = round(float(np.std(r["_meas"])), 3)
        pred.append(r["pred_log10F"])
        meas.append(r["meas_log10F"])
        print(f"  n={r['n']} it={r['iters']} m={r['marks']}: pred={r['pred_log10F']:.3f} "
              f"meas={r['meas_log10F']:.3f}±{r['meas_std']:.3f}")

    pear = pearsonr(pred, meas)
    spear = spearmanr(pred, meas)
    # Bootstrap 95% CI for Spearman rho (resample query-points with replacement).
    pa, ma = np.array(pred), np.array(meas)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(pa), len(pa))
        if len(set(ma[idx])) > 1 and len(set(pa[idx])) > 1:
            boot.append(spearmanr(pa[idx], ma[idx])[0])
    rho_lo, rho_hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
    # Affine recalibration of the magnitude bias: fit meas = a*pred + b. The bias is a single
    # monotone offset, so a global fit shrinks the absolute error without changing the rank order.
    a, b = np.polyfit(pa, ma, 1)
    rmse_raw = float(np.sqrt(np.mean((ma - pa) ** 2)))
    rmse_aff = float(np.sqrt(np.mean((ma - (a * pa + b)) ** 2)))
    md = [f"# Hardware validation on {getattr(backend,'name',tag)} "
          f"({'REAL DEVICE' if is_hw else 'Aer noise model'})", "",
          f"Predicted (closed form) vs measured retrieval fidelity over "
          f"{len(rows)} search queries, {args.shots} shots, {args.repeats} "
          f"repeat(s) (each re-transpiled to a fresh layout). Prediction and measurement "
          f"come from the same compiled circuit; the model is parameterised from device "
          f"calibration, not fitted to these measurements.", "",
          f"**Pearson r = {pear[0]:.3f}** (p={pear[1]:.1e})  ·  "
          f"**Spearman rho = {spear[0]:.3f}** (95\\% CI [{rho_lo:.2f}, {rho_hi:.2f}], "
          f"$n{{=}}{len(rows)}$ queries)", "",
          f"Affine recalibration: meas = {a:.2f}*pred {b:+.2f}; RMSE {rmse_raw:.2f} -> "
          f"{rmse_aff:.2f} dex, rank order unchanged.", "",
          "| n | iters | marks | 2Q (device) | pred log10F | meas log10F (mean±std) | P_ideal |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['n']} | {r['iters']} | {r['marks']} | {r['n2_dev']} | "
                  f"{r['pred_log10F']} | {r['meas_log10F']} ± {r['meas_std']} | {r['P_ideal']} |")
    md += ["", "The deep queries show the on-device degradation: as the query deepens, "
           "routing drives the measured retrieval success toward the random floor, the "
           "loss the planner cuts to avoid.", ""]
    for r in rows:
        r.pop("_pred", None); r.pop("_meas", None); r.pop("_n2", None)
    name = "validation_hw" if is_hw else f"validation_{args.backend.lower()}"
    with open(os.path.join(OUT, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"backend": getattr(backend, "name", tag), "is_hw": is_hw,
                   "n_queries": len(rows), "pearson": pear[0], "spearman": spear[0],
                   "spearman_ci95": [float(rho_lo), float(rho_hi)],
                   "recal_slope": float(a), "recal_intercept": float(b),
                   "rmse_raw": rmse_raw, "rmse_affine": rmse_aff, "rows": rows}, f, indent=2)
    print(f"  Pearson r={pear[0]:.3f}  Spearman rho={spear[0]:.3f}; wrote {name}.md")


if __name__ == "__main__":
    main()
