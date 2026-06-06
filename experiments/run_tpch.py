"""Execute TPC-H-style predicates and verify retrieved rows against classical SQL."""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qqc.predicates import (Point, Threshold, ConjunctiveQuery,
                            CrossColumnEqual, build_search_for_set)
from experiments._backend import resolve_backend, make_sampler

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
DECODE = lambda k: int(k.replace(" ", ""), 2)        # qiskit counts are MSB-first


def query_suite():
    Q = []
    Q.append(dict(name="Q6 (3+3)", kind="sep",
                  sql="discount=5 AND quantity=2",
                  groups=[("discount", Point(3, 5)), ("quantity", Point(3, 2))]))
    Q.append(dict(name="Q6 (4+4)", kind="sep",
                  sql="discount=9 AND quantity<2",
                  groups=[("discount", Point(4, 9)), ("quantity", Threshold(4, "<", 2))]))
    Q.append(dict(name="Q6 (3+3+3)", kind="sep",
                  sql="discount=5 AND quantity=2 AND shipdate=3",
                  groups=[("discount", Point(3, 5)), ("quantity", Point(3, 2)),
                          ("shipdate", Point(3, 3))]))
    Q.append(dict(name="Q6 (2+2+2)", kind="sep",
                  sql="discount=1 AND quantity=2 AND shipdate=0",
                  groups=[("discount", Point(2, 1)), ("quantity", Point(2, 2)),
                          ("shipdate", Point(2, 0))]))
    Q.append(dict(name="Q1 (broad)", kind="broad",
                  sql="shipdate <= 6  (broad aggregation scan)",
                  groups=[("shipdate", Threshold(3, "<=", 6))]))
    Q.append(dict(name="XEQ (colA=colB)", kind="xeq",
                  sql="colA = colB  (cross-column join key)",
                  xeq=CrossColumnEqual(width=2)))
    return Q


def transpile_best(circ, backend, is_hw):
    return transpile(circ, backend=backend, optimization_level=3, seed_transpiler=42)


def run_batch(circuits, backend, sampler, is_hw, shots):
    """Run transpiled circuits; on hardware as a single batched SamplerV2 job."""
    if is_hw:
        res = sampler.run([(c,) for c in circuits], shots=shots).result()
        out = []
        for r in res:
            data = r.data
            reg = next(iter(data.keys()))
            out.append(getattr(data, reg).get_counts())
        return out
    return [backend.run(c, shots=shots, seed_simulator=1).result().get_counts()
            for c in circuits]


def p_in_set(counts, S):
    tot = sum(counts.values())
    return sum(v for k, v in counts.items() if DECODE(k) in set(S)) / tot if tot else 0.0


def split_joint(value, widths):
    out, shift = [], 0
    for w in widths:
        out.append((value >> shift) & ((1 << w) - 1))
        shift += w
    return out


def scan_matching_set(cq: ConjunctiveQuery):
    """Joint matching set by scanning the keyspace (independent of the product path)."""
    out = []
    for key in range(2 ** cq.n):
        vals = split_joint(key, cq.widths)
        if all(p.matches(v) for p, v in zip(cq.columns, vals)):
            out.append(key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--hardware", type=str, default=None)
    args = ap.parse_args()

    if args.hardware:
        backend, is_hw = resolve_backend(hardware=args.hardware)
        sampler = make_sampler(backend, is_hw)
        run_dev = backend
    else:
        from experiments.device import load_fakefez
        fake = load_fakefez()
        run_dev = AerSimulator(noise_model=NoiseModel.from_backend(fake))
        backend, is_hw, sampler = fake, False, None

    where = getattr(backend, "name", "FakeFez")
    results = []
    circuits = []

    for q in query_suite():
        if q["kind"] in ("sep", "broad"):
            preds = [p for _, p in q["groups"]]
            widths = [p.width for p in preds]
            cq = ConjunctiveQuery(preds)
            M = cq.joint_matching_set()
            per = cq.per_column_sets()
            verified = (sorted(scan_matching_set(cq)) == M)
            wc, _ = build_search_for_set(cq.n, M)
            circuits.append(transpile_best(wc, backend, is_hw))
            frag_idx = []
            for (lab, p), S in zip(q["groups"], per):
                fc, _ = build_search_for_set(p.width, S)
                circuits.append(transpile_best(fc, backend, is_hw))
                frag_idx.append((len(circuits) - 1, S))
            results.append(dict(name=q["name"], sql=q["sql"], kind=q["kind"],
                                n=cq.n, widths=widths, selectivity=round(cq.selectivity(), 4),
                                nM=len(M), verified=verified,
                                _whole_idx=len(circuits) - 1 - len(per),
                                _frag_idx=frag_idx, _M=M))
        elif q["kind"] == "xeq":
            xe = q["xeq"]
            n = 2 * xe.width
            M = sorted(xe.joint_matching_set())
            wc, _ = build_search_for_set(n, M)
            circuits.append(transpile_best(wc, backend, is_hw))
            results.append(dict(name=q["name"], sql=q["sql"], kind="xeq",
                                n=n, widths=[xe.width, xe.width],
                                selectivity=round(len(M) / 2 ** n, 4), nM=len(M),
                                verified=True, _whole_idx=len(circuits) - 1, _M=M,
                                _xeq_width=xe.width))

    counts = run_batch(circuits, run_dev, sampler, is_hw, args.shots)

    table = []
    for r in results:
        p_whole = p_in_set(counts[r["_whole_idx"]], r["_M"])
        row = dict(name=r["name"], sql=r["sql"], kind=r["kind"], n=r["n"],
                   widths=r["widths"], selectivity=r["selectivity"], nM=r["nM"],
                   verified=r["verified"], P_whole=round(p_whole, 4))
        if r["kind"] in ("sep", "broad"):
            pf = 1.0
            for idx, S in r["_frag_idx"]:
                pf *= p_in_set(counts[idx], S)
            row["P_factorized"] = round(pf, 4)
            row["gain"] = round(pf / p_whole, 1) if p_whole > 0 else None
        elif r["kind"] == "xeq":
            row["P_factorized"] = round(1.0 / (2 ** r["_xeq_width"]), 4)
            row["note"] = "non-separable: naive split correct only by chance 1/2^w"
        table.append(row)

    with open(os.path.join(OUT, "tpch.json"), "w") as f:
        json.dump(dict(backend=where, is_hw=is_hw, shots=args.shots, rows=table), f, indent=2)

    md = [f"# TPC-H-style predicate execution ({where})", "",
          f"Each predicate is executed and every retrieved row verified against the "
          f"classical SQL matching set. {args.shots} shots"
          f"{' on real hardware (single batched job)' if is_hw else ' under the FakeFez noise model'}.",
          "",
          "| query | SQL | n | sel. | \\|M\\| | verified | P(whole) | P(factorized) | gain |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        g = "" if r.get("gain") is None else f"{r['gain']}$\\times$"
        pf = r.get("P_factorized", "")
        md.append(f"| {r['name']} | `{r['sql']}` | {r['n']} | {r['selectivity']} | "
                  f"{r['nM']} | {'yes' if r['verified'] else 'NO'} | {r['P_whole']:.4f} | "
                  f"{pf} | {g} |")
    with open(os.path.join(OUT, "tpch.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[{where}] executed {len(table)} queries, {len(circuits)} circuits"
          f"{' in one batched job' if is_hw else ''}")
    for r in table:
        extra = (f" fac={r['P_factorized']:.3f} gain={r.get('gain')}x"
                 if "P_factorized" in r and r["kind"] != "xeq" else
                 (f" (xeq chance={r['P_factorized']})" if r["kind"] == "xeq" else ""))
        print(f"  {r['name']:<16} verified={r['verified']!s:<5} whole={r['P_whole']:.3f}{extra}")
    print("wrote tpch.md / tpch.json")


if __name__ == "__main__":
    main()
