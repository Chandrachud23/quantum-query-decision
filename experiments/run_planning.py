"""Closed-form planner tables: routing penalty, selectivity, and a scaling reference."""

from __future__ import annotations

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qqc.fidelity_model import predict_log10_fidelity, routing_penalty
from qqc.cut_decision import GAMMA2_WIRE_CC
from experiments.device import load_fakefez, device_noise_profile
from experiments.workloads import profile_for_width

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def _write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {name}")


def routing(noise, widths=(3, 4, 5, 6, 7, 8)):
    rows = []
    for n in widths:
        prof = profile_for_width(n)
        fs = predict_log10_fidelity(prof["synth"], noise)["log10_F"]
        fr = predict_log10_fidelity(prof["routed"], noise)["log10_F"]
        pen = round(fs - fr, 3)
        rows.append({"n": n, "n2_synth": prof["synth"].n_2q, "n2_routed": prof["routed"].n_2q,
                     "swap_factor": round(prof["routed"].n_2q / max(1, prof["synth"].n_2q), 2),
                     "routing_penalty": pen,
                     "rel_drop": round(1 - 10 ** (-pen) if pen > 0 else 0.0, 4)})
    md = ["# Routing dominates the fidelity loss of a search query (Heron r2, real counts)", "",
          "| n | 2Q synth | 2Q routed | SWAP factor | routing penalty | rel. drop |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['n']} | {r['n2_synth']} | {r['n2_routed']} | {r['swap_factor']} | "
                  f"{r['routing_penalty']} | {r['rel_drop']:.1%} |")
    _write("routing.md", md)
    return rows


def selectivity(noise, n_qubits=5):
    """Documented benchmark selectivities -> iterations -> routing penalty -> plan."""
    n_keys = 2 ** n_qubits
    thresh = math.log10(GAMMA2_WIRE_CC)
    # (label, source, documented selectivity, separable?)
    queries = [
        ("point lookup (key=v)", "OLTP, 1/N", 1.0 / n_keys, True),
        ("TPC-H Q6 (revenue)", "TPC-H [Dreseler'20]", 0.02, True),
        ("kNN similarity", "ANN-Benchmarks", 0.06, False),
        ("TPC-H Q1 (aggregation)", "TPC-H", 0.98, True),
    ]
    rows = []
    for label, src, sel, separable in queries:
        matches = min(n_keys - 1, max(1, round(sel * n_keys)))
        n_iter = max(1, int(round((math.pi / 4) * math.sqrt(n_keys / matches))))
        pen = routing_penalty(profile_for_width(n_qubits, n_iter=n_iter)["synth"],
                              profile_for_width(n_qubits, n_iter=n_iter)["routed"], noise)
        routing_heavy = pen > thresh
        plan = ("factorize" if (routing_heavy and separable) else
                "whole (hard)" if routing_heavy else "whole")
        rows.append({"query": label, "source": src, "selectivity": sel,
                     "routing_penalty": round(pen, 2), "plan": plan})
    md = [f"# Selectivity read off as plans ({2 ** n_qubits}-key index)", "",
          "| query | source | selectivity | routing pen. | plan |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['query']} | {r['source']} | {r['selectivity']} | "
                  f"{r['routing_penalty']} | {r['plan']} |")
    _write("selectivity.md", md)
    return rows


def scaling(noise):
    """Two regimes: width-3 groups (narrow attributes) vs a fixed two groups (wide)."""
    def F(width):
        return 10 ** predict_log10_fidelity(profile_for_width(width)["routed"], noise,
                                            saturate=True)["log10_F"]
    Fw3 = F(3)
    rows = []
    for n in (4, 6, 8, 10, 12):
        g = -(-n // 3)                                   # ceil(n/3) width-3 groups
        rows.append({"n": n, "F_monolithic": round(F(n), 4),
                     "F_width3_groups": round(Fw3 ** g, 4),
                     "F_two_groups": round(F(-(-n // 2)) ** 2, 4)})
    md = [f"# Scaling of factorization (F(3)={Fw3:.2f})", "",
          "| n | monolithic | factor (width-3 groups) | factor (2 groups) |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['n']} | {r['F_monolithic']} | {r['F_width3_groups']} | "
                  f"{r['F_two_groups']} |")
    _write("scaling.md", md)
    return {"F_width3": Fw3, "rows": rows}


def main():
    noise = device_noise_profile(load_fakefez())
    print("routing:");      r = routing(noise)
    print("selectivity:");  s = selectivity(noise)
    print("scaling:");      sc = scaling(noise)
    with open(os.path.join(OUT, "planning.json"), "w") as f:
        json.dump({"routing": r, "selectivity": s, "scaling": sc}, f, indent=2)
    print("wrote planning.json")


if __name__ == "__main__":
    main()
