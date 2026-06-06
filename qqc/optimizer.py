"""Sweep subcircuit width to minimize the closed-form cut-plan cost."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .cut_decision import GAMMA2_WIRE_CC


def cuts_for_width(n: int, w: int) -> int:
    """Estimated cuts to split an n-wide circuit into width-<=w parts (one per boundary)."""
    if w >= n:
        return 0
    m = math.ceil(n / w)
    return m - 1


def parts_for_width(n: int, w: int) -> int:
    return max(1, math.ceil(n / w))


@dataclass
class PlanPoint:
    width: int
    parts: int
    cuts: int
    log10_F_sub: float
    log10_cost: float
    feasible: bool


@dataclass
class OptResult:
    best: PlanPoint
    intact: PlanPoint
    curve: list[PlanPoint]
    cut: bool

    def summary(self) -> str:
        b, i = self.best, self.intact
        head = "CUT" if self.cut else "INTACT"
        return (
            f"decision        : {head}\n"
            f"intact  w={i.width:2d}  log10C={i.log10_cost:7.3f}  log10F={i.log10_F_sub:7.3f}\n"
            f"optimum w={b.width:2d}  parts={b.parts}  cuts={b.cuts}  "
            f"log10C={b.log10_cost:7.3f}  log10F_sub={b.log10_F_sub:7.3f}\n"
            f"cost reduction  : 10^{i.log10_cost - b.log10_cost:.3f}x fewer shots-to-target"
        )


def optimize_cut_plan(n: int,
                      island_capacity: int,
                      log10_F_of_width: Callable[[int], float],
                      gamma2_per_cut: float = GAMMA2_WIRE_CC,
                      min_width: int = 2) -> OptResult:
    """Sweep subcircuit width and return the minimum-cost feasible plan.

    log10_F_of_width is the only place the cost model enters, so the same
    optimizer serves both the analytical predictor and a noisy profiler.
    """
    def cost_point(w: int, feasible_cap: bool) -> PlanPoint:
        parts = parts_for_width(n, w)
        cuts = cuts_for_width(n, w)
        lf = log10_F_of_width(w)
        log10_overhead = cuts * math.log10(gamma2_per_cut)
        log10_cost = log10_overhead + math.log10(parts) - 2.0 * lf
        return PlanPoint(w, parts, cuts, round(lf, 4), round(log10_cost, 4),
                         feasible_cap)

    intact = cost_point(n, True)

    curve: list[PlanPoint] = []
    best = intact
    hi = min(island_capacity, n - 1)
    for w in range(min_width, hi + 1):
        pt = cost_point(w, feasible_cap=(w <= island_capacity))
        curve.append(pt)
        if pt.feasible and pt.log10_cost < best.log10_cost:
            best = pt

    curve.append(intact)
    return OptResult(best=best, intact=intact, curve=curve,
                     cut=(best.width < n and best.log10_cost < intact.log10_cost))
