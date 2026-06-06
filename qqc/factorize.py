"""Cost and selection for predicate factorization of conjunctive queries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FactorPlan:
    separable: bool
    group_widths: list[int]
    log10_F_groups: list[float]
    log10_F_joint: float
    log10_cost: float
    oracle_calls: int


def is_separable(group_bits: list[list[int]]) -> bool:
    """Separable iff the groups touch disjoint index bits."""
    seen: set[int] = set()
    for g in group_bits:
        if seen.intersection(g):
            return False
        seen.update(g)
    return len(group_bits) >= 2


def factorized_plan(log10_F_groups: list[float],
                    group_widths: list[int],
                    oracle_calls_per_group: list[int]) -> FactorPlan:
    """Score the factorized strategy from per-sub-search fidelities (product = joint)."""
    log10_F_joint = sum(log10_F_groups)
    return FactorPlan(
        separable=True,
        group_widths=list(group_widths),
        log10_F_groups=[round(f, 4) for f in log10_F_groups],
        log10_F_joint=round(log10_F_joint, 4),
        log10_cost=round(-2.0 * log10_F_joint, 4),
        oracle_calls=sum(oracle_calls_per_group),
    )


def choose_strategy(log10_cost_whole: float,
                    log10_cost_cut: float | None,
                    factor: FactorPlan | None) -> str:
    """Pick the cheapest available strategy (lower log10_cost is cheaper)."""
    options = {"whole": log10_cost_whole}
    if log10_cost_cut is not None:
        options["cut"] = log10_cost_cut
    if factor is not None and factor.separable:
        options["factorize"] = factor.log10_cost
    return min(options, key=options.get)
