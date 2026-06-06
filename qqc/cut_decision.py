"""Compare running a circuit intact versus cutting it, on shots-to-precision cost."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fidelity_model import CircuitProfile, NoiseProfile, predict_log10_fidelity

# Per-cut sampling overhead gamma^2.
GAMMA2_WIRE_CC = 4.0       # wire cut with classical communication (Lowe 2022)
GAMMA2_WIRE_LOCAL = 16.0   # wire cut, local only (Peng 2020)
GAMMA2_GATE_CNOT = 9.0     # gate cut (Piveteau & Sutter 2023)


@dataclass
class CutVerdict:
    cut: bool
    num_cuts: int
    log10_cost_intact: float
    log10_cost_cut: float
    log10_F_intact: float
    log10_F_subcircuit: float
    reason: str


def _log10_shot_cost(log10_F: float, log10_overhead: float = 0.0) -> float:
    """log10 of sampling_overhead / F^2 (lower is cheaper)."""
    return log10_overhead - 2.0 * log10_F


def decide_cut(intact_routed: CircuitProfile,
               subcircuit_routed: CircuitProfile,
               num_cuts: int,
               noise: NoiseProfile,
               gamma2_per_cut: float = GAMMA2_WIRE_CC,
               n_subcircuits: int = 2) -> CutVerdict:
    """Decide whether cutting beats running intact, on the shot-cost yardstick."""
    f_intact = predict_log10_fidelity(intact_routed, noise)["log10_F"]
    f_sub = predict_log10_fidelity(subcircuit_routed, noise)["log10_F"]

    cost_intact = _log10_shot_cost(f_intact, 0.0)

    log10_overhead = num_cuts * math.log10(gamma2_per_cut)
    cost_cut = _log10_shot_cost(f_sub, log10_overhead) + math.log10(max(1, n_subcircuits))

    cut = cost_cut < cost_intact
    reason = (
        f"cut: classical overhead 10^{log10_overhead:.2f} for {num_cuts} cuts "
        f"is cheaper than the routing/idle fidelity penalty of the intact query"
        if cut else
        f"keep intact: {num_cuts} cuts cost 10^{log10_overhead:.2f} sampling "
        f"overhead, not repaid by the per-subcircuit fidelity gain"
    )
    return CutVerdict(
        cut=cut,
        num_cuts=num_cuts,
        log10_cost_intact=round(cost_intact, 3),
        log10_cost_cut=round(cost_cut, 3),
        log10_F_intact=f_intact,
        log10_F_subcircuit=f_sub,
        reason=reason,
    )
