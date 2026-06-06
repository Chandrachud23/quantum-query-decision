"""Closed-form log10-fidelity model from gate counts and device calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseProfile:
    """Device noise parameters (defaults track IBM Heron r2)."""
    p_2q: float = 3e-3
    p_1q: float = 3e-4
    p_ro: float = 1.6e-2
    t2_us: float = 79.0
    dur_2q_us: float = 0.068
    dur_1q_us: float = 0.032

    @property
    def log10_surv_2q(self) -> float:
        return math.log10(1.0 - self.p_2q)

    @property
    def log10_surv_1q(self) -> float:
        return math.log10(1.0 - self.p_1q)


@dataclass(frozen=True)
class CircuitProfile:
    """Gate-count summary of a (sub)circuit at one compilation stage."""
    n_2q: int
    n_1q: int
    n_qubits: int
    depth_2q: int
    depth_tot: int
    n_qubits_phys: int | None = None


def predict_log10_fidelity(circ: CircuitProfile, noise: NoiseProfile,
                           include_t2: bool = True,
                           include_readout: bool = True,
                           saturate: bool = False) -> dict:
    """Per-channel breakdown of predicted log10 fidelity (all terms <= 0).

    saturate applies the depolarizing floor F -> F*(1-1/d) + 1/d, d=2^n, for
    comparison against a success-probability metric.
    """
    log10_f_2q = circ.n_2q * noise.log10_surv_2q
    log10_f_1q = circ.n_1q * noise.log10_surv_1q
    log10_f_gate = log10_f_2q + log10_f_1q

    # Idle decoherence: spectator qubits sit idle while the critical path runs.
    depth_1q = max(0, circ.depth_tot - circ.depth_2q)
    exec_us = circ.depth_2q * noise.dur_2q_us + depth_1q * noise.dur_1q_us
    n_phys = circ.n_qubits_phys if circ.n_qubits_phys is not None else circ.n_qubits
    active_us = circ.n_2q * noise.dur_2q_us * 2 + circ.n_1q * noise.dur_1q_us
    idle_us = max(0.0, n_phys * exec_us - active_us)
    log10_f_t2 = -(idle_us / noise.t2_us) * math.log10(math.e)

    log10_f = log10_f_gate
    if include_t2:
        log10_f += log10_f_t2
    if saturate:
        d = 2 ** circ.n_qubits
        lam = 10 ** log10_f if log10_f > -300 else 0.0
        log10_f = math.log10(lam * (1.0 - 1.0 / d) + 1.0 / d)
    log10_f_ro = circ.n_qubits * math.log10(1.0 - noise.p_ro)
    if include_readout:
        log10_f += log10_f_ro

    return {
        "log10_F": round(log10_f, 4),
        "log10_F_gate": round(log10_f_gate, 4),
        "log10_F_2q": round(log10_f_2q, 4),
        "log10_F_1q": round(log10_f_1q, 4),
        "log10_F_T2": round(log10_f_t2, 4),
        "log10_F_readout": round(log10_f_ro, 4),
        "exec_us": round(exec_us, 3),
        "F": 10 ** log10_f if log10_f > -300 else 0.0,
    }


def routing_penalty(synth: CircuitProfile, routed: CircuitProfile,
                    noise: NoiseProfile) -> float:
    """Routing penalty: log10 F(synthesis) - log10 F(routed), >= 0."""
    f_syn = predict_log10_fidelity(synth, noise)["log10_F"]
    f_rt = predict_log10_fidelity(routed, noise)["log10_F"]
    return round(f_syn - f_rt, 4)
