"""Backend calibration (FakeFez by default) to a NoiseProfile and low-noise islands."""

from __future__ import annotations

import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit_ibm_runtime.fake_provider import FakeFez

from qqc.fidelity_model import NoiseProfile
from qqc.islands import find_islands, Island


def load_fakefez():
    return FakeFez()


def device_noise_profile(backend) -> NoiseProfile:
    """Aggregate a backend's calibration into a mean-rate NoiseProfile."""
    target = backend.target
    two_q = next((g for g in ("cz", "ecr", "cx") if g in target), "cz")

    p2 = [p.error for p in target[two_q].values() if p and p.error is not None and p.error < 1.0]
    sx = [p.error for p in target["sx"].values() if p and p.error is not None and p.error < 1.0] \
        if "sx" in target else [3e-4]
    ro = [p.error for p in target["measure"].values() if p and p.error is not None and p.error < 1.0] \
        if "measure" in target else [1.6e-2]

    d2 = [p.duration for p in target[two_q].values() if p and p.duration]
    d1 = [p.duration for p in target["sx"].values() if p and p.duration] if "sx" in target else [3.2e-8]
    t2s = [q.t2 for q in target.qubit_properties if q and q.t2] if target.qubit_properties else [79e-6]

    return NoiseProfile(
        p_2q=statistics.mean(p2),
        p_1q=statistics.mean(sx),
        p_ro=statistics.mean(ro),
        t2_us=statistics.mean(t2s) * 1e6,     # s -> us
        dur_2q_us=statistics.mean(d2) * 1e6,
        dur_1q_us=statistics.mean(d1) * 1e6,
    )


def device_calibration(backend):
    """Return (coupling_map, edge_errors, qubit_ro_errors) from FakeFez."""
    target = backend.target
    two_q = next((g for g in ("cz", "ecr", "cx") if g in target), "cz")
    cmap = [tuple(e) for e in backend.coupling_map] if backend.coupling_map else []
    cmap = sorted({tuple(sorted(e)) for e in cmap})

    edge_errors = {}
    for qpair, props in target[two_q].items():
        if props and props.error is not None and props.error < 1.0:
            edge_errors[tuple(sorted(qpair))] = props.error

    qubit_ro = {}
    if "measure" in target:
        for q, props in target["measure"].items():
            if props and props.error is not None and props.error < 1.0:
                qubit_ro[q[0]] = props.error
    return cmap, edge_errors, qubit_ro


def device_islands(backend, z_thresh: float = 1.0) -> list[Island]:
    cmap, edge_errors, qubit_ro = device_calibration(backend)
    return find_islands(cmap, edge_errors, qubit_ro, z_thresh=z_thresh)


if __name__ == "__main__":
    b = load_fakefez()
    np_ = device_noise_profile(b)
    print("FakeFez noise profile:")
    print(f"  p_2q={np_.p_2q:.2e}  p_1q={np_.p_1q:.2e}  p_ro={np_.p_ro:.2e}")
    print(f"  T2={np_.t2_us:.1f}us  dur_2q={np_.dur_2q_us:.4f}us  dur_1q={np_.dur_1q_us:.4f}us")
    for z in (0.5, 1.0, 1.5, 2.0):
        isl = device_islands(b, z_thresh=z)
        caps = sorted((i.size for i in isl), reverse=True)[:8]
        print(f"  z={z}: {len(isl)} islands, top sizes {caps}, capacity {caps[0] if caps else 0}")
