"""Unit tests for the planner internals (pure Python, no qiskit needed).

Run:  pytest -q        (from the repo root)
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qqc.fidelity_model import (NoiseProfile, CircuitProfile, predict_log10_fidelity,
                                routing_penalty)
from qqc.optimizer import cuts_for_width, parts_for_width, optimize_cut_plan
from qqc.cut_decision import decide_cut, GAMMA2_WIRE_CC, GAMMA2_WIRE_LOCAL
from qqc.islands import find_islands
from qqc.factorize import is_separable, factorized_plan, choose_strategy

NOISE = NoiseProfile()


def prof(n2, n1=10, nq=4):
    return CircuitProfile(n_2q=n2, n_1q=n1, n_qubits=nq, depth_2q=n2, depth_tot=n2 + n1,
                          n_qubits_phys=nq)


# ---- fidelity model -------------------------------------------------------
def test_fidelity_monotone_in_2q():
    """More two-qubit gates -> strictly lower predicted fidelity."""
    f_small = predict_log10_fidelity(prof(10), NOISE)["log10_F"]
    f_big = predict_log10_fidelity(prof(100), NOISE)["log10_F"]
    assert f_big < f_small <= 0.0


def test_fidelity_in_unit_interval():
    F = predict_log10_fidelity(prof(50), NOISE)["F"]
    assert 0.0 <= F <= 1.0


def test_routing_penalty_nonnegative():
    synth, routed = prof(50), prof(120)          # routing only adds gates
    assert routing_penalty(synth, routed, NOISE) >= 0.0



# ---- optimizer geometry ---------------------------------------------------
def test_cuts_and_parts():
    assert parts_for_width(8, 3) == 3 and cuts_for_width(8, 3) == 2
    assert cuts_for_width(8, 8) == 0 and parts_for_width(8, 8) == 1
    assert cuts_for_width(8, 10) == 0          # width >= n -> no cut


def test_optimizer_cuts_when_routing_dominates():
    """If fragments are far more faithful than the whole, the planner cuts."""
    def logF(w):                                # whole (w=8) terrible, fragments fine
        return -0.05 * w if w < 8 else -3.0
    res = optimize_cut_plan(8, island_capacity=6, log10_F_of_width=logF,
                            gamma2_per_cut=GAMMA2_WIRE_CC)
    assert res.cut and res.best.width < 8


def test_optimizer_stays_whole_when_overhead_dominates():
    """A huge per-cut overhead makes cutting never worth it."""
    def logF(w):
        return -0.2 if w < 8 else -0.25         # whole is nearly as good
    res = optimize_cut_plan(8, island_capacity=6, log10_F_of_width=logF,
                            gamma2_per_cut=1e6)
    assert not res.cut and res.best.width == 8


def test_decide_cut_threshold_consistency():
    """decide_cut agrees with the sign of the Prop-1 gap."""
    intact, sub = prof(800), prof(120)
    v = decide_cut(intact, sub, num_cuts=2, noise=NOISE, n_subcircuits=3)
    assert v.cut == (v.log10_cost_cut < v.log10_cost_intact)


# ---- islands --------------------------------------------------------------
def test_islands_prune_high_error_and_split():
    """A bad middle edge splits a line into two islands."""
    cmap = [(0, 1), (1, 2), (2, 3), (3, 4)]
    edge_err = {(0, 1): 1e-3, (1, 2): 1e-3, (2, 3): 9e-1, (3, 4): 1e-3}  # (2,3) is the outlier
    ro = {q: 1e-2 for q in range(5)}
    isl = find_islands(cmap, edge_err, ro, z_thresh=1.0)
    assert max(i.size for i in isl) <= 3        # the bad edge breaks the chain
    assert all((2, 3) not in i.edges for i in isl)


# ---- factorization --------------------------------------------------------
def test_separability():
    assert is_separable([[0, 1, 2], [3, 4, 5]])          # disjoint groups
    assert not is_separable([[0, 1, 2], [2, 3]])         # overlap -> not separable
    assert not is_separable([[0, 1, 2, 3]])              # single group -> nothing to factor


def test_factorized_joint_is_product():
    """Joint log-fidelity is the sum of sub-search log-fidelities (product of F)."""
    fp = factorized_plan([-0.1, -0.1], [3, 3], [2, 2])
    assert abs(fp.log10_F_joint - (-0.2)) < 1e-9
    assert fp.log10_cost == round(0.4, 4) and fp.oracle_calls == 4


def test_factorize_chosen_when_whole_is_destroyed():
    """A routing-destroyed whole query loses to factorization with no overhead."""
    factor = factorized_plan([-0.1, -0.1], [3, 3], [2, 2])   # cost 0.4
    # whole is destroyed (log10 F = -2.5 -> cost 5.0); cut still pays gamma overhead (1.9)
    assert choose_strategy(log10_cost_whole=5.0, log10_cost_cut=1.9, factor=factor) == "factorize"


def test_no_factorize_when_not_separable():
    assert choose_strategy(5.0, 1.9, factor=None) == "cut"
