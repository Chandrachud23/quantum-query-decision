"""Demo: score whole/cut/factorize for a search query from the closed-form model."""

import math

from qqc import (
    CircuitProfile, NoiseProfile, predict_log10_fidelity,
    optimize_cut_plan, factorized_plan, choose_strategy, GAMMA2_WIRE_CC,
)


def grover_logF(width: int, n_iter: int, noise: NoiseProfile, routed: bool = True) -> float:
    """Closed-form log10 fidelity of an n-iter Grover search of the given width."""
    mcx = max(1, 2 * width - 3)
    n_2q = n_iter * 2 * mcx
    if routed:
        n_2q = int(round(n_2q * (1.6 + 0.25 * width)))
    n_1q = n_iter * 4 * width
    prof = CircuitProfile(n_2q=n_2q, n_1q=n_1q, n_qubits=width,
                          depth_2q=n_2q, depth_tot=n_2q + n_1q, n_qubits_phys=width)
    return predict_log10_fidelity(prof, noise, include_t2=False)["log10_F"]


def plan(n, groups, noise, capacity=4, label=""):
    """groups: list of (width, matches) per conjunctive predicate group; [] = non-separable."""
    n_iter = max(1, min(8, round((math.pi / 4) * math.sqrt(2 ** n))))
    F_whole = grover_logF(n, n_iter, noise)
    cost_whole = -2 * F_whole

    res = optimize_cut_plan(n, capacity, lambda w: grover_logF(w, n_iter, noise),
                            gamma2_per_cut=GAMMA2_WIRE_CC)
    cost_cut = res.best.log10_cost if res.best.width < n else None

    factor = None
    if groups:
        fis = [grover_logF(w, max(1, min(8, round((math.pi/4)*math.sqrt(2**w/m)))), noise)
               for (w, m) in groups]
        factor = factorized_plan(fis, [w for w, _ in groups],
                                 [max(1, round((math.pi/4)*math.sqrt(2**w/m))) for w, m in groups])

    choice = choose_strategy(cost_whole, cost_cut, factor)
    print(f"--- {label} (n={n}) ---")
    print(f"  whole     : log10F={F_whole:.2f}  cost={cost_whole:.2f}")
    print(f"  cut       : cost={cost_cut if cost_cut is None else round(cost_cut,2)}")
    if factor:
        print(f"  factorize : groups={factor.group_widths}  log10F_joint={factor.log10_F_joint}"
              f"  cost={factor.log10_cost}")
    else:
        print(f"  factorize : unavailable (predicate not separable)")
    print(f"  => DECISION: {choice.upper()}\n")


def main():
    noise = NoiseProfile()
    plan(6, groups=[(3, 1), (3, 1)], noise=noise,
         label="Conjunctive selection over a composite index")
    plan(6, groups=[], noise=noise, label="Single dense predicate (non-separable)")


if __name__ == "__main__":
    main()
