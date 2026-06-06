# Decision cost: closed-form criterion vs noisy-sim profiling

Evaluating widths 2..8 once each.

| method | wall time (s) |
|---|---|
| closed-form criterion | 1.09 |
| noisy-sim profiling | 21.5 |

**Speedup: 20x.** The gap widens with width, since simulation is exponential in qubits while the criterion is linear in gate count.

