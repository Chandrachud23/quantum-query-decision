# Hardware validation on ibm_fez (REAL DEVICE)

Predicted (closed form) vs measured retrieval fidelity over 10 search queries, 4096 shots, 3 repeat(s) (each re-transpiled to a fresh layout). Prediction and measurement come from the same compiled circuit; the model is parameterised from device calibration, not fitted to these measurements.

**Pearson r = 0.976** (p=1.4e-06)  ·  **Spearman rho = 0.964**

| n | iters | marks | 2Q (device) | pred log10F | meas log10F (mean±std) | P_ideal |
|---|---|---|---|---|---|---|
| 3 | 1 | 1 | 19 | -0.09 | -0.069 ± 0.002 | 0.7812 |
| 3 | 2 | 1 | 39 | -0.141 | -0.089 ± 0.007 | 0.9453 |
| 4 | 1 | 1 | 37 | -0.178 | -0.046 ± 0.003 | 0.4727 |
| 4 | 2 | 1 | 75 | -0.304 | -0.14 ± 0.003 | 0.9084 |
| 5 | 1 | 1 | 128 | -0.512 | -0.356 ± 0.013 | 0.2583 |
| 5 | 2 | 1 | 262 | -0.926 | -0.681 ± 0.014 | 0.6024 |
| 6 | 1 | 1 | 304 | -1.22 | -0.715 ± 0.042 | 0.1348 |
| 4 | 1 | 2 | 54 | -0.238 | -0.106 ± 0.01 | 0.7812 |
| 5 | 1 | 2 | 195 | -0.727 | -0.489 ± 0.02 | 0.4727 |
| 6 | 1 | 2 | 467 | -1.625 | -0.858 ± 0.04 | 0.2583 |

The deep queries show the on-device degradation: as the query deepens, routing drives the measured retrieval success toward the random floor, the loss the planner cuts to avoid.

