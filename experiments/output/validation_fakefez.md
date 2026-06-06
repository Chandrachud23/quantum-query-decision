# Hardware validation on fake_fez (Aer noise model)

Predicted (closed form) vs measured retrieval fidelity over 6 search queries, 4096 shots, 1 repeat(s) (each re-transpiled to a fresh layout). Prediction and measurement come from the same compiled circuit; the model is parameterised from device calibration, not fitted to these measurements.

**Pearson r = 0.996** (p=2.7e-05)  ·  **Spearman rho = 1.000** (95\% CI [1.00, 1.00], $n{=}6$ queries)

Affine recalibration: meas = 0.34*pred -0.02; RMSE 0.38 -> 0.01 dex, rank order unchanged.

| n | iters | marks | 2Q (device) | pred log10F | meas log10F (mean±std) | P_ideal |
|---|---|---|---|---|---|---|
| 3 | 1 | 1 | 19 | -0.072 | -0.034 ± 0.0 | 0.7812 |
| 3 | 2 | 1 | 39 | -0.129 | -0.058 ± 0.0 | 0.9453 |
| 4 | 1 | 1 | 37 | -0.162 | -0.062 ± 0.0 | 0.4727 |
| 4 | 2 | 1 | 75 | -0.3 | -0.126 ± 0.0 | 0.9084 |
| 5 | 1 | 1 | 128 | -0.523 | -0.22 ± 0.0 | 0.2583 |
| 6 | 1 | 1 | 307 | -1.3 | -0.445 ± 0.0 | 0.1348 |

The deep queries show the on-device degradation: as the query deepens, routing drives the measured retrieval success toward the random floor, the loss the planner cuts to avoid.

