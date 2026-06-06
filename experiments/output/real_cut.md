# Real circuit-cutting validation (qiskit-addon-cutting)

Full QPD pipeline -- `partition_problem` -> `generate_cutting_experiments` -> noisy `AerSimulator` -> `reconstruct_expectation_values`. The observable is the projector on the marked key, so its reconstructed expectation is the retrieval success P(marked). Gate cuts (LO); sampling overhead = product of the QPD basis overheads. `F = P/P_ideal`.

num_samples = 20000, shots = 4096, seed = 1.

| n | frag w | cuts | sampling overhead | P_ideal | P_intact (noisy) | P_cut (recon) | recon \|err\| | F_intact | F_cut |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 81 | 1.0 | 0.9548 | 0.9652 | 0.0348 | 0.9548 | 0.9652 |

**Reading.** At the tractable frontier the reconstruction recovers the ideal retrieval value within sampling error, confirming a real cut+reconstruct rather than a proxy. The cut count is k=O(n) because a search oracle and its diffusion are all-to-all, so the sampling overhead the optimizer scores grows quickly with width; exact reconstruction becomes intractable by small n. That is exactly the regime the cost-based planner is for: it decides whether to cut without paying the reconstruction it is reasoning about.

