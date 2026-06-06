# Factorization vs.\ baselines for a conjunctive search (n=8, fake_fez)

A conjunctive query over a composite index ($n_A{=}4$, $n_B{=}4$). One-shot retrieval success of the correct record, 8192 shots under the FakeFez noise model.

| strategy | sub-searches | Grover iters | 2Q routed | P(correct) |
|---|---|---|---|---|
| whole (heavy-hex line) | $1{\times}8$q | 8 | 8441 | 0.0034 |
| whole (best layout) | $1{\times}8$q | 8 | 6729 | 0.0035 |
| **factorized** | $4$q$+4$q | 3+3 | $\le$111 | **0.4412** |

**Factorization wins: 124.6$\times$ over the best-layout monolithic**, with fewer oracle calls (8 vs 6) and no reconstruction overhead. Noise-adaptive placement alone does not rescue the monolithic search---the cross-group routing is intrinsic to the conjunctive oracle, and only removing it (by factorizing) recovers the query.

