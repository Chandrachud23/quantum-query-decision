# Factorization matrix (ibm_fez, 4096 shots, 3 layouts each)

One-shot retrieval success P(correct), mean $\pm$ spread over layouts, for monolithic vs factorized execution of conjunctive queries of varied width and group structure on real hardware (single batched job).

| query | $n$ | groups | marks | P(whole) | P(factorized) | gain |
|---|---|---|---|---|---|---|
| 3+3 | 6 | 3+3 | 1+1 | 0.0119$\pm$0.001 | **0.5753**$\pm$0.011 | 48.4$\times$ |
| 3+3 (kNN) | 6 | 3+3 | 2+1 | 0.0283$\pm$0.004 | **0.1802**$\pm$0.006 | 6.4$\times$ |
| 4+4 | 8 | 4+4 | 1+1 | 0.0048$\pm$0.000 | **0.1869**$\pm$0.001 | 38.9$\times$ |
| 3+3+3 | 9 | 3+3+3 | 1+1+1 | 0.0024$\pm$0.001 | **0.4269**$\pm$0.013 | 174.8$\times$ |
| 4+5 | 9 | 4+5 | 1+1 | 0.0020$\pm$0.001 | **0.0186**$\pm$0.001 | 9.5$\times$ |
| 3+3+3+3 | 12 | 3+3+3+3 | 1+1+1+1 | 0.0002$\pm$0.000 | **0.3235**$\pm$0.005 | 1325.2$\times$ |
| 3+4+5 | 12 | 3+4+5 | 1+1+1 | 0.0002$\pm$0.000 | **0.0166**$\pm$0.001 | 68.0$\times$ |

