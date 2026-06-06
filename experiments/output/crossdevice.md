# Factorization across three physical IBM Heron devices (six-qubit conjunctive query, 4096 shots)

One batched job per device; one-shot retrieval success of the correct record.

| device | $p_{2q}$ | P(whole) | P(factorized) | gain |
|---|---|---|---|---|
| ibm_fez | 0.00532 | 0.0129 | **0.5503** | 42.5$\times$ |
| ibm_marrakesh | 0.00583 | 0.0137 | **0.6702** | 49.0$\times$ |
| ibm_kingston | 0.00654 | 0.0081 | **0.7231** | 89.7$\times$ |

