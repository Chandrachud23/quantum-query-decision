# End-to-end TPC-H-style predicate execution (ibm_fez)

Each predicate is executed (oracle synthesised, amplified, run) and every retrieved row verified against the classical SQL matching set over the same toy table. 4096 shots on real hardware (single batched job).

| query | SQL | n | sel. | \|M\| | verified | P(whole) | P(factorized) | gain |
|---|---|---|---|---|---|---|---|---|
| Q6 (3+3) | `discount=5 AND quantity=2` | 6 | 0.0156 | 1 | yes | 0.0149 | 0.62 | 41.6$\times$ |
| Q6 (4+4) | `discount=9 AND quantity<2` | 8 | 0.0078 | 2 | yes | 0.0059 | 0.306 | 52.2$\times$ |
| Q6 (3+3+3) | `discount=5 AND quantity=2 AND shipdate=3` | 9 | 0.002 | 1 | yes | 0.0020 | 0.4653 | 238.2$\times$ |
| Q6 (2+2+2) | `discount=1 AND quantity=2 AND shipdate=0` | 6 | 0.0156 | 1 | yes | 0.0129 | 0.0137 | 1.1$\times$ |
| Q1 (broad) | `shipdate <= 6  (broad aggregation scan)` | 3 | 0.875 | 7 | yes | 0.4316 | 0.4343 | 1.0$\times$ |
| XEQ (colA=colB) | `colA = colB  (cross-column join key)` | 4 | 0.25 | 4 | yes | 0.1973 | 0.25 |  |

Separable Q6 queries factorize and recover; the broad Q1 stays whole and survives; the cross-column XEQ predicate is non-separable, so naive factorization is correct only by chance $1/2^w$ -- the planner keeps it whole.

