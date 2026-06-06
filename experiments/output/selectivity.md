# Selectivity read off as plans (32-key index)

| query | source | selectivity | routing pen. | plan |
|---|---|---|---|---|
| point lookup (key=v) | OLTP, 1/N | 0.03125 | 2.18 | factorize |
| TPC-H Q6 (revenue) | TPC-H [Dreseler'20] | 0.02 | 2.18 | factorize |
| kNN similarity | ANN-Benchmarks | 0.06 | 1.59 | whole (hard) |
| TPC-H Q1 (aggregation) | TPC-H | 0.98 | 0.44 | whole |
