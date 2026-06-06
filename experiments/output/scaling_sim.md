# Factorization scaling by full noisy simulation (FakeFez)

One-shot retrieval success, 8192 shots, best device layout. Monolithic for $n\ge10$ is skipped (routing penalty $>9$, success $<10^{-3}$).

| n | monolithic | factorize, 2 groups | factorize, width-3 groups |
|---|---|---|---|
| 4 | 0.6559 | 0.0653 | -- |
| 6 | 0.0171 | 0.6809 | 0.6809 |
| 8 | 0.0049 | 0.4302 | -- |
| 9 | $<10^{-3}$ | 0.1123 | 0.5619 |
| 10 | $<10^{-3}$ | 0.0293 | -- |
| 12 | $<10^{-3}$ | 0.0003 | 0.4637 |

Per-width sub-search success (measured once): F(2)=0.256, F(3)=0.825, F(4)=0.656, F(5)=0.171, F(6)=0.017, F(8)=0.005.
