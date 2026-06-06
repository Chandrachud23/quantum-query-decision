# Generality across circuit classes (FakeFez): Pearson r=0.994, Spearman rho=0.812

| class | n | routing penalty | pred log10F | meas log10F | verdict |
|---|---|---|---|---|---|
| search | 3 | 0.05 | -0.129 | -0.154 | routes cheaply |
| search | 4 | 0.34 | -0.611 | -0.446 | routes cheaply |
| search | 5 | 2.18 | -1.529 | -1.437 | routing-dominated |
| qaoa | 3 | 0.01 | -0.045 | -0.03 | routes cheaply |
| qaoa | 4 | 0.03 | -0.089 | -0.001 | routes cheaply |
| qaoa | 5 | 0.04 | -0.14 | -0.088 | routes cheaply |
| ghz | 3 | 0.0 | -0.023 | -0.02 | routes cheaply |
| ghz | 4 | 0.0 | -0.034 | -0.027 | routes cheaply |
| ghz | 5 | 0.0 | -0.046 | -0.03 | routes cheaply |
